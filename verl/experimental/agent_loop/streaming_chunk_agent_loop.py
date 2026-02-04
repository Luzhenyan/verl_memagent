"""
Streaming-chunk agent loop (no tools): 复刻 eval 的“手动分段 + 固定 prompt + summary + final answer”流程。

特点：
- 不调用任何工具（tool_config_path=None）
- 外部手动分段：从样本的 `context`（或 extra_info.context）按 chunk_size 切块
- 固定模板：TEMPLATE_FIRST / TEMPLATE_THINK / TEMPLATE_SUMMARIZE / TEMPLATE_FINAL
- 仅基于最终答案（boxed）做 0/1 reward_score（便于 GRPO 直接用）
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Dict, List, Optional
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register

logger = logging.getLogger(__file__)


def _dbg(msg: str):
    if os.getenv("VERL_SW_DEBUG", "0") in ("1", "true", "True"):
        logger.warning(f"[SW_AGENT][pid={os.getpid()}] {msg}")


TEMPLATE_FIRST = """You are presented with a problem and the first section of an article. Please read and extract ONLY information that is directly relevant to answering the problem.

<problem> 
{prompt}
</problem>

<section>
{chunk}
</section>

IMPORTANT: Only discuss information that is directly relevant to answering the problem. Ignore unrelated content.
Note: You will later be asked to provide a final answer, so extract all facts that could help (names, dates, places, etc.).

In addition to extraction, you MAY decide to update a running summary when it is helpful.
Meta: chunks_since_summary={chunks_since_summary}, summary_interval_hint={summary_interval_hint}, is_last_section={is_last_section}.

Heuristic: if chunks_since_summary >= summary_interval_hint, OR if you already have enough information, OR if the context is getting long, then set should_summarize=true.
Hard rule: if is_last_section=true, you MUST set should_summarize=true and provide a non-empty summary.

Return ONLY the following tagged blocks (no markdown, no extra text):

<NOTES>
Key facts from this section (names, dates, numbers, direct answers):
</NOTES>
<SHOULD_SUMMARIZE>
true|false
</SHOULD_SUMMARIZE>
<SUMMARY>
...updated running summary (empty if SHOULD_SUMMARIZE=false)...
</SUMMARY>

Summary rules (when should_summarize=true):
- MUST start by restating the original problem/question.
- MUST list ALL specific facts found (names, dates, numbers, connections).
- If facts answer the question, state: "Answer: [the answer]" or "Partial: [what's found]".
- MUST NOT be empty.

Output:
"""

TEMPLATE_THINK = """Here is another section of the article. Please continue extracting:

<section>
{chunk}
</section>

IMPORTANT: Only discuss information that is directly relevant to answering the problem we are working on. If this section contains no relevant information, simply state that and move on. Do NOT summarize unrelated content.

You MAY decide to update a running summary when it is helpful.
Meta: chunks_since_summary={chunks_since_summary}, summary_interval_hint={summary_interval_hint}, is_last_section={is_last_section}.

Heuristic: if chunks_since_summary >= summary_interval_hint, OR if you already have enough information, OR if the context is getting long, then set should_summarize=true.
Hard rule: if is_last_section=true, you MUST set should_summarize=true and provide a non-empty summary.

Return ONLY the following tagged blocks (no markdown, no extra text):

<NOTES>
Key facts from this section (names, dates, numbers, direct answers):
</NOTES>
<SHOULD_SUMMARIZE>
true|false
</SHOULD_SUMMARIZE>
<SUMMARY>
...updated running summary (empty if SHOULD_SUMMARIZE=false)...
</SUMMARY>

Summary rules (when should_summarize=true):
- MUST start by restating the original problem/question.
- MUST list ALL specific facts found (names, dates, numbers, connections).
- If facts answer the question, state: "Answer: [the answer]" or "Partial: [what's found]".
- MUST NOT be empty.

Output:
"""

TEMPLATE_SUMMARIZE = """Based on our discussion, provide a concise summary:

REQUIRED FORMAT:
Question: [restate the original problem]
Findings: [list ALL specific facts found: names, dates, numbers, document references]
Answer Status: [Complete/Partial/Not Found]

RULES:
- MUST start with "Question:" to restate the problem
- MUST list all relevant facts found (even from earlier sections)
- If you have enough to answer, state "Answer Status: Complete - [answer]"
- Never output empty summary - always include the question and findings

Summary:
"""

TEMPLATE_FINAL = """Based on the following summary, provide the final answer in \\boxed{{}}.

<summary>
{summary}
</summary>

INSTRUCTIONS:
- If the summary contains facts that answer the question, extract and provide the answer in \\boxed{{answer}}
- You may need to combine information from multiple documents - this is expected
- Only return empty \\boxed{{}} if the summary explicitly states no relevant information was found
- Be concise: only include the direct answer (e.g., a name, date, or number), not explanations

Your answer:
"""


def _chunk_context(context: str, chunk_size: int) -> List[str]:
    if chunk_size <= 0:
        return [context]
    return [context[i : i + chunk_size] for i in range(0, len(context), chunk_size)]


def _strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def _extract_boxed(text: str) -> str:
    if not text:
        return ""
    idx = text.rfind("\\boxed")
    if idx < 0:
        return text.strip()
    i = idx
    right = None
    open_braces = 0
    while i < len(text):
        if text[i] == "{":
            open_braces += 1
        elif text[i] == "}":
            open_braces -= 1
            if open_braces == 0:
                right = i
                break
        i += 1
    if right is None:
        return text[idx:].strip()
    return text[idx + 7 : right].strip()


def _normalize(s: str) -> str:
    s = (s or "").lower().strip()
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s\u4e00-\u9fff]", "", s)
    return s.strip()


def _is_correct(pred_text: str, ground_truths: List[str]) -> bool:
    pred = _normalize(_extract_boxed(_strip_think_blocks(pred_text)))
    if not pred:
        return False
    for gt in ground_truths:
        gt_norm = _normalize(str(gt))
        if not gt_norm:
            continue
        if pred == gt_norm or (gt_norm in pred) or (pred in gt_norm):
            return True
    return False


@register("streaming_chunk_agent")
class StreamingChunkAgentLoop(AgentLoopBase):
    """固定流程分段阅读 agent loop（无工具）"""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        t0 = time.perf_counter()
        _dbg(f"ENTER run() file={__file__}")
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_model_len = int(rollout_cfg.max_model_len or (rollout_cfg.prompt_length + rollout_cfg.response_length))
        response_len_cap = int(rollout_cfg.response_length)

        extra_info = kwargs.get("extra_info") or {}
        raw_prompt = kwargs.get("raw_prompt") or []

        question = (
            kwargs.get("question")
            or extra_info.get("question")
            or (raw_prompt[0].get("content") if raw_prompt and isinstance(raw_prompt[0], dict) else "")
            or ""
        )
        context = kwargs.get("context") or extra_info.get("context") or ""

        reward_model = kwargs.get("reward_model") or {}
        gt = reward_model.get("ground_truth", []) or extra_info.get("all_answers", []) or []
        gt_list = [str(x) for x in (gt if isinstance(gt, list) else [gt])]

        chunk_size = int(extra_info.get("chunk_size", 6400))
        summary_interval = int(extra_info.get("summary_interval", 4))
        max_chunks = extra_info.get("max_chunks")
        if max_chunks is not None:
            max_chunks = int(max_chunks)

        chunks = _chunk_context(str(context), chunk_size) if context else [""]
        if max_chunks is not None:
            _dbg(f"⚠️ Applying max_chunks={max_chunks} constraint (original chunks: {len(chunks)})")
            chunks = chunks[:max_chunks]

        total_chunks = len(chunks)
        _dbg(
            "parsed sample: "
            f"question_len={len(question)} context_len={len(str(context))} "
            f"gt_n={len(gt_list)} chunk_size={chunk_size} summary_interval={summary_interval} "
            f"chunks={total_chunks}"
        )

        messages: List[Dict[str, str]] = []
        response_mask: List[int] = []
        response_logprobs: Optional[List[float]] = [] if sampling_params.get("logprobs") else None
        total_response_ids: List[int] = []
        total_gen_tokens = 0  # 统计模型真正生成的总 token 数

        dump_io = os.getenv("VERL_SW_DUMP_IO", "0") in ("1", "true", "True")
        dump_io_first_n = int(os.getenv("VERL_SW_DUMP_IO_FIRST_N", "1"))
        dump_io_max_chars = int(os.getenv("VERL_SW_DUMP_IO_MAX_CHARS", "2000"))
        sink_keep_prefix_tokens = int(os.getenv("VERL_SW_SINK_KEEP_PREFIX_TOKENS", "256"))
        
        gen_idx = 0

        def _clip_text(s: str, max_chars: int) -> str:
            if max_chars <= 0 or len(s) <= max_chars: return s
            return s[:max_chars//2] + "\n...\n" + s[-max_chars//2:]

        async def _do_turn(user_content: str, label: str = "think") -> str:
            nonlocal gen_idx, total_response_ids, response_mask, response_logprobs, total_gen_tokens
            gen_idx += 1
            request_id = uuid4().hex
            
            # 1. 编码用户输入
            prev_len = len(self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)) if messages else 0
            messages.append({"role": "user", "content": user_content})
            full_ids = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            
            new_prompt_tokens = full_ids[prev_len:]
            if gen_idx > 1:
                total_response_ids += new_prompt_tokens
                response_mask += [0] * len(new_prompt_tokens)
                if response_logprobs is not None:
                    response_logprobs += [0.0] * len(new_prompt_tokens)

            # 2. 截断逻辑
            input_ids = full_ids
            original_input_len = len(input_ids)
            if max_model_len > 0 and len(input_ids) >= max_model_len:
                target = max_model_len - 1
                input_ids = input_ids[:sink_keep_prefix_tokens] + input_ids[-(target - sink_keep_prefix_tokens):]
            
            final_input_len = len(input_ids)
            _dbg(f"--- [TURN {gen_idx}] Stage: {label} | Context: {original_input_len} tokens (Actual sent: {final_input_len}) ---")

            # 3. 打印 IO
            if dump_io and gen_idx <= dump_io_first_n:
                prompt_text = self.tokenizer.decode(input_ids, skip_special_tokens=False)
                _dbg(f"[DUMP_IO] gen#{gen_idx} Stage={label} PROMPT:\n{_clip_text(prompt_text, dump_io_max_chars)}\n")

            # 4. 生成
            call_params = dict(sampling_params or {})
            call_params.pop("max_new_tokens", None)
            remaining_model_ctx = max_model_len - len(input_ids) - 1 if max_model_len > 0 else 8192
            call_params["max_tokens"] = max(1, int(remaining_model_ctx))
            
            t_req0 = time.perf_counter()
            out = await self.server_manager.generate(request_id=request_id, prompt_ids=input_ids, sampling_params=call_params)
            
            # 5. 记录生成结果
            gen_ids = out.token_ids
            output_len = len(gen_ids)
            total_gen_tokens += output_len
            
            # 计算当前已占用的 Response 空间
            current_res_usage = len(total_response_ids) + output_len
            
            _dbg(f"--- [TURN {gen_idx}] Done | Output: {output_len} tokens | Cumul_Gen: {total_gen_tokens} | Res_Buffer: {current_res_usage}/{response_len_cap} ---")
            if current_res_usage >= response_len_cap:
                _dbg(f"⚠️ WARNING: Res_Buffer is approaching or exceeding response_len_cap ({response_len_cap})! Truncation will occur.")

            gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
            messages.append({"role": "assistant", "content": gen_text})
            
            total_response_ids += gen_ids
            response_mask += [1] * len(gen_ids)
            if response_logprobs is not None:
                response_logprobs += (out.log_probs if out.log_probs else [0.0] * len(gen_ids))

            if dump_io and gen_idx <= dump_io_first_n:
                _dbg(f"[DUMP_IO] gen#{gen_idx} RESPONSE (tokens={len(gen_ids)}):\n{gen_text}\n")
            
            return gen_text

        # --- 运行流程 ---
        # 1. 初始思考
        chunks_since_summary = 0
        is_last_section = (total_chunks == 1)
        await _do_turn(
            TEMPLATE_FIRST.format(
                prompt=question,
                chunk=chunks[0],
                chunks_since_summary=chunks_since_summary,
                summary_interval_hint=summary_interval,
                is_last_section=is_last_section,
            ),
            label=f"chunk 1/{total_chunks}",
        )
        current_summary = ""

        # 2. 循环分块
        for i, chunk in enumerate(chunks[1:], start=2):
            chunks_since_summary += 1
            is_last_section = (i == total_chunks)
            await _do_turn(
                TEMPLATE_THINK.format(
                    chunk=chunk,
                    chunks_since_summary=chunks_since_summary,
                    summary_interval_hint=summary_interval,
                    is_last_section=is_last_section,
                ),
                label=f"chunk {i}/{total_chunks}",
            )
            
            if summary_interval > 0 and i % summary_interval == 0 and i != total_chunks:
                current_summary = await _do_turn(TEMPLATE_SUMMARIZE, label=f"inter-summary at {i}")
                chunks_since_summary = 0

        # 3. 最终总结
        current_summary = await _do_turn(TEMPLATE_SUMMARIZE, label="final-summary")

        # 4. 最终回答
        final_answer = await _do_turn(TEMPLATE_FINAL.format(summary=current_summary), label="final-answer")

        reward_score = 1.0 if _is_correct(final_answer, gt_list) else 0.0
        _dbg(f"EXIT run() reward={reward_score} total_tokens={len(total_response_ids)}")

        initial_prompt_ids = self.tokenizer.apply_chat_template(
            [
                {
                    "role": "user",
                    "content": TEMPLATE_FIRST.format(
                        prompt=question,
                        chunk=chunks[0],
                        chunks_since_summary=0,
                        summary_interval_hint=summary_interval,
                        is_last_section=(total_chunks == 1),
                    ),
                }
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        
        return AgentLoopOutput(
            prompt_ids=initial_prompt_ids,
            # 严格根据 response_len_cap 进行物理截断，防止下游 padding 溢出
            response_ids=total_response_ids[:response_len_cap],
            response_mask=response_mask[:response_len_cap],
            response_logprobs=response_logprobs[:response_len_cap] if response_logprobs else None,
            multi_modal_data={},
            num_turns=gen_idx,
            metrics={},
            reward_score=reward_score,
        )
