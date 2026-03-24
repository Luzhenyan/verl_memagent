"""
Streaming-chunk agent loop (no tools): 复刻 eval 的"手动分段 + 固定 prompt + summary + final answer"流程。

特点：
- 不调用任何工具（tool_config_path=None）
- 外部手动分段：从样本的 `context`（或 extra_info.context）按 chunk_size 切块
- 固定模板：TEMPLATE_FIRST / TEMPLATE_THINK / TEMPLATE_FINAL
- 模型自主决定何时总结：在回复中附带 <SUMMARY>...</SUMMARY> 块即触发总结更新
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

Output plain text notes.
Optionally, if you decide to update the running summary, append exactly one block:
<SUMMARY>
...updated running summary...
</SUMMARY>
Summary rules:
- MUST start by restating the original problem/question.
- extract ONLY information that is directly relevant to answering the problem.
- If facts answer the question, state: "Answer: [the answer]" or "Partial: [what's found]".

Meta: chunks_since_summary={chunks_since_summary}, summary_interval_hint={summary_interval_hint}, is_last_section={is_last_section}.
Heuristic: consider adding <SUMMARY> if chunks_since_summary >= summary_interval_hint, or context is getting long, or you already have enough.
Hard rule: if is_last_section=true, you MUST include a non-empty <SUMMARY> block.

Output:
"""

TEMPLATE_THINK = """Here is another section of the article. Please continue extracting:

<section>
{chunk}
</section>

Output plain text notes. If no relevant info, say so briefly.
Optionally append exactly one <SUMMARY>...</SUMMARY> block if you choose to update the running summary.
Summary rules:
- MUST start by restating the original problem/question.
- extract ONLY information that is directly relevant to answering the problem.
- If facts answer the question, state: "Answer: [the answer]" or "Partial: [what's found]".

Meta: chunks_since_summary={chunks_since_summary}, summary_interval_hint={summary_interval_hint}, is_last_section={is_last_section}.
Heuristic: consider adding <SUMMARY> if chunks_since_summary >= summary_interval_hint, or context is getting long, or you already have enough.
Hard rule: if is_last_section=true, you MUST include a non-empty <SUMMARY> block.

Output:
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


def _chunk_context_by_tokens(context: str, chunk_tokens: int, tokenizer) -> List[str]:
    """按 token 数分块，与 eval query_runner_sw.py 保持一致。

    先将整段 context encode 成 token ids，按 chunk_tokens 切分，再 decode 回文本。
    这样能保证训练和 eval 的分块粒度完全相同（默认 1600 tokens/块）。
    """
    if chunk_tokens <= 0:
        return [context]
    ids = tokenizer.encode(context)
    chunks = []
    for i in range(0, len(ids), chunk_tokens):
        chunk_text = tokenizer.decode(ids[i : i + chunk_tokens])
        chunks.append(chunk_text)
    return chunks if chunks else [context]


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


_SUMMARY_RE = re.compile(r"<SUMMARY>\s*(.*?)\s*</SUMMARY>", re.S | re.I)


def _parse_summary(text: str) -> Optional[str]:
    """从模型输出中提取 <SUMMARY>...</SUMMARY> 块，如果有则返回内容，否则返回 None。"""
    if not text:
        return None
    m = _SUMMARY_RE.search(text)
    if m:
        summary = m.group(1).strip()
        return summary if summary else None
    return None


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


# ---------------------------------------------------------------------------
# DocQA ability 专属评分函数
# ---------------------------------------------------------------------------

import string as _string
from collections import Counter as _Counter


def _normalize_qa_answer(text: Optional[str]) -> str:
    """对英文 QA 答案做规范化（去标点、冠词、多余空格）。"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", _string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def _qa_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _normalize_qa_answer(prediction).split()
    gt_tokens = _normalize_qa_answer(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = _Counter(pred_tokens) & _Counter(gt_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    return (2 * precision * recall) / (precision + recall)


def _extract_choice_label(text: Optional[str]) -> Optional[str]:
    """从 doc-mc 模型输出中提取 A/B/C/D 标签。"""
    if not text:
        return None
    text = str(text).strip()
    for pattern in (
        r"^\(?([A-D])\)?\.?$",
        r"^The correct answer is\s*\(?([A-D])\)?\.?$",
        r"^Answer\s*[:：]?\s*\(?([A-D])\)?\.?$",
    ):
        m = re.fullmatch(pattern, text, flags=re.I)
        if m:
            return m.group(1).upper()
    m = re.search(r"\(([A-D])\)", text, flags=re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([A-D])\b", text, flags=re.I)
    if m:
        return m.group(1).upper()
    return None


def _extract_number(text: Optional[str]) -> Optional[float]:
    """从字符串中提取数值。"""
    if not text:
        return None
    text = str(text).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        pass
    m = re.search(r"-?\d+\.?\d*", text)
    if m:
        try:
            return float(m.group())
        except ValueError:
            return None
    return None


def _compute_reward(ability: str, pred_text: str, ground_truths: List[str]) -> float:
    """
    根据 ability 计算 0/1 奖励分数。

    - doc-qa:   EM（精确匹配）或 F1 ≥ 0.5 均视为正确（宽松版，鼓励学习）
    - doc-mc:   字母标签精确匹配
    - doc-math: 数值相等（允许 1% 相对误差）
    - 其他/空:   回退到通用 _is_correct
    """
    if not ground_truths:
        return 0.0

    raw_pred = _extract_boxed(_strip_think_blocks(pred_text))
    if not raw_pred:
        return 0.0

    if ability == "doc-mc":
        pred_label = _extract_choice_label(raw_pred)
        if pred_label is None:
            return 0.0
        for gt in ground_truths:
            gt_label = _extract_choice_label(str(gt))
            if gt_label and pred_label == gt_label:
                return 1.0
        return 0.0

    if ability == "doc-math":
        pred_num = _extract_number(raw_pred)
        if pred_num is None:
            return 0.0
        for gt in ground_truths:
            gt_num = _extract_number(str(gt))
            if gt_num is None:
                continue
            if pred_num == gt_num:
                return 1.0
            if gt_num != 0 and abs(pred_num - gt_num) / abs(gt_num) < 0.01:
                return 1.0
            if abs(pred_num - gt_num) < 1e-6:
                return 1.0
        return 0.0

    if ability == "doc-qa":
        pred_norm = _normalize_qa_answer(raw_pred)
        for gt in ground_truths:
            gt_norm = _normalize_qa_answer(str(gt))
            if not gt_norm:
                continue
            if pred_norm == gt_norm:
                return 1.0
            if _qa_f1(raw_pred, str(gt)) >= 0.5:
                return 1.0
        return 0.0

    # 通用回退：复用原始 _is_correct
    return 1.0 if _is_correct(pred_text, ground_truths) else 0.0


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

        # ability 字段：支持 hotpotqa（空）、doc-qa/doc-mc/doc-math
        ability = str(kwargs.get("ability") or extra_info.get("ability") or "").strip()

        reward_model = kwargs.get("reward_model") or {}
        gt = reward_model.get("ground_truth", []) or extra_info.get("all_answers", []) or []
        gt_list = [str(x) for x in (gt if isinstance(gt, list) else [gt])]

        summary_interval = int(extra_info.get("summary_interval", 4))
        # 统一按 token 数分块，与 eval query_runner_sw.py 的 RECURRENT_CHUNK_SIZE 保持一致
        # hotpotqa 数据无 chunk_tokens 字段，走默认值 1600；docqa 数据显式设为 1200
        chunk_tokens = int(extra_info.get("chunk_tokens", 1600))
        max_chunks = extra_info.get("max_chunks")
        if max_chunks is not None:
            max_chunks = int(max_chunks)

        chunks = _chunk_context_by_tokens(str(context), chunk_tokens, self.tokenizer) if context else [""]
        _dbg(f"分块模式：token ({chunk_tokens} tokens/块)")

        if max_chunks is not None:
            _dbg(f"⚠️ Applying max_chunks={max_chunks} constraint (original chunks: {len(chunks)})")
            chunks = chunks[:max_chunks]

        total_chunks = len(chunks)
        _dbg(
            "parsed sample: "
            f"question_len={len(question)} context_len={len(str(context))} "
            f"ability={ability!r} gt_n={len(gt_list)} "
            f"chunk_tokens={chunk_tokens} summary_interval={summary_interval} chunks={total_chunks}"
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
        current_summary: Optional[str] = None

        is_last_section = (total_chunks == 1)
        gen_text = await _do_turn(
            TEMPLATE_FIRST.format(
                prompt=question,
                chunk=chunks[0],
                chunks_since_summary=chunks_since_summary,
                summary_interval_hint=summary_interval,
                is_last_section=is_last_section,
            ),
            label=f"chunk 1/{total_chunks}",
        )
        # 解析模型输出中的 <SUMMARY> 块
        parsed_summary = _parse_summary(gen_text)
        if parsed_summary:
            current_summary = parsed_summary
            chunks_since_summary = 0
            _dbg(f"模型在 chunk 1 触发了总结更新")

        # 2. 循环分块
        for i, chunk in enumerate(chunks[1:], start=2):
            chunks_since_summary += 1
            is_last_section = (i == total_chunks)
            gen_text = await _do_turn(
                TEMPLATE_THINK.format(
                    chunk=chunk,
                    chunks_since_summary=chunks_since_summary,
                    summary_interval_hint=summary_interval,
                    is_last_section=is_last_section,
                ),
                label=f"chunk {i}/{total_chunks}",
            )
            # 解析模型输出中的 <SUMMARY> 块
            parsed_summary = _parse_summary(gen_text)
            if parsed_summary:
                current_summary = parsed_summary
                chunks_since_summary = 0
                _dbg(f"模型在 chunk {i} 触发了总结更新")

        # 3. 最终回答（直接使用模型自行产生的 summary）
        final_answer = await _do_turn(
            TEMPLATE_FINAL.format(summary=current_summary or "No information gathered"),
            label="final-answer",
        )

        reward_score = _compute_reward(ability, final_answer, gt_list)
        _dbg(f"EXIT run() ability={ability!r} reward={reward_score} total_tokens={len(total_response_ids)}")

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
