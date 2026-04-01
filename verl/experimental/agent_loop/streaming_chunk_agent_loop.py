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


def _extract_solution_text(text: str) -> str:
    """去除 think 块，与 eval extract_solution_text 对齐（按最后一个 </think> 切分）。"""
    if not text:
        return ""
    text = str(text).strip()
    if "</think>" in text:
        return text.split("</think>")[-1].strip()
    return text


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
# DocQA ability 专属评分函数（与 eval/eval_sw_docqa.py 保持完全一致）
# ---------------------------------------------------------------------------

import math as _math
import string as _string
from collections import Counter as _Counter


# --- normalize_answer (doc-qa EM/F1) ---

def _normalize_answer(text: Optional[str]) -> str:
    """去标点、冠词、多余空格（与 eval normalize_answer 完全相同）。"""
    if not text:
        return ""
    text = text.lower()
    text = text.translate(str.maketrans("", "", _string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = " ".join(text.split())
    return text.strip()


def _exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    pred_norm = _normalize_answer(prediction)
    for gt in ground_truths:
        if pred_norm == _normalize_answer(str(gt)):
            return 1.0
    return 0.0


def _sub_exact_match_score(prediction: str, ground_truths: List[str]) -> float:
    """子串匹配（eval 中 doc-qa 的 primary metric）。"""
    pred_norm = _normalize_answer(prediction)
    if not pred_norm:
        return 0.0
    for gt in ground_truths:
        gt_norm = _normalize_answer(str(gt))
        if gt_norm and (gt_norm in pred_norm or pred_norm in gt_norm):
            return 1.0
    return 0.0


# --- doc-mc ---

def _extract_choice_label(text: Optional[str]) -> Optional[str]:
    """从模型输出中提取 A/B/C/D 标签（与 eval extract_choice_label 完全相同）。"""
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


def _score_docmc(prediction: str, ground_truths: List[str]) -> float:
    """与 eval score_docmc 完全对齐：提取不到 label 直接返回 0.0，不回退到 EM。"""
    if not prediction or not prediction.strip():
        return 0.0
    pred_label = _extract_choice_label(prediction)
    if pred_label is None:
        return 0.0
    gold_labels = {
        label for label in
        (_extract_choice_label(str(gt)) for gt in ground_truths)
        if label is not None
    }
    return 1.0 if pred_label in gold_labels else 0.0


# --- doc-math ---

def _is_number(s: str) -> bool:
    return bool(re.match(r"^[-+]?(\d{1,3}(,\d{3})*|(\d+))(\.\d+)?$", s))


def _round_up_to_decimal(number: float, decimals: int) -> float:
    factor = 10 ** decimals
    return _math.ceil(number * factor) / factor


def _within_eps(pred: float, gt: float) -> bool:
    eps = abs(gt) * 0.0015
    return gt - eps <= pred <= gt + eps


def _normalize_docmath_value(prediction: Optional[str]):
    """与 eval normalize_docmath_value 完全相同。"""
    if not isinstance(prediction, str):
        prediction = str(prediction) if prediction is not None else "0"

    prediction = prediction.strip().rstrip(".")

    for money in ["£", "€", "¥", "million", "billion", "thousand", "US", "USD", "RMB"]:
        prediction = prediction.replace(money, "")

    if "=" in prediction:
        prediction = prediction.split("=")[-1].strip()
    if "≈" in prediction:
        prediction = prediction.split("≈")[-1].strip()
    if "`" in prediction:
        prediction = prediction.replace("`", "")
    if "%" in prediction:
        prediction = prediction.replace("%", "")
    if "$" in prediction:
        prediction = prediction.replace("$", "")
    if "°" in prediction:
        prediction = prediction.replace("°", "")

    if prediction in ["true", "yes", "false", "no"]:
        prediction = "True" if prediction in ["true", "yes"] else "False"
    if "True" in prediction or "False" in prediction:
        prediction = "True" if "True" in prediction else "False"

    if "approximately" in prediction:
        prediction = prediction.replace("approximately", "").strip()
    if " or " in prediction:
        prediction = prediction.split(" or ")[0]

    if re.match(r"[-+]?(?:[\d,]*\.*\d+) [^0-9 ]+$", prediction):
        m = re.search(r"([-+]?(?:[\d,]*\.*\d+)) [^0-9 ]+$", prediction)
        if m:
            prediction = m.group(1)
    if re.match(r"[^0-9 ]+ [-+]?(?:[\d,]*\.*\d+)$", prediction):
        m = re.search(r"[^0-9 ]+ ([-+]?(?:[\d,]*\.*\d+))$", prediction)
        if m:
            prediction = m.group(1)
    if re.match(r"[-+]?(?:[\d,]*\.*\d+)[^\d]{1,2}$", prediction):
        m = re.search(r"([-+]?(?:[\d,]*\.*\d+))[^\d]{1,2}$", prediction)
        if m:
            prediction = m.group(1)
    if re.match(r"[^-+\d]{1,2}(?:[\d,]*\.*\d+)$", prediction):
        m = re.search(r"[^-+\d]{1,2}((?:[\d,]*\.*\d+))$", prediction)
        if m:
            prediction = m.group(1)

    if "10^" in prediction:
        prediction = re.sub(r"10\^(-?\d+)", r"_math.pow(10, \1)", prediction)
    if " x " in prediction:
        prediction = prediction.replace(" x ", "*")
    if " × " in prediction:
        prediction = prediction.replace(" × ", "*")
    if _is_number(prediction):
        prediction = prediction.replace(",", "")

    if "(a)" in prediction or "(b)" in prediction or "(c)" in prediction or "(d)" in prediction:
        m = re.search(r"\([a-d]\)", prediction)
        if m:
            prediction = '"' + m.group(0) + '"'

    if not prediction:
        prediction = "0"

    try:
        prediction = eval(prediction)  # noqa: S307
    except Exception:
        prediction = 0

    if isinstance(prediction, (set, tuple)):
        prediction = list(prediction)
        if prediction:
            if isinstance(prediction[0], complex):
                prediction = [v.real for v in prediction]
    elif isinstance(prediction, list):
        pass
    else:
        if isinstance(prediction, complex):
            prediction = prediction.real

    return prediction


def _compare_two_numbers(prediction, ground_truth) -> bool:
    """与 eval compare_two_numbers 完全相同。"""
    if not isinstance(prediction, (int, float)):
        return False
    try:
        v1 = max(abs(ground_truth), abs(prediction))
        v2 = min(abs(ground_truth), abs(prediction))
        if (v1 != 0 and v2 != 0) and int(_math.log10(v1) - _math.log10(v2)) == (_math.log10(v1) - _math.log10(v2)):
            return True
        if v2 <= v1 / 50 and _within_eps(pred=v2 * 100, gt=v1):
            return True
        if v2 <= v1 / 500 and _within_eps(pred=v2 * 1000, gt=v1):
            return True
        if v2 <= v1 / 50000 and _within_eps(pred=v2 * 100000, gt=v1):
            return True
        if _round_up_to_decimal(v1, 3) == _round_up_to_decimal(v2, 3):
            return True
        return _within_eps(pred=prediction, gt=ground_truth)
    except (OverflowError, ValueError):
        return False


def _score_docmath(prediction: str, ground_truth: str) -> float:
    """与 eval score_docmath_qwen 完全相同。"""
    if not prediction:
        return 0.0
    pred_value = _normalize_docmath_value(prediction)
    gt_value = _normalize_docmath_value(ground_truth)
    answer_type = type(gt_value).__name__
    if answer_type == "bool":
        return 1.0 if pred_value == gt_value else 0.0
    if answer_type in ["int", "float", "float64"]:
        return 1.0 if _compare_two_numbers(pred_value, gt_value) else 0.0
    return 0.0


# --- 统一入口 ---

def _parse_docqa_answer(response: Optional[str]) -> Optional[str]:
    """与 eval parse_docqa_answer 完全对齐：先找 \\boxed{}，再找 'the answer is' 后缀。"""
    text = _extract_solution_text(response or "").replace("*", "")
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        val = m.group(1).strip()
        if val:
            return val
    lowered = text.lower()
    marker = "the answer is"
    if marker not in lowered:
        return None
    idx = lowered.rfind(marker)
    answer = text[idx + len(marker):].strip()
    answer = answer.replace("<｜Assistant｜>", "").replace("<｜end▁of▁sentence｜>", "")
    return answer.strip().strip(".").strip() or None


def _parse_docmc_answer(response: Optional[str]) -> Optional[str]:
    """与 eval parse_docmc_answer 完全对齐：boxed→label，再找 'The correct answer is'，再全文 label。"""
    text = _extract_solution_text(response or "").replace("*", "")
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        val = m.group(1).strip()
        label = _extract_choice_label(val)
        if label:
            return label
    m = re.search(r"The correct answer is \(([A-D])\)", text)
    if m:
        return m.group(1)
    m = re.search(r"The correct answer is ([A-D])", text)
    if m:
        return m.group(1)
    return _extract_choice_label(text)


def _parse_docmath_answer(response: Optional[str]) -> Optional[str]:
    """与 eval parse_docmath_answer 完全对齐：boxed 内容清洗前缀符号，再找 'the answer is <number>'。"""
    text = _extract_solution_text(response or "").replace("*", "")
    m = re.search(r"\\boxed\{([^}]+)\}", text)
    if m:
        val = m.group(1).strip().replace(",", "")
        val = re.sub(r"^[=≈`%\$°£€¥]+", "", val).rstrip(".")
        if val:
            return val
    m = re.search(
        r"the answer is ([=≈`%\$°£€¥]?-?[0-9\.,]+)",
        text,
        re.IGNORECASE,
    )
    if not m:
        return None
    return m.group(1).replace(",", "").rstrip(".")


_DOCQA_GOLD_PATTERNS = (
    re.compile(r"^Therefore, the answer is (?P<answer>.*)\.$"),
    re.compile(r"^The correct answer is (?P<answer>.*)\.$"),
)


def _parse_gold_answer(raw: str) -> str:
    """从 reward_model ground_truth 原始字符串中提取答案跨度。

    与 eval parse_docqa_gold 完全对齐：
      "Therefore, the answer is 670."  → "670"
      "The correct answer is (B)."     → "(B)"
      其他（已是裸答案）               → 原样返回
    """
    raw = (raw or "").strip()
    for pattern in _DOCQA_GOLD_PATTERNS:
        m = pattern.fullmatch(raw)
        if m:
            return m.group("answer").strip()
    return raw


def _build_answer_aliases(ability: str, answer: str) -> List[str]:
    """为 ground truth 构建别名列表，与 eval build_answer_aliases 对齐。

    doc-mc 扩充 label / (label) / "The correct answer is (label)." 三种形式；
    其他 ability 直接返回单元素列表。
    """
    aliases: List[str] = [answer]
    if ability == "doc-mc":
        label = _extract_choice_label(answer)
        if label:
            for extra in [label, f"({label})", f"The correct answer is ({label})."]:
                if extra not in aliases:
                    aliases.append(extra)
    return aliases


def _compute_reward(ability: str, pred_text: str, ground_truths: List[str]) -> float:
    """与 eval compute_sample_metrics 完全对齐。

    流程：
    1. 用 _parse_gold_answer 将原始 ground_truth（如 "Therefore, the answer is 670."）
       解析为纯答案字符串（"670"），与 eval parse_docqa_gold 保持一致。
    2. 用 _build_answer_aliases 为 doc-mc 扩充别名列表。
    3. 按 ability 调用专属提取函数 + 专属评分函数：
       - doc-qa:   parse_docqa_answer → sub_exact_match_score
       - doc-mc:   parse_docmc_answer → score_docmc（无 label 直接 0.0）
       - doc-math: parse_docmath_answer → normalize_docmath_value + compare_two_numbers
       - 其他/空:   回退到通用 _is_correct（hotpotqa 等）
    """
    if not ground_truths:
        return 0.0

    gold_answer = _parse_gold_answer(ground_truths[0])
    parsed_gts = _build_answer_aliases(ability, gold_answer)

    if ability == "doc-qa":
        prediction = _parse_docqa_answer(pred_text) or ""
        return _sub_exact_match_score(prediction, parsed_gts)

    if ability == "doc-mc":
        prediction = _parse_docmc_answer(pred_text) or ""
        return _score_docmc(prediction, parsed_gts)

    if ability == "doc-math":
        prediction = _parse_docmath_answer(pred_text) or ""
        return _score_docmath(prediction, gold_answer)

    # 通用回退（hotpotqa 等）
    return 1.0 if _is_correct(pred_text, ground_truths) else 0.0

def _apply_overlong_reward(
    reward: float,
    *,
    response_length: int,
    max_resp_len: int,
    overlong_buffer_cfg: Any,
) -> tuple[float, float]:
    """按 DAPORewardManager 同款公式叠加 overlong reward shaping。"""
    if overlong_buffer_cfg is None or not getattr(overlong_buffer_cfg, "enable", False):
        return reward, 0.0

    overlong_buffer_len = int(getattr(overlong_buffer_cfg, "len", 0) or 0)
    overlong_penalty_factor = float(getattr(overlong_buffer_cfg, "penalty_factor", 0.0) or 0.0)
    if overlong_buffer_len <= 0 or overlong_penalty_factor <= 0 or max_resp_len <= 0:
        return reward, 0.0

    if max_resp_len < overlong_buffer_len:
        _dbg(
            "overlong_buffer.len is larger than max_resp_len; "
            f"clamping len from {overlong_buffer_len} to {max_resp_len}"
        )
        overlong_buffer_len = max_resp_len

    expected_len = max_resp_len - overlong_buffer_len
    exceed_len = response_length - expected_len
    overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0.0)
    return reward + overlong_reward, overlong_reward


def _apply_avg_turn_length_penalty(
    reward: float,
    *,
    total_gen_tokens: int,
    num_turns: int,
    penalty_start: int,
    penalty_max: int,
    penalty_factor: float,
) -> tuple[float, float, float]:
    """按平均单轮生成长度叠加 reward shaping。

    当 avg_turn_tokens > penalty_start 时开始线性惩罚；
    当 avg_turn_tokens >= penalty_max 时达到最大惩罚 penalty_factor。
    """
    if num_turns <= 0:
        return reward, 0.0, 0.0

    avg_turn_tokens = total_gen_tokens / num_turns
    if penalty_factor <= 0 or penalty_start <= 0:
        return reward, 0.0, avg_turn_tokens

    if penalty_max <= penalty_start:
        penalty_max = penalty_start

    if avg_turn_tokens <= penalty_start:
        return reward, 0.0, avg_turn_tokens

    if penalty_max == penalty_start:
        avg_turn_penalty = -penalty_factor
    else:
        penalty_ratio = min((avg_turn_tokens - penalty_start) / (penalty_max - penalty_start), 1.0)
        avg_turn_penalty = -penalty_ratio * penalty_factor

    return reward + avg_turn_penalty, avg_turn_penalty, avg_turn_tokens


@register("streaming_chunk_agent")
class StreamingChunkAgentLoop(AgentLoopBase):
    """固定流程分段阅读 agent loop（无工具）"""

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        t0 = time.perf_counter()
        _dbg(f"ENTER run() file={__file__}")
        rollout_cfg = self.config.actor_rollout_ref.rollout
        max_model_len = int(rollout_cfg.max_model_len or (rollout_cfg.prompt_length + rollout_cfg.response_length))
        response_len_cap = int(rollout_cfg.response_length)
        per_turn_response_cap = int(os.getenv("VERL_SW_MAX_NEW_PER_CALL", str(response_len_cap)) or response_len_cap)
        if per_turn_response_cap <= 0:
            per_turn_response_cap = response_len_cap
        per_turn_response_cap = min(per_turn_response_cap, response_len_cap)
        avg_turn_penalty_start = int(os.getenv("VERL_SW_AVG_TURN_PENALTY_START", "500") or 500)
        avg_turn_penalty_max = int(os.getenv("VERL_SW_AVG_TURN_PENALTY_MAX", "1000") or 1000)
        avg_turn_penalty_factor = float(os.getenv("VERL_SW_AVG_TURN_PENALTY_FACTOR", "1.0") or 1.0)

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
            f"chunk_tokens={chunk_tokens} summary_interval={summary_interval} "
            f"chunks={total_chunks} per_turn_response_cap={per_turn_response_cap} "
            f"avg_turn_penalty=[start={avg_turn_penalty_start}, max={avg_turn_penalty_max}, factor={avg_turn_penalty_factor}]"
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
            turn_response_budget = max(1, min(int(remaining_model_ctx), per_turn_response_cap))
            call_params["max_tokens"] = turn_response_budget
            
            t_req0 = time.perf_counter()
            out = await self.server_manager.generate(request_id=request_id, prompt_ids=input_ids, sampling_params=call_params)
            
            # 5. 记录生成结果
            gen_ids = out.token_ids
            output_len = len(gen_ids)
            total_gen_tokens += output_len
            
            # 计算当前已占用的 Response 空间
            current_res_usage = len(total_response_ids) + output_len
            
            _dbg(
                f"--- [TURN {gen_idx}] Done | Output: {output_len} tokens "
                f"| Turn_Cap: {turn_response_budget} "
                f"| Cumul_Gen: {total_gen_tokens} "
                f"| Res_Buffer: {current_res_usage}/{response_len_cap} ---"
            )
            if output_len >= turn_response_budget:
                _dbg(
                    f"⚠️ TURN {gen_idx} hit per-turn response cap "
                    f"({turn_response_budget}); stopping this round of generation."
                )
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

        base_reward_score = _compute_reward(ability, final_answer, gt_list)
        response_length = min(total_gen_tokens, response_len_cap)
        max_resp_len = int(getattr(getattr(self.config, "data", None), "max_response_length", response_len_cap))
        overlong_buffer_cfg = getattr(getattr(self.config, "reward_model", None), "overlong_buffer", None)
        reward_score, overlong_reward = _apply_overlong_reward(
            base_reward_score,
            response_length=response_length,
            max_resp_len=max_resp_len,
            overlong_buffer_cfg=overlong_buffer_cfg,
        )
        reward_score, avg_turn_penalty, avg_turn_tokens = _apply_avg_turn_length_penalty(
            reward_score,
            total_gen_tokens=total_gen_tokens,
            num_turns=gen_idx,
            penalty_start=avg_turn_penalty_start,
            penalty_max=avg_turn_penalty_max,
            penalty_factor=avg_turn_penalty_factor,
        )
        _dbg(
            f"EXIT run() ability={ability!r} "
            f"base_reward={base_reward_score} "
            f"overlong_reward={overlong_reward} "
            f"avg_turn_tokens={avg_turn_tokens:.2f} "
            f"avg_turn_penalty={avg_turn_penalty} "
            f"reward={reward_score} "
            f"response_len={response_length}/{max_resp_len} "
            f"total_gen_tokens={total_gen_tokens}"
        )

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
            metrics={
                "avg_turn_tokens": avg_turn_tokens,
                "avg_turn_penalty": avg_turn_penalty,
                "overlong_reward": overlong_reward,
            },
            reward_score=reward_score,
        )
