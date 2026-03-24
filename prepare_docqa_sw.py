"""
prepare_docqa_sw.py - 将 DocQA RL parquet 转换为 SW 训练格式

将 DocQA_RL_1.6K_train.parquet（包含 doc-qa/doc-mc/doc-math 三种 ability）
转换为与 hotpotqa_train_32k_sw.parquet 兼容的训练格式。

转换逻辑：
  - 从 prompt[0]["content"] 中解析出 context 和 question
  - 从 reward_model["ground_truth"] 中解析并规范化答案
  - 按 ability 构建 answer aliases
  - 输出包含 streaming_chunk_agent 所需字段的 parquet

用法：
    python prepare_docqa_sw.py \
        --input /var/luzhenyan/data/DocQA_RL_1.6K_train.parquet \
        --output /var/luzhenyan/data/docqa_train_sw.parquet

    # 同时处理 train + test：
    python prepare_docqa_sw.py \
        --input /var/luzhenyan/data/DocQA_RL_1.6K_train.parquet \
        --output /var/luzhenyan/data/docqa_train_sw.parquet \
        --val_input /var/luzhenyan/data/DocQA_RL_1.6K_test.parquet \
        --val_output /var/luzhenyan/data/docqa_val_sw.parquet
"""

import argparse
import re
import string
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 复用 eval_sw_docqa.py 中的解析逻辑
# ---------------------------------------------------------------------------

DOCQA_PROMPT_PATTERN = re.compile(
    r"<text>\n(?P<context>.*)\n</text>\n\n(?P<question>.*?)\n\n"
    r"Format your response as follows:",
    re.S,
)

DOCQA_GOLD_PATTERNS = (
    re.compile(r"^Therefore, the answer is (?P<answer>.*)\.$"),
    re.compile(r"^The correct answer is (?P<answer>.*)\.$"),
)


def extract_prompt_content(prompt_value: Any) -> str:
    """从 RL 格式的 prompt 字段提取文本内容。"""
    if hasattr(prompt_value, "tolist"):
        prompt_value = prompt_value.tolist()
    if isinstance(prompt_value, dict):
        return str(prompt_value.get("content", ""))
    if isinstance(prompt_value, (list, tuple)):
        if not prompt_value:
            return ""
        first = prompt_value[0]
        if isinstance(first, dict):
            return str(first.get("content", ""))
        return str(first)
    return str(prompt_value)


def parse_docqa_prompt(prompt_text: str) -> Optional[Tuple[str, str]]:
    """从原始 prompt 中分离 context 和 question。"""
    match = DOCQA_PROMPT_PATTERN.search(prompt_text)
    if not match:
        return None
    context = match.group("context").strip()
    question = match.group("question").strip()
    return context, question


def parse_docqa_gold(reward_model: Any) -> str:
    """从 reward_model['ground_truth'] 中提取答案文本。"""
    if isinstance(reward_model, dict):
        gold_text = str(reward_model.get("ground_truth", "")).strip()
    else:
        gold_text = str(reward_model or "").strip()

    for pattern in DOCQA_GOLD_PATTERNS:
        match = pattern.fullmatch(gold_text)
        if match:
            return match.group("answer").strip()
    return gold_text


def dedupe_keep_order(values: List[str]) -> List[str]:
    seen: set = set()
    output: List[str] = []
    for value in values:
        if value is None:
            continue
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def extract_choice_label(text: Optional[str]) -> Optional[str]:
    """从 doc-mc 答案中提取 A/B/C/D 标签。"""
    if not text:
        return None
    text = str(text).strip()
    fullmatch_patterns = (
        r"^\(?([A-D])\)?\.?$",
        r"^The correct answer is\s*\(?([A-D])\)?\.?$",
        r"^Answer\s*[:：]?\s*\(?([A-D])\)?\.?$",
    )
    for pattern in fullmatch_patterns:
        match = re.fullmatch(pattern, text, flags=re.I)
        if match:
            return match.group(1).upper()
    match = re.search(r"\(([A-D])\)", text, flags=re.I)
    if match:
        return match.group(1).upper()
    match = re.search(r"\b([A-D])\b", text, flags=re.I)
    if match:
        return match.group(1).upper()
    return None


def _extract_number(text: Optional[str]) -> Optional[float]:
    if not text:
        return None
    text = str(text).strip().replace(",", "").replace("$", "").replace("%", "")
    try:
        return float(text)
    except ValueError:
        pass
    match = re.search(r"-?\d+\.?\d*", text)
    if match:
        try:
            return float(match.group())
        except ValueError:
            return None
    return None


def build_answer_aliases(ability: str, answer: str) -> List[str]:
    """按 ability 构建答案别名列表，用于奖励函数匹配。"""
    aliases = [answer]
    if ability == "doc-mc":
        label = extract_choice_label(answer)
        if label:
            aliases.extend([
                label,
                f"({label})",
                f"The correct answer is ({label}).",
                f"The correct answer is {label}.",
            ])
    elif ability == "doc-math":
        num = _extract_number(answer)
        if num is not None:
            aliases.append(str(num))
            if float(num).is_integer():
                aliases.append(str(int(num)))
    return dedupe_keep_order(aliases)


# ---------------------------------------------------------------------------
# 转换主逻辑
# ---------------------------------------------------------------------------

# 与 eval/memagent_eval/config.py 中的 RECURRENT_CHUNK_SIZE 保持一致
DEFAULT_CHUNK_TOKENS = 1200
DEFAULT_SUMMARY_INTERVAL = 3


def convert_docqa_to_sw(
    df: pd.DataFrame,
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS,
    summary_interval: int = DEFAULT_SUMMARY_INTERVAL,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    将 DocQA RL 格式的 DataFrame 转换为 SW 训练格式。

    输入列：data_source, prompt, ability, reward_model, extra_info
    输出列：data_source, prompt, context, ability, reward_model, extra_info, agent_name

    分块方式使用 chunk_tokens（token 数），与 eval query_runner_sw.py 中的
    RECURRENT_CHUNK_SIZE=1600 保持一致，agent loop 会按 token 数切分 context。
    """
    supported_abilities = {"doc-qa", "doc-mc", "doc-math"}

    records = []
    stats = {
        "raw_rows": len(df),
        "skipped_ability": 0,
        "parse_failed": 0,
        "converted": 0,
        "ability_counts": {},
    }

    for row_idx, (_, row) in enumerate(df.iterrows()):
        ability = str(row.get("ability", "")).strip()
        if ability not in supported_abilities:
            stats["skipped_ability"] += 1
            continue

        prompt_text = extract_prompt_content(row.get("prompt"))
        parsed = parse_docqa_prompt(prompt_text)
        if parsed is None:
            stats["parse_failed"] += 1
            if verbose:
                print(f"[WARN] row {row_idx}: 解析 prompt 失败，跳过")
            continue

        context, question = parsed
        gold_answer = parse_docqa_gold(row.get("reward_model"))
        aliases = build_answer_aliases(ability, gold_answer)

        extra_info_raw = row.get("extra_info")
        if not isinstance(extra_info_raw, dict):
            extra_info_raw = {}

        original_index = extra_info_raw.get("index", row_idx)
        input_length = extra_info_raw.get("input_length")
        reasoning_hop = extra_info_raw.get("reasoning_hop")

        new_extra_info = {
            "index": original_index,
            "question": question,
            "ability": ability,
            "chunk_tokens": chunk_tokens,   # 按 token 数分块，与 eval RECURRENT_CHUNK_SIZE=1600 一致
            "summary_interval": summary_interval,
        }
        if input_length is not None:
            new_extra_info["input_length"] = input_length
        if reasoning_hop is not None:
            new_extra_info["reasoning_hop"] = reasoning_hop

        new_reward_model = {
            "ground_truth": aliases,
            "style": "rule",
        }

        new_prompt = [{"role": "user", "content": question}]

        records.append({
            "data_source": str(row.get("data_source", "")),
            "prompt": new_prompt,
            "context": context,
            "ability": ability,
            "reward_model": new_reward_model,
            "extra_info": new_extra_info,
            "agent_name": "streaming_chunk_agent",
        })

        stats["ability_counts"][ability] = stats["ability_counts"].get(ability, 0) + 1
        stats["converted"] += 1

    if verbose:
        print(f"转换统计：")
        print(f"  原始行数：{stats['raw_rows']}")
        print(f"  跳过（不支持 ability）：{stats['skipped_ability']}")
        print(f"  解析失败：{stats['parse_failed']}")
        print(f"  成功转换：{stats['converted']}")
        for ability_name, count in sorted(stats["ability_counts"].items()):
            print(f"    {ability_name}: {count}")

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser(description="将 DocQA RL parquet 转换为 SW 训练格式")
    parser.add_argument(
        "--input",
        type=str,
        default="/var/luzhenyan/data/DocQA_RL_1.6K_train.parquet",
        help="输入 DocQA parquet 路径（训练集）",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/var/luzhenyan/data/docqa_train_sw.parquet",
        help="输出 SW 训练格式 parquet 路径",
    )
    parser.add_argument(
        "--val_input",
        type=str,
        default=None,
        help="输入 DocQA parquet 路径（验证集，可选）",
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="/var/luzhenyan/data/docqa_val_sw.parquet",
        help="输出 SW 验证集 parquet 路径（当 --val_input 指定时生效）",
    )
    parser.add_argument(
        "--chunk_tokens",
        type=int,
        default=DEFAULT_CHUNK_TOKENS,
        help=f"每块的 token 数，与 eval RECURRENT_CHUNK_SIZE 对齐，默认 {DEFAULT_CHUNK_TOKENS}",
    )
    parser.add_argument(
        "--summary_interval",
        type=int,
        default=DEFAULT_SUMMARY_INTERVAL,
        help=f"模型触发总结的建议间隔，默认 {DEFAULT_SUMMARY_INTERVAL}",
    )
    args = parser.parse_args()

    print(f"读取训练集：{args.input}")
    df_train = pd.read_parquet(args.input)
    print(f"原始训练集行数：{len(df_train)}")

    df_out = convert_docqa_to_sw(
        df_train,
        chunk_tokens=args.chunk_tokens,
        summary_interval=args.summary_interval,
        verbose=True,
    )

    df_out.to_parquet(args.output, index=False)
    print(f"\n训练集已保存至：{args.output}（{len(df_out)} 行）")

    if args.val_input:
        print(f"\n读取验证集：{args.val_input}")
        df_val = pd.read_parquet(args.val_input)
        print(f"原始验证集行数：{len(df_val)}")
        df_val_out = convert_docqa_to_sw(
            df_val,
            chunk_tokens=args.chunk_tokens,
            summary_interval=args.summary_interval,
            verbose=True,
        )
        df_val_out.to_parquet(args.val_output, index=False)
        print(f"\n验证集已保存至：{args.val_output}（{len(df_val_out)} 行）")


if __name__ == "__main__":
    main()
