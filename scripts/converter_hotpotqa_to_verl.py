#!/usr/bin/env python3
"""
将HotpotQA评估数据集转换为VERL可用的格式
"""

import json
from pathlib import Path
from typing import List, Dict, Any
import argparse


def convert_hotpotqa_sample(sample: Dict[str, Any], doc_index: int, output_dir: Path) -> Dict[str, Any]:
    """
    将单个HotpotQA样本转换为VERL格式
    
    Args:
        sample: HotpotQA样本
        doc_index: 文档索引
        output_dir: 输出目录
    
    Returns:
        VERL格式的样本
    """
    # 提取文档内容
    context = sample.get("context", "")
    question = sample.get("input", "")
    answers = sample.get("answers", [])
    
    # 如果answers是字符串，转换为列表
    if isinstance(answers, str):
        answers = [answers]
    
    # 将文档分段（每个Document作为一个segment）
    segments = []
    documents = context.split("\n\nDocument ")
    
    for i, doc in enumerate(documents):
        if not doc.strip():
            continue
        
        # 第一个文档可能没有"Document "前缀
        if i == 0 and not doc.startswith("Document"):
            doc_text = doc.strip()
        else:
            # 添加回"Document "前缀
            doc_text = f"Document {doc}".strip()
        
        if doc_text:
            segments.append({
                "segment_id": i,
                "content": doc_text
            })
    
    # 保存文档到JSON文件
    doc_file_path = output_dir / "documents" / f"hotpotqa_doc_{doc_index}.json"
    doc_file_path.parent.mkdir(parents=True, exist_ok=True)
    
    doc_data = {
        "question": question,
        "segments": segments,
        "num_segments": len(segments)
    }
    
    with open(doc_file_path, 'w', encoding='utf-8') as f:
        json.dump(doc_data, f, ensure_ascii=False, indent=2)
    
    # 创建空的summary文件
    summary_file_path = Path("/user/luzhenyan") / f"hotpotqa_doc_{doc_index}_summary.txt"
    summary_file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_file_path, 'w', encoding='utf-8') as f:
        f.write("")
    
    # 创建VERL格式的样本
    verl_sample = {
        "data_source": "segmented_reading",
        "prompt": [
            {
                "role": "user",
                "content": (
                    f"Please read the document and answer the question: {question}\n\n"
                    f"Document information:\n"
                    f"- Document file: {str(doc_file_path)}\n"
                    f"- Total segments: {len(segments)}\n"
                    f"- Summary file: {str(summary_file_path)}\n\n"
                    f"Instructions:\n"
                    f"1. Use read_segment_file to read specific segments from the document\n"
                    f"2. After each reading, use write_summary_file to save your progress to the summary file\n"
                    f"3. Use read_summary_file to check your previous progress if needed\n\n"
                    f"Please start reading the document segment by segment."
                )
            }
        ],
        "ability": "reading_comprehension",
        "reward_model": {
            "style": "rule",
            "ground_truth": answers[0] if answers else ""
        },
        "extra_info": {
            "question": question,
            "document_file": str(doc_file_path),
            "num_segments": len(segments),
            "all_answers": answers,
            "split": "eval"
        },
        "agent_name": "tool_agent"
    }
    
    return verl_sample


def convert_hotpotqa_dataset(input_file: Path, output_file: Path, output_dir: Path):
    """
    转换整个HotpotQA数据集
    
    Args:
        input_file: 输入JSON文件路径
        output_file: 输出Parquet文件路径
        output_dir: 输出目录（用于存储文档文件）
    """
    print(f"正在加载数据集: {input_file}")
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"数据集包含 {len(data)} 个样本")
    
    # 转换每个样本
    verl_samples = []
    for i, sample in enumerate(data):
        if (i + 1) % 10 == 0:
            print(f"已处理 {i + 1}/{len(data)} 个样本")
        
        verl_sample = convert_hotpotqa_sample(sample, i, output_dir)
        verl_samples.append(verl_sample)
    
    # 保存为JSON格式（方便调试）
    json_output = output_file.with_suffix('.json')
    print(f"\n保存JSON格式到: {json_output}")
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(verl_samples, f, ensure_ascii=False, indent=2)
    
    # 保存为Parquet格式
    try:
        import pandas as pd
        import pyarrow as pa
        import pyarrow.parquet as pq
        
        print(f"保存Parquet格式到: {output_file}")
        df = pd.DataFrame(verl_samples)
        
        # 将列表和字典转换为JSON字符串
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: json.dumps(x, ensure_ascii=False) if isinstance(x, (list, dict)) else x)
        
        table = pa.Table.from_pandas(df)
        pq.write_table(table, output_file)
        print(f"✓ 成功保存 {len(verl_samples)} 个样本")
        
    except ImportError as e:
        print(f"警告: 无法导入pandas或pyarrow，跳过Parquet保存: {e}")
        print("仅保存了JSON格式")


def main():
    parser = argparse.ArgumentParser(description="转换HotpotQA数据集为VERL格式")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/luzhenyan/data/hotpotqa_eval/eval_100.json",
        help="输入JSON文件路径"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/luzhenyan/data/hotpotqa_eval/eval_verl.parquet",
        help="输出Parquet文件路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/luzhenyan/data/hotpotqa_eval",
        help="输出目录（用于存储文档文件）"
    )
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    output_dir = Path(args.output_dir)
    
    if not input_file.exists():
        print(f"错误: 输入文件不存在: {input_file}")
        return
    
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    convert_hotpotqa_dataset(input_file, output_file, output_dir)
    print("\n✓ 数据转换完成!")


if __name__ == "__main__":
    main()

