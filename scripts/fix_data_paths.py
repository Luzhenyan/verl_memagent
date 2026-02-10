#!/usr/bin/env python3
"""
修复数据文件中的路径问题
将相对路径更新为绝对路径
"""

import pandas as pd
import os

def fix_data_paths():
    """修复数据文件中的路径"""
    
    # 训练数据
    train_file = "/home/wangyicheng/data/triviaqa_docs/train_small.parquet"
    if os.path.exists(train_file):
        print(f"修复训练数据: {train_file}")
        df_train = pd.read_parquet(train_file)
        
        # 更新document_file路径
        df_train['document_file'] = df_train['document_file'].apply(
            lambda x: x.replace('data/triviaqa_docs/', '/home/wangyicheng/data/triviaqa_docs/')
        )
        
        # 保存修复后的数据
        df_train.to_parquet(train_file, index=False)
        print(f"训练数据修复完成，样本数: {len(df_train)}")
        print(f"第一行document_file: {df_train['document_file'].iloc[0]}")
    
    # 验证数据
    val_file = "/home/wangyicheng/data/triviaqa_docs/val.parquet"
    if os.path.exists(val_file):
        print(f"\n修复验证数据: {val_file}")
        df_val = pd.read_parquet(val_file)
        
        # 更新document_file路径
        df_val['document_file'] = df_val['document_file'].apply(
            lambda x: x.replace('data/triviaqa_docs/', '/home/wangyicheng/data/triviaqa_docs/')
        )
        
        # 保存修复后的数据
        df_val.to_parquet(val_file, index=False)
        print(f"验证数据修复完成，样本数: {len(df_val)}")
        print(f"第一行document_file: {df_val['document_file'].iloc[0]}")

if __name__ == "__main__":
    fix_data_paths()
    print("\n路径修复完成！")






