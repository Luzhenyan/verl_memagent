#!/bin/bash
# 清空所有summary文件的脚本

echo "正在清空所有summary文件..."

# 清空/user/luzhenyan目录下的所有summary文件
if [ -d "/user/luzhenyan" ]; then
    # 删除所有summary文件
    rm -f /user/luzhenyan/document_*_summary.txt
    echo "已清空 /user/luzhenyan/ 目录下的所有summary文件"
    
    # 重新创建空的summary文件
    for i in {0..23}; do
        touch /user/luzhenyan/document_${i}_summary.txt
    done
    echo "已重新创建24个空的summary文件"
else
    echo "创建 /user/luzhenyan 目录..."
    mkdir -p /user/luzhenyan
    
    # 创建空的summary文件
    for i in {0..23}; do
        touch /user/luzhenyan/document_${i}_summary.txt
    done
    echo "已创建24个空的summary文件"
fi

echo "Summary文件清空完成！"
echo "文件列表："
ls -la /user/luzhenyan/document_*_summary.txt


