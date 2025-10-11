#!/bin/bash
# 启动TensorBoard服务器

echo "启动TensorBoard服务器..."
echo "日志目录: /data/tensorboard"
echo "访问地址: http://localhost:6006"

# 启动TensorBoard
tensorboard --logdir=/data/tensorboard --port=6006 --bind_all
