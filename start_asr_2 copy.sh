#!/bin/bash
set -e  # 遇到错误立即退出

# 定义启动命令，用nohup+&实现后台运行，同时输出日志便于排查
echo "启动 Qwen3-ASR 服务..."
CUDA_VISIBLE_DEVICES=2 qwen-asr-serve \
    /data/shared/Qwen3-ASR \
    --gpu-memory-utilization 0.8 \
    --host 0.0.0.0 \
    --max_model_len=2048 \
    --port 80