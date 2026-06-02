#!/bin/bash
# ============================================================
# Qwen3-ASR 容器启动脚本
# 功能：启动 Docker 容器并自动运行 Qwen3-ASR 服务
# ============================================================

set -e

# ---------- 配置项（按需修改）----------
CONTAINER_NAME="qwen3-asr_1.7-official_hermes"
IMAGE="qwenllm/qwen3-asr:latest"
HOST_PORT=8875
CONTAINER_PORT=80
MODEL_PATH="/opt/data/LLM/Qwen3-ASR-1.7B"
MODEL_MOUNT="/data/shared/Qwen3-ASR"
START_SCRIPT_HOST="/opt/data/LLM/run_qwen_asr.sh"
START_SCRIPT_CONTAINER="/start_asr.sh"
SHM_SIZE="40gb"
CUDA_DEVICE=5                  # 使用的 GPU 编号
GPU_MEM_UTIL=0.8               # GPU 显存利用率
MAX_NUM_SEQS=128               # 最大并发序列数
# ---------------------------------------

echo "============================================"
echo "  Qwen3-ASR 服务启动脚本"
echo "============================================"

# ---------- 检查旧容器 ----------
if docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    STATUS=$(docker inspect -f '{{.State.Status}}' "${CONTAINER_NAME}")
    echo "[INFO] 发现已有容器: ${CONTAINER_NAME}（状态: ${STATUS}）"

    if [ "${STATUS}" = "running" ]; then
        echo "[WARN] 容器正在运行，先停止并删除..."
        docker stop "${CONTAINER_NAME}"
    fi

    echo "[INFO] 删除旧容器..."
    docker rm "${CONTAINER_NAME}"
fi

# ---------- 检查依赖路径 ----------
if [ ! -d "${MODEL_PATH}" ]; then
    echo "[ERROR] 模型目录不存在: ${MODEL_PATH}"
    exit 1
fi

if [ ! -f "${START_SCRIPT_HOST}" ]; then
    echo "[ERROR] 启动脚本不存在: ${START_SCRIPT_HOST}"
    exit 1
fi

# ---------- 启动容器 ----------
echo "[INFO] 启动容器: ${CONTAINER_NAME} ..."

docker run --gpus all \
    --name "${CONTAINER_NAME}" \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p "${HOST_PORT}:${CONTAINER_PORT}" \
    --restart=always \
    --mount type=bind,source="${MODEL_PATH}",target="${MODEL_MOUNT}" \
    --mount type=bind,source="${START_SCRIPT_HOST}",target="${START_SCRIPT_CONTAINER}" \
    --shm-size="${SHM_SIZE}" \
    -e CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
    -d "${IMAGE}" \
    /bin/bash -c "
        echo '启动 Qwen3-ASR 服务...'
        qwen-asr-serve \
            ${MODEL_MOUNT} \
            --gpu-memory-utilization ${GPU_MEM_UTIL} \
            --host 0.0.0.0 \
            --port ${CONTAINER_PORT} \
            --max-num-seqs ${MAX_NUM_SEQS}
    "

echo "[INFO] 容器已启动，等待服务初始化..."

# ---------- 等待服务就绪 ----------
MAX_WAIT=120   # 最长等待秒数
INTERVAL=5
ELAPSED=0

echo "[INFO] 探测服务端口 localhost:${HOST_PORT} (最多等待 ${MAX_WAIT}s)..."

until curl -sf "http://localhost:${HOST_PORT}/health" > /dev/null 2>&1; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "[WARN] 服务在 ${MAX_WAIT}s 内未响应 /health，请手动检查日志："
        echo "       docker logs -f ${CONTAINER_NAME}"
        exit 0
    fi
    echo "  ... 等待中 (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep "${INTERVAL}"
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo ""
echo "============================================"
echo "  [SUCCESS] Qwen3-ASR 服务已就绪！"
echo "  访问地址: http://localhost:${HOST_PORT}"
echo "  查看日志: docker logs -f ${CONTAINER_NAME}"
echo "============================================"