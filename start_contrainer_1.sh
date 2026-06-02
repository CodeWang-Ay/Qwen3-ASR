docker run --gpus all --name qwen3-asr_official_node_2 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8872:80 \
    --mount type=bind,source=/opt/data/LLM/Qwen3-ASR-0.6B,target=/data/shared/Qwen3-ASR \
    --mount type=bind,source=/opt/wjg/workspace/nvcr_workspace/interview/Qwen3-ASR/start_asr_2.sh,target=/start_asr_2.sh \
    --shm-size=10gb \
    -d qwenllm/qwen3-asr:latest /bin/bash /start_asr_2.sh

docker run --gpus all --name qwen3-asr_official_node_4 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8874:80 \
    --mount type=bind,source=/opt/data/LLM/Qwen3-ASR-0.6B,target=/data/shared/Qwen3-ASR \
    --mount type=bind,source=/opt/wjg/workspace/nvcr_workspace/interview/Qwen3-ASR/start_asr_2.sh,target=/start_asr_2.sh \
    --shm-size=10gb \
    -d qwenllm/qwen3-asr:latest /bin/bash /start_asr_2.sh

docker run --gpus all --name qwen3-asr_official_node_3 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8873:80 \
    --mount type=bind,source=/opt/data/LLM/Qwen3-ASR-0.6B,target=/data/shared/Qwen3-ASR \
    --mount type=bind,source=/opt/wjg/workspace/nvcr_workspace/interview/Qwen3-ASR/start_asr_3.sh,target=/start_asr_3.sh \
    --shm-size=40gb \
    -d qwenllm/qwen3-asr:latest /bin/bash /start_asr_3.sh

docker run --gpus all --name qwen3-asr_official_node_4 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8874:80 \
    --mount type=bind,source=/opt/data/LLM/Qwen3-ASR-0.6B,target=/data/shared/Qwen3-ASR \
    --mount type=bind,source=/opt/wjg/workspace/nvcr_workspace/interview/Qwen3-ASR/start_asr_4.sh,target=/start_asr_4.sh \
    --shm-size=40gb \
    -d qwenllm/qwen3-asr:latest /bin/bash /start_asr_4.sh

# 1.7B 模型
docker run --gpus all --name qwen3-asr_official_node_2 \
    -v /var/run/docker.sock:/var/run/docker.sock \
    -p 8872:80 \
    --mount type=bind,source=/opt/data/LLM/Qwen3-ASR-1.7B,target=/data/shared/Qwen3-ASR \
    --mount type=bind,source=/opt/wjg/workspace/nvcr_workspace/interview/Qwen3-ASR/start_asr_2.sh,target=/start_asr_2.sh \
    --shm-size=10gb \
    -d qwenllm/qwen3-asr:latest /bin/bash /start_asr_2.sh
