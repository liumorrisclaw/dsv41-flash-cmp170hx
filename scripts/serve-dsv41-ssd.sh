#!/bin/bash
# DSV41-Flash on 4×CMP 170HX — PP4 + EXL3 2bpw + DSpark + SSD-Engram
# 挂载采用 zebgop 官方机制:overlay 逐文件覆盖进镜像 vllm 包内
set -u

NAME=${NAME:-dsv41-pp4}
PORT=${PORT:-8095}
MODEL=${MODEL:-/models/DSV41-2bpw}
GRAFT=${GRAFT:-$HOME/v41-graft}
IMAGE=${IMAGE:-pp4-exl3-ssd:v1}
VLLM_SITE=/usr/local/lib/python3.12/dist-packages/vllm

docker rm -f $NAME 2>/dev/null; sleep 3

USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | paste -sd+ | bc)
[ "$USED" -gt 2000 ] && { echo "GPU busy: $USED MiB"; exit 1; }

mkdir -p $HOME/tilelang-cache

# overlay 逐文件挂载(官方法)
MOUNTS=()
while IFS= read -r f; do
  rel=${f#$GRAFT/dsv41reap/overlay/vllm/}
  MOUNTS+=(-v "$f:$VLLM_SITE/$rel:ro")
done < <(find $GRAFT/dsv41reap/overlay/vllm -type f -name "*.py")
echo "overlay 文件挂载数: ${#MOUNTS[@]}"

docker run -d --name $NAME \
  --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/AIModelDeploy/models:/models:ro \
  "${MOUNTS[@]}" \
  -v $GRAFT/dsv41reap/overlay/engram_ssd:/opt/dsv41/engram_ssd:ro \
  -v $GRAFT/dsv41reap/overlay/hybrid:/opt/dsv41/hybrid:ro \
  -v $HOME/tilelang-cache:/root/.tilelang \
  -p $PORT:$PORT \
  -e VLLM_PLUGINS=vllm_exl3 \
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_CUMEM_ENABLE=0 \
  -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_PP_LAYER_PARTITION=10,10,11,9 \
  -e TORCH_CUDA_ARCH_LIST="8.0" \
  -e DSV41_ENGRAM_STORAGE=ssd \
  -e DSV41_ENGRAM_THREADS=10 \
  -e DSV41_SKIP_WEIGHT_RE='layers\.(1|14)\.engram\.embed\.' \
  -e PYTHONPATH=/opt/dsv41 \
  --entrypoint vllm \
  $IMAGE serve $MODEL \
  --pipeline-parallel-size 4 \
  --tensor-parallel-size 1 \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --port $PORT \
  --max-model-len 1048576 \
  --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload": false}' \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,16,20,24,32]}' \
  --speculative-config '{"method":"ngram","num_speculative_tokens":5}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41

echo "launched $NAME on :$PORT — watch: docker logs -f $NAME 2>&1 | grep -E 'Engram|served from|error'"
