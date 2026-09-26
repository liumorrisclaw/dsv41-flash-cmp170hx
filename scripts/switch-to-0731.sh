#!/bin/bash
# 回滚/切换到 0731(权重需已从冷盘恢复到 /home/AIModelDeploy/models/DeepSeekV4-0731)
set -e
docker rm -f dsv41-pp4 2>/dev/null || true
sleep 3
if [ ! -f /home/AIModelDeploy/models/DeepSeekV4-0731/model-00001-of-00048.safetensors ]; then
  echo "!! 0731 权重不在服务器,先从冷盘恢复(Mac 执行):"
  echo "   rsync -a --progress /Volumes/AIBackup/models/DeepSeekV4-0731/ hym@192.168.3.133:/home/AIModelDeploy/models/DeepSeekV4-0731/"
  exit 1
fi
docker start deepseek-vllm-0731-prod
echo "0731 已启动,约 10 分钟加载后 http://192.168.3.133:8000/health 应为 200"
