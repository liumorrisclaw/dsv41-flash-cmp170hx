#!/bin/bash
# Engram 热区预载:把分片 47/48 头部热区灌进页缓存(n-gram 哈希表高频段)
# 在 dsv41-pp4 服务启动后运行;PREWARM_GB 可调(默认 60)
set -e
PREWARM_GB=${PREWARM_GB:-60}
SRC=/home/AIModelDeploy/models/DSV41-2bpw
for s in $SRC/model-0004[78]-of-00048.safetensors; do
  [ -f "$s" ] || { echo "缺 $s"; exit 1; }
  head -c $((PREWARM_GB*1024*1024*1024/2)) "$s" | cat > /dev/null &
done
wait
echo "prewarmed ${PREWARM_GB}GiB engram head region ($(date))"
free -g | head -2
