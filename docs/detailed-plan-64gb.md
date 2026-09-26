# DeepSeek V4.1-Flash 满血部署方案(CMP170 / 62GB RAM / SATA / 零新增硬件)
## 命令级执行手册 v2

**日期**:2026-09-18 | **执行人**:ZCode + 用户(仅下载授权与最终验收)
**硬约束**:不加内存(62GB)· 不加硬盘(仅现有 SATA)· 不加任何卡 · 0731 数据保留
**路线**:sfxnz EXL3 2.0bpw 满血包(333.5GiB,含 Engram)+ zebgop overlay 的 **SSD-Engram 模式**(`DSV41_ENGRAM_STORAGE=ssd`,Engram 永不进内存,页缓存当缓存)

---

## 0. 一页速览

```
架构:  GPU×4(256GB) ← 145GB 权重(EXL3 2bpw)+ ~90GB KV 池
        宿主 RAM 62GB ← vLLM 进程 ~20GB + ~40GB 页缓存(Engram 热区)
        SATA ← 333.5GiB 完整包,分片 47/48(2×94.56GiB)= Engram,查询时 pread
预期:  decode 散文 ≥20 / 思考 ≥60 tok/s(SATA 影响极小)
        prefill 冷 300 → 热 800-2000 tok/s(看命中率,本方案核心调优点)
        质量 = 满血 764B,无剪枝
时间:  下载 ~3h(后台)→ 组装 ~1.5h → 首启调试 ~3h → 调优 ~2h
回退:  任何一步失败 → 0731 原样恢复(全程不动它的文件与容器)
```

**互斥说明(重要)**:V4.1 与 0731 **共用四张 GPU,不能同时服务**。本方案提供 `switch-to-dsv41.sh` / `switch-to-0731.sh` 一键互切;0731 的 restart 策略回改 `no` 防止重启时二者抢卡。

---

## 1. 资源预算(每阶段变化)

| 时点 | 磁盘已用 | 剩余 | RAM 占用 | 显存 |
|---|---|---|---|---|
| 现在 | 368GiB | 522GiB | 62GB(0731 常驻) | 0731 占用 |
| +下载 333.5GiB | 702GiB | **236GiB** | 不变 | 不变 |
| +vLLM 镜像 ~20GB | 722GiB | **216GiB** | 不变 | 不变 |
| V4.1 服务中 | 同上 | 同上 | ~20GB 进程 + ~40GB 页缓存 | 145GB 权重 + KV(0.93 util) |

---

## 2. Phase 0 —— 前置检查(每项必须过,预期输出已标)

```bash
# P0-1 磁盘余量 ≥ 340GiB
ssh hym@192.168.3.133 'df -h / | awk "NR==2{print \$4}"'
# 预期:≥ 500G(当前 522G)

# P0-2 Gen2 状态(Engram 的 UVA/D2H 依赖链路带宽)
ssh hym@192.168.3.133 'cat /sys/bus/pci/devices/0000:03:00.0/current_link_speed'
# 预期:5.0 GT/s PCIe(四卡都要)

# P0-3 0731 健康且会话内无任务(切换时机由你定)
ssh hym@192.168.3.133 'curl -s -o /dev/null -w "%{http_code}" http://192.168.3.133:8000/health'
# 预期:200

# P0-4 docker 可用 + 剩余镜像空间
ssh hym@192.168.3.133 'docker system df | head -2'

# P0-5 CPU/内存基线记录(留档对比)
ssh hym@192.168.3.133 'free -g | head -2; uptime'

# P0-6 (到时执行)下载服务器连通性
curl -sI --max-time 10 https://huggingface.co | head -1
# 预期:HTTP/2 200(CDN 限速与否到 Phase 1 见分晓)
```

---

## 3. Phase 1 —— 下载权重包(333.5GiB,断点续传,后台 ~3h)

仓库:`sfxnz/DeepSeek-V4.1-Flash-EXL3` @ 分支 `2.0bpw-mcg`(已核实:48 分片,末两片 94.56GiB×2 = Engram,与原版分片切分一致 → 张量命名大概率保留,Phase 2 有终验)

**下载脚本**(存服务器 `/home/hym/dl_sfxnz_2bpw.sh`,16 并发分片、断点续传、单文件分块):

```bash
#!/usr/bin/env bash
# 并行断点续传下载 sfxnz 2.0bpw 包。重跑即续传。
set -uo pipefail
REPO_URL="https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/resolve/2.0bpw-mcg"
DEST=/home/AIModelDeploy/models/DSV41-2bpw
LIST="config.json model.safetensors.index.json tokenizer.json \
$(seq -f 'model-%05g-of-00048.safetensors' 1 48 | tr '\n' ' ')"
mkdir -p "$DEST"; cd "$DEST"
dl() {  # 单文件:8 连接分块续传
  f=$1; url="$REPO_URL/$f"; part="$DEST/.partial/$f"; mkdir -p "$DEST/.partial"
  aria2c -x8 -s8 -c -d "$DEST/.partial" -o "$f" "$url" --file-allocation=none \
    --max-tries=0 --retry-wait=5 --connect-timeout=20 && mv -n "$part/$f" "$DEST/$f"
}
export -f dl; export REPO_URL DEST
# 16 并发跑文件级队列(aria2 未装则 apt -y install aria2)
printf '%s\n' $LIST | xargs -P 16 -I{} bash -c 'dl {}'
echo "DONE: $(ls "$DEST"/*.safetensors 2>/dev/null | wc -l)/48 shards"
```

**检查点**(下载完成后立即):
```bash
# C1-1 分片数与总量
ls /home/AIModelDeploy/models/DSV41-2bpw/*.safetensors | wc -l   # 预期 48
du -sh /home/AIModelDeploy/models/DSV41-2bpw                        # 预期 ~334G
# C1-2 Engram 张量命名终验(SSD 模式兼容的决定性检查)
python3 - <<'EOF'
import json
wm = json.load(open('/home/AIModelDeploy/models/DSV41-2bpw/model.safetensors.index.json'))['weight_map']
need = [k for k in wm if '.engram.embed.weight' in k]
scale= [k for k in wm if '.engram.embed.scale'  in k]
shards = {wm[k] for k in need}
print(f"engram.weight {len(need)} 个 / engram.scale {len(scale)} 个 / 所在分片 {sorted(shards)}")
assert need and all(wm[n]==wm[s.replace('weight','scale')] for n,s in zip(need,scale))
print("PASS: find_engram_shard 兼容")
EOF
# C1-3 失败处理:命名不匹配 → 停,转回退路线 A(REAP,见第 9 节)
```

---

## 4. Phase 2 —— 镜像组装(在服务器上,~1.5h,0731 不受影响)

按 kaka86mm README 六步展开,**新增第 7 步 = SSD 模式**:

```bash
# S1 基础镜像(~20GB)
docker pull vllm/vllm-openai:deepseekv41-flash-0909

# S2 材料
git clone https://github.com/zebgop-ops/dsv41reap-pp.git ~/v41-graft/dsv41reap
git clone https://github.com/vcruz305/vllm-exl3.git       ~/v41-graft/vllm-exl3

# S3 拷 overlay(vllm 的 54 个 .py + hybrid 包)
VLLM_DIR=$(docker run --rm --entrypoint bash vllm/vllm-openai:deepseekv41-flash-0909 -c \
  'python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))"')
# 用临时容器把文件拷进去后 docker commit,或直接 docker build(写个 Dockerfile,见 S8)

# S4 打 4 个补丁(kaka86mm patches/,幂等,按序)
python3 patches/pp_relay_img_ids.py  <vllm>/models/deepseek_v4_1/nvidia/model.py
python3 patches/engram_sm80_fp8.py   <vllm>/models/deepseek_v4_1/common/engram.py
python3 patches/moe_cand_key_fix.py  <vllm>/v1/worker/gpu/model_runner.py
python3 patches/plugin_v150_shim.py  <vllm_exl3>/exl3.py    # 若 vllm-exl3 ≥ 94c29ba 则跳过

# S5 编译 exllamav3 v1.5.0 扩展(容器内 nvcc,sm80)
docker run --rm -v ~/v41-graft:/w --entrypoint bash <工作镜像> -c \
  'cd /w/exllamav3 && TORCH_CUDA_ARCH_LIST="8.0" pip install --no-build-isolation -e . \
   # cusparse 缺头时:pip install nvidia-cusparse-cu13 && 拷头到 /usr/local/cuda/include

# S6 ★SSD 模式(本方案新增,kaka86mm 没做的部分)
docker run --rm -v ~/v41-graft/dsv41reap/overlay/engram_ssd:/w --entrypoint bash <工作镜像> /w/build.sh
#   产物 librow_store.so;把 overlay/engram_ssd/ 整目录拷进 site-packages
#   确认 overlay/vllm/.../engram.py 内含 _engram_storage()/DSV41_ENGRAM_STORAGE 开关
#   (zebgop 原版已内置;若 kaka86mm 的补丁版本覆盖了它,则用 zebgop 原版 engram.py 重打 sm80 补丁)

# S7 组装 Dockerfile(示意)
#   FROM vllm/vllm-openai:deepseekv41-flash-0909
#   COPY overlay/vllm/  /usr/local/lib/python3.12/dist-packages/vllm/
#   COPY overlay/hybrid/ overlay/engram_ssd/ /usr/local/lib/python3.12/dist-packages/
#   COPY vllm-exl3/ → pip install --no-build-isolation ./vllm-exl3
#   ENV VLLM_PLUGINS=vllm_exl3
docker build -t pp4-exl3-ssd:v1 ~/v41-graft/

# S8 CPU 侧导入自检(不碰 GPU)
docker run --rm --entrypoint bash pp4-exl3-ssd:v1 \
  ~/v41-graft/dsv41reap/overlay/import-check.sh
# 预期:全部 overlaid 模块 import PASS
```

---

## 5. Phase 3 —— 启动(含 0731 互切)

**switch-to-dsv41.sh**(存 `/home/AIModelDeploy/scripts/`):
```bash
#!/usr/bin/env bash
set -euo pipefail
docker stop -t 90 deepseek-vllm-0731-prod        # 让出 GPU
sleep 5; docker rm -f dsv41-pp4 2>/dev/null || true
docker run -d --name dsv41-pp4 --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v /home/AIModelDeploy/models:/models:ro \
  -v ~/v41-graft/dsv41reap/overlay/engram_ssd:/engram_ssd_ro:ro \
  -v ~/tilelang-cache:/root/.tilelang \
  -p 8095:8095 \
  -e VLLM_PLUGINS=vllm_exl3 -e VLLM_ENGINE_READY_TIMEOUT_S=3600 \
  -e NCCL_CUMEM_ENABLE=0 -e VLLM_USE_V2_MODEL_RUNNER=0 \
  -e VLLM_PP_LAYER_PARTITION=10,10,11,9 \
  -e TORCH_CUDA_ARCH_LIST="8.0" \
  -e DSV41_ENGRAM_STORAGE=ssd \
  -e DSV41_ENGRAM_THREADS=10 \
  -e PYTHONPATH=/overlay \
  --entrypoint vllm pp4-exl3-ssd:v1 serve /models/DSV41-2bpw \
  --pipeline-parallel-size 4 --tensor-parallel-size 1 \
  --served-model-name deepseek-v4.1-flash \
  --host 0.0.0.0 --port 8095 \
  --max-model-len 524288 --max-num-seqs 32 \
  --max-num-batched-tokens 2048 \
  --gpu-memory-utilization 0.93 \
  --kv-cache-dtype fp8_ds_mla \
  --engram-config '{"cpu_offload": false}' \
  --enable-prefix-caching \
  --compilation-config '{"cudagraph_mode":"PIECEWISE","cudagraph_capture_sizes":[1,2,3,4,5,6,8,10,12,16,20,24,32]}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}' \
  --tokenizer-mode deepseek_v41 \
  --enable-auto-tool-choice --tool-call-parser deepseek_v41 --reasoning-parser deepseek_v41
```
**相对 kaka86mm 原版的五处改动**(每处都有依据):
1. 删掉 `AVAIL<200GB exit` 守卫(我们靠页缓存,不钉内存)
2. `-e DSV41_ENGRAM_STORAGE=ssd`(核心开关)+ `DSV41_ENGRAM_THREADS=10`(12C/24T 留余量,默认 16)
3. `--engram-config '{"cpu_offload": false}'`(不再钉宿主)
4. `--max-num-batched-tokens 2048`(8192→2048:zebgop 在 SATA 弱盘场景的同款选择,把冷块 prefill 的最坏等待从 ~31s 压到 ~8s)
5. 端口 8095、容器名 dsv41-pp4,与 0731(8000)隔离

**启动观察**(首个启动预计 8-12 分钟):
```bash
docker logs -f dsv41-pp4 2>&1 | grep -E "Engram layer|served from SSD|Engine|error|Error"
# 关键正常信号:"Engram layer N served from SSD: ... threads=10"(每层一条)
# 关键异常:UVA/pin 相关字样 = SSD 模式没生效,查 S6
```

---

## 6. Phase 4 —— 验收(kaka86mm bench 一键套件)

```bash
git clone https://github.com/kaka86mm/dsv41-flash-pp4-170hx.git ~/v41-bench
cd ~/v41-bench && python3 bench/accept.py --base-url http://192.168.3.133:8095 \
  --model deepseek-v4.1-flash
```

| # | 验收项 | 通过线 | 不达标处理 |
|---|---|---|---|
| 1 | 质量 7 项 | 7/7 | <7 → 查补丁顺序,对照 kaka86mm FINDINGS §测量纪律 |
| 2 | 视觉 2 项 | 2/2 | 同上 |
| 3 | decode 散文/思考 | ≥20 / ≥60 tok/s | 差太多 → `DSV41_ENGRAM_THREADS` 扫 8/12/16 |
| 4 | prefill 8K(冷) | ≥250 tok/s | 低于 → 缩 batched-tokens 到 1024 + 跑热区预载(Phase 5) |
| 5 | prefill 8K(热,第 3 次) | ≥800 tok/s | 不达 → 命中率诊断(Phase 5 监控),考虑回退 A |
| 6 | 2h 混合负载 | 零 Xid/OOM/退出 | Xid → 记录场景,先查驱动侧 |
| 7 | 稳态命中率 | ≥85% | 数据本身要记录(可发布) |

---

## 7. Phase 5 —— SATA 特化调优(自研增量)

**热区预载脚本** `engram-prewarm.sh`(启动完成后台跑,把分片 47/48 头部热区灌进页缓存):
```bash
#!/usr/bin/env bash
# 依据:n-gram 哈希表行号≈训练语料频率的哈希分布,头部页集中了高频 n-gram;
# row_store_stats 的命中分布可事后校正预载量。预载 60GiB ≈ 4 分钟(顺序读)。
PREWARM_GB=${PREWARM_GB:-60}
for s in /home/AIModelDeploy/models/DSV41-2bpw/model-0004{7,8}-of-00048.safetensors; do
  head -c $((PREWARM_GB*1024*1024*1024/2)) "$s" | cat > /dev/null
done
vmtouch -e /home/AIModelDeploy/models/DSV41-2bpw/model-0004[78]* 2>/dev/null || true
echo "prewarmed ${PREWARM_GB}GiB engram head region"
```

**监控三件套**(跑负载时开三个窗口):
```bash
watch -n5 'cat /proc/meminfo | grep -E "^Cached|^Dirty"; df -h / | tail -1'
iostat -x 5 /dev/sda          # 随机读 IOPS vs 6328 基线、util%、温度降速征兆
docker exec dsv41-pp4 python3 -c \
  "from engram_ssd import *; print(open_engram_stats())"   # row_store_stats 命中计数(封装按 S6 落位调整)
```

**调优旋钮优先级**:`DSV41_ENGRAM_THREADS`(8/10/12/16 扫)→ `--max-num-batched-tokens`(1024/2048)→ `PREWARM_GB`(40/60/90)→ `vm.swappiness=10` + `vm.vfs_cache_pressure=50`。

---

## 8. 回退决策树

```
C1-2 engram 命名不匹配 ──────────► 路线 A(REAP-272E,zebgop 原样配方,
                                    下载 397GiB:REAP 208 + 原版 47/48)
S6 SSD 模式编不过/不生效 ─────────► 同上(路线 A 的 overlay 原生含 SSD 路径)
首启 OOM / Xid 反复 ────────────► 检查 pin_memory 路径(zebgop 的"钉驻陷阱"警告),
                                    仍不行 → 路线 A
prefill 热 < 300 tok/s 且命中率高 ► 结构性瓶颈,接受 decode 场景或加盘(你已排除)→ 维持现状汇报
任何时点想回 0731 ───────────────► bash switch-to-0731.sh(docker stop dsv41-pp4
                                    && docker start deepseek-vllm-0731-prod,~10 分钟恢复)
```

## 9. 路线 A 备用清单(仅回退时启用)

- 下载:`hf download LibertAIDAI/DeepSeek-V4.1-Flash-REAP-272E`(208GiB)+ 原版 `deepseek-ai/DeepSeek-V4.1-Flash` 仅取 `model-0004{7,8}-of-00048.safetensors`(--include 过滤,189GiB)
- `link-reap-engram.sh` 链接分片 → `run-dsv41reap-pp4.sh` 启动(参数全默认,他们实跑过的组合)
- 代价:MMLU −7.7(代码无损);disk 368+397=765GiB,余 173GiB

## 10. 附录

**新增端口/目录/容器**
| 项 | 值 |
|---|---|
| 容器 | dsv41-pp4(restart=no,手动/脚本切换) |
| 端口 | 8095 |
| 模型目录 | /home/AIModelDeploy/models/DSV41-2bpw(334GiB) |
| 工作材料 | ~/v41-graft(overlay+插件)、~/tilelang-cache、~/v41-bench |
| 关键 env | DSV41_ENGRAM_STORAGE=ssd、DSV41_ENGRAM_THREADS=10 |

**0731 变更(仅一项)**:`docker update --restart no deepseek-vllm-0731-prod`(防止重启抢卡;现有容器/数据/权重零改动)

**已核实的事实依据**:sfxnz@2.0bpw-mcg 分片表(48 片/333.5GiB/末两片 94.56GiB×2)· 本机 fio 4K QD16 直读 6,328 IOPS · zebgop engram_ssd.py+row_store.cpp 源码(线程池 pread/页缓存/cudaLaunchHostFunc)· kaka86mm launch-pp4.sh 全文与五处差异 · 磁盘/内存现值(368GiB 用/62GB RAM)
