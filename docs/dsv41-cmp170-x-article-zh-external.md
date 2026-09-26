# 1. 全球首个:满血 DeepSeek V4.1-Flash 跑在 4× CMP 170HX 上,不加内存不加盘

昨晚凌晨一点,health check 返回 200 的时候我盯着屏幕愣了几秒。

764B 参数。189GB Engram 检索表。跑在四张矿卡上,62GB 内存,一块 SATA 盘。

没有 NVMe。没有加内存条。没有换主板。

prefill 3,355 tok/s,decode 35.5 tok/s,原生视觉,1M 上下文实测通过。

为什么要做这件事？说实话，输出效率不会高——62GB 内存跑 764B 模型，Engram 查表走 SATA，prefill 和 decode 都会被硬件卡死。这不是一个追求性能的项目。

我真正想知道的是：**在硬件严重受限的条件下，有哪些优化思路可以撑起一个"不可能"的部署？** 189GB 的检索表能不能当数据库用而不是当权重加载？62GB 内存里的页缓存能不能扛住一个 189GB 表的查询压力？SATA 的随机读够不够喂一个 764B 的 Engram 门控？

这些问题没有现成答案——社区已有方案要么依赖 373GB 内存，要么依赖 NVMe，都跳过了硬件受限这个场景。我把整个过程和踩过的坑写出来，希望能给做类似优化的人多一些参考。

假期抽空把整个测试报告整理了出来，以下是完整数据。

# 2. 为什么这事以前没人做成

V4.1-Flash 的 Engram 是个 189GB 的 n-gram 检索表——不是矩阵,不能量化,必须 FP8 原样放。kaka86mm 跑通了 4× 170HX,但他用了 373GB 内存把这张表钉死。Mia 的双 DGX Spark 用了 NVMe packing。

我的机器:62GB 内存,SATA 盘随机读 50MB/s。Engram 表比内存大三倍。

以前所有人都把这张表当成"必须完整加载"的东西。但其实它是个查找表——每 token 只需要查 ~24 行,每次 264 字节。

# 3. 主流方案对比与思路来源

在定方案之前,我把社区已有的几条路线过了一遍:

**路线 A:kaka86mm 的 373GB 内存钉表方案。** 同款 4× 170HX,质量 7/7,速度最稳。但硬性要求宿主 RAM ≥373GB(Engram 钉 189GiB + 余量),我的 62GB 差了 6 倍。参考价值:证明了 170HX 能跑 EXL3 2bpw,benchmark 套件可直接复用。

**路线 B:Mia 双 DGX Spark 的 NVMe packing 方案。** Engram 按节点拆成 ~94GiB 打进 NVMe,用 `posix_fadvise` 预热。核心洞察:**Engram 是查找表,不需要完整加载,按需 pread 即可**。但他的机器有 NVMe(随机读 ~400MB/s),我的 SATA 只有 ~50MB/s。

**路线 C:zebgop-ops 的 SSD-Engram overlay。** 在 dsv41reap-pp 仓库里实现了完整的 `DSV41_ENGRAM_STORAGE=ssd` 模式:GPU 算哈希 → CUDA 宿主回调 → 线程池 pread 直读 safetensors → 页缓存当 LRU → 设备端反量化。他的测试机器 123GB RAM + NVMe。

**路线 D:PCIe 电容模组 + EPYC 平台。** 硬件路线,24 焊点/卡 + 换主板。能解决 ×16 带宽但不解决 Engram 存储问题,投入 ¥2 万+。

我的方案 = **路线 B 的核心思想(Engram 按需查表,不加载)+ 路线 C 的现成实现(SSD 模式开关)+ 路线 A 的权重包和 benchmark**。赌的是:62GB 内存里腾出 40GB 做页缓存,n-gram 的 Zipf 分布能让大部分查询命中,冷查询的 SATA 带宽够 decode 用(prefill 慢但 decode 快)。

# 4. 实际部署(9月18日晚间)

权重下载和 docker 镜像拉取都是体力活——换镜像源、调并发、等断点续传，不展开。真正有技术含量的是两件事：**首启四关**和**减少磁盘读取**。

## 首启四关

**第一关，overlay 挂载。** zebgop 的 overlay 有 54 个 .py 文件，覆盖了 vLLM 的注意力、Engram、DSpark、权重加载器等核心模块。最初我尝试用 PYTHONPATH 让 Python 优先加载 overlay 目录——结果只对顶层包生效，vLLM 内部的子模块导入（`from vllm.models.deepseek_v4_1.common.engram import ...`）仍然指向镜像内的原版文件。解法是逐文件 bind-mount：遍历 overlay/vllm/ 下的每一个 .py，分别 mount 到镜像内 vllm 包的对应路径上（共 108 个文件）。这样 Python 的 import 机制天然命中 overlay 版本，零侵入。

**第二关，94.5GB mmap 被内核拒绝。** Engram 表太大，vLLM 的权重加载器用 `mmap` 惰性映射 safetensors 文件。62GB 内存的机器，默认 `vm.overcommit_memory=0`（启发式模式）下，内核发现 94.5GB 的映射接近物理内存的 150% 直接拒绝。`vm.overcommit_memory=1`（总是允许）解决——但这也意味着如果真的同时触碰所有页就会 OOM。实际上 Engram 的查询是稀疏的（每 token 只触 ~24 行），配合后面的 SKIP_WEIGHT_RE 跳过加载，实际触碰量极小。这是 62GB 机器的独有坑，zebgop 的 123GB 机器从没遇到过。

**第三关，Engram 跳过正则。** 即使 mmap 允许了，也不该真的把 189GB 表读进内存。overlay 提供了 `DSV41_SKIP_WEIGHT_RE` 环境变量——一个正则，匹配到的权重名会被加载器在 open 文件之前就跳过。正确值是 `layers\.(1|14)\.engram\.embed\.`（Engram 只挂在第 1 和第 14 层）。这个正则在 shell 嵌套里转义写错了三次才搞对。

**第四关，DSpark 草稿模型不支持 PP4。** vLLM 原版对 speculative decoding + pipeline parallelism 的组合有硬编码检查（要求草稿模型实现 SupportsPP 接口），而 DSpark 的草稿不是流水线化的——它整体住在最后一个 rank 上。overlay 的 speculative.py 补丁加了一段旁路：当 method=dspark 且 pipeline_parallel_size>1 时，强制把 draft_parallel_config.pipeline_parallel_size 设为 1。这段代码必须通过逐文件挂载才能生效。

## 减少磁盘读取

189GB 检索表在 SATA 上，随机读 ~50MB/s，这是整个方案最大的性能瓶颈。三层缓存体系来扛：

**第一层：前缀缓存。** vLLM 自带的 prefix caching，同提示词重发时盘读从 1.3GB 降到 0.1MB（-99.9%）。这是免费的——vLLM 把已见过的 KV 块按 block hash 索引，相同前缀直接复用，不需要重新 prefill 也就不需要重新查 Engram。

**第二层：Linux 页缓存。** 62GB 内存里 ~40GB 交给内核自动管理。关键洞察：n-gram 的 Zipf 分布意味着高频行集中在哈希表的头部区域，页缓存的 LRU 淘汰策略天然把最热的页留在内存里。不需要任何手动干预，内核自己就把最常用的 Engram 行缓存住了。稳态下 decode 期盘读几乎为零（2000 tok 生成只读了 42MB，其中大部分是一次性的冷行）。

**第三层：静态热区预载。** 页缓存的问题是冷启动——刚开机时缓存是空的，前几个长文档的 prefill 会大量落盘。解法是开机后把分片头部 30GB 顺序读一遍灌进页缓存（engram-prewarm.sh，约 4 分钟）。依据是自然语言的 n-gram 频率分布高度偏向头部（Zipf定律：排名第 k 的 n-gram 出现频率约为第 1 名的 1/k），头部 30GB 大约覆盖了 80%+ 的常见查询。实测效果：自然语言 16K×3 prefill 的盘读从 3,840MB 降到 8MB（-99.8%），32K prefill 从全冷的 622 tok/s 升到稳态的 1,821 tok/s。

**自适应守护进程（正在运行）。** 每 60 秒检测盘读活动，有查询就自动 fadvise(WILLNEED) 头部 15GB×2 分片，防止 LRU 把热页冲出去。后续计划用 row_store_stats() 积累真实命中分布（哪些行号被读最多），定向预载最热页，比"猜头部"更精准。

三层叠加后，自然语言场景的 Engram 查询几乎全部命中页缓存，SATA 的随机读带宽不再是 decode 的瓶颈——prefill 仍然受限（每 token 31-40KB 盘读），但这可以通过热区预载扩大覆盖来缓解。如果把 SATA 换成一块 NVMe（Gen3×4，随机读约 8 倍），prefill 预计从 ~1,800 提升到 ~4,000 tok/s（接近 0731 的全显存水平），同时保留 SSD-Engram 的零内存占用优势。

# 5. 实测数据(9月19日~20日)

验收 5/7 质量 + 2/2 视觉(计数 400/400 全对,发票 OCR 精确到 $1,337.42)。

**prefill 逐档:**

| 上下文 | DSV41 稳态 | 双 DGX Spark | 自家 0731 |
|---|---|---|---|
| 16K | 3,355 | 987 | 4,911 |
| 32K | 1,821 | 1,055 | 4,814 |
| 128K | 2,548 | 961 | 4,911 |
| 256K | 1,615 | 873 | 4,211 |
| 512K | 1,284 | ~850 | 3,247 |
| 1M wall | 875s | —(600K 上限) | 483s |

**decode:**

| 场景 | DSV41 | 双 Spark | 0731 |
|---|---|---|---|
| 短提示 | 54.6 | 31.6 | 93.3 |
| 128K | 33.1 | 19.0 | 104.4 |
| 计数 | 59.9 | 40 | — |

prefill 全档位压双 Spark(1.5-3.4×),decode 主力场景 1.7× 领先。对 0731 付出 2-3× 速度代价,换满血 764B + Engram + 视觉 + 1M——模型整整一代。

**DSH 实测**(DeepSeek Harness,真实 Agent 环境):

| 场景 | tok/s | 轮次 | finish |
|---|---|---|---|
| 短对话(150tok) | 11.3 | 3 轮均值 | length |
| 工具调用(get_weather) | 11.6 | 3 轮均值 | stop ✓ |
| 代码生成(500tok) | 11.5 | 3 轮均值 | length |
| 长回复(800tok) | 11.2 | 3 轮均值 | length |

四场景稳定在 **11.2-11.6 tok/s**(标准差 <0.3),工具调用正确触发并正常收尾。DSpark 投机草稿接受率 70-90%,前缀缓存命中率 44-59%。26 轮 API 调用无 OOM,无 Xid。Agent 层的循环问题通过 repetition_penalty 1.1 缓解(根因是 2bpw mcg 包的指令遵循精度)。

磁盘:SMART 磨损计数 0,写入仅 59GB(0.02% 寿命)。Engram 路径纯读,零 NAND 磨损。

# 6. 下一步:测试 GLM-5.3-Flash(假期继续)

同一台机器,同样的 SSD-Engram 架构,下一个目标是 GLM-5.3-Flash。Mia 已经在双 DGX Spark 上跑通了 GLM 5.3 Flash EXL3(单流 40 tok/s,C4 78 tok/s),GLM-5.3 用 DeepSeek 同款稀疏注意力架构,理论上同样可以走这条路。

社区也有人整理了 GLM-5.3-Flash 在 4× CMP 170HX 上的部署配方:[Morrowmake/glm53-flash-cmp170hx-recipe](https://github.com/Morrowmake/glm53-flash-cmp170hx-recipe) — 后续实测后会跟 DSV41 的数据做一轮横向对比,看两个模型在同套硬件上各自的定位。

# 7. 致谢与开源

[zebgop-ops/dsv41reap-pp](https://github.com/zebgop-ops/dsv41reap-pp) — SSD-Engram 机制与完整 vLLM overlay

[kaka86mm/dsv41-flash-pp4-170hx](https://github.com/kaka86mm/dsv41-flash-pp4-170hx) — EXL3 2bpw 权重包与 benchmark 套件

[amoghmunikote/cmpunlocker](https://github.com/amoghmunikote/cmpunlocker) — CMP 170HX 解锁工具链

[0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/0xSero/deepseek-v4.1-flash-4x-rtx-pro-6000) — row_store.cpp 原型

[Consensus-Protocol/cmp170hx](https://github.com/Consensus-Protocol/cmp170hx) — CMP 170HX 知识库

[sfxnz/DeepSeek-V4.1-Flash-EXL3](https://huggingface.co/sfxnz/DeepSeek-V4.1-Flash-EXL3/tree/2.0bpw-mcg) — 2.0bpw 权重包

完整部署方案、测试数据和脚本已开源:[dsv41-flash-cmp170hx](https://github.com/liumorrisclaw/dsv41-flash-cmp170hx)

Gen2 修复套件:[cmp170hx-gen2-window-fix](https://github.com/liumorrisclaw/cmp170hx-gen2-window-fix)
