# DSV41-Flash 部署方案 v3(执行版)
**2026-09-19 | CMP170(SERVER)| 62GB RAM / SATA / 4×64GB / 零新增硬件**

## 0. 已验证的事实基础

| 项 | 状态 |
|---|---|
| 路线 | sfxnz EXL3 2.0bpw(334GiB 单包,含 Engram 189GiB)+ zebgop overlay SSD 模式(`DSV41_ENGRAM_STORAGE=ssd`)|
| Engram 在包内 | ✅ 分片 47/48 = 94.56GiB×2(2026-09-18 核实)|
| SATA 能力 | 4K randread QD16 = 6,328 IOPS(fio 实测)→ decode 无压力,prefill 靠页缓存命中率 |
| **回滚生命线** | ✅ **0731 冷盘备份完整**(AIBackup:48/48 片、156G、`cold-seed-complete` 标记),恢复≈30 分钟 LAN 传输 |
| 磁盘现状 | 938G 总 / 368G 用 / **522G 空闲** |

## 1. 磁盘释放计划(用户已授权:释放 0731 及部分模型数据)

| 动作 | 释放 | 风险与回滚 |
|---|---|---|
| 停 0731 容器(保留容器配置+镜像,只删权重) | — | 回滚:冷盘 rsync 回 + docker start |
| 删 `/home/AIModelDeploy/models/DeepSeekV4-0731`(156G) | **+156G** | ✅ 冷盘已验证 |
| 清理已退出容器(candidate/paused)+ docker volume prune(9 个卷全无容器在用) | +16.5G | mode4 数据,长期停用;Qwen 权重另有 /models 目录副本 |
| **保留**:ComfyUI 全家(text_encoders 47G + diffusion 20G + vae/loras 9G)、Mode4 Qwen 模型 15G | 0 | 无备份,作为二线储备不轻动 |

**释放后:~694G 空闲**(需求:包 334G + 镜像 ~25G + 缓存 ~15G = 375G,余 ~320G 储备——够回退路线 A 的原版分片 189G)

## 2. 执行阶段(依赖顺序)

- **P1 磁盘释放**(本文件 §1)→ **P2 下载**(aria2 16 并发断点续传,~3h 后台,装 aria2)→ **P3 镜像组装**(pull vllm-openai:deepseekv41-flash-0909 + overlay 54 文件 + 4 补丁 + vllm-exl3 + exllamav3 sm80 编译 + **engram_ssd build(S6 关键)**)→ **P4 SSD 模式首启**(launch 五改动:删 200G 守卫 / SSD 开关 / THREADS=10 / batched=2048 / 端口 8095)→ **P5 验收**(kaka86mm bench/accept.py:质量 7 项+视觉 2 项+性能)→ **P6 调优**(热区预载 60GiB/线程扫描/命中率记录)

## 3. 失败判定与回滚协议(按用户要求:不轻言失败)

**遇到问题先修,修的方向**(按序尝试,每步记录):
1. 启动失败 → 看 docker logs 定位(常见:补丁顺序/编译环境/V1 runner 开关),对照 FINDINGS 排障表
2. SSD 模式不生效 → 检查 engram.py 是否含开关、librow_store.so 是否装入、PYTHONPATH
3. C1-2 命名不匹配 → 转路线 A(REAP-272E + 原版 47/48 分片,zebgop 原样配方)
4. prefill 过慢(<300 tok/s 热)→ 先调优(P6 全套),仍不行评估业务可接受性
5. OOM/Xid → 检查 pin_memory 路径(zebgop 警告过),降 util/seqs

**只有以下情形向用户申报"建议回滚",由用户拍板**:
- 质量验收 <7/7 且排查后判定为路线固有缺陷
- 连续 2 天主要问题无进展且 0731 业务等待压力大
- 出现硬件风险信号(反复 Xid 31 类)

**回滚步骤**(30-40 分钟):冷盘 rsync 0731 权重回服务器 → docker start deepseek-vllm-0731-prod → 健康检查 → (可选)删 dsv41 释放磁盘。dsv41 容器/配置保留以便再战。

## 4. 互斥与运维
- 0731 restart 策略改为 `no`(防重启抢卡);dsv41 同样 `no`,手动/脚本切换
- `switch-to-dsv41.sh` / `switch-to-0731.sh` 放 /home/AIModelDeploy/scripts/
- 智能插座链路不受影响(Gen2 hammer 是 udev 级,与容器无关)
