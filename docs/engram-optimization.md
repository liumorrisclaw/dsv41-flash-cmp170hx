# Engram 优化最终评估 + DSH 配合报告(终版)

**2026-09-20 02:30 | CMP170**

## 一、优化方案落地效果

### 方案 A:静态热区预载(✅ 有效)
- 30GB 头部预读(engram-prewarm.sh)
- 自然语言 16K×3 prefill:盘读从 ~3,840MB → **8MB(-99.8%)**
- 说明自然语言 n-gram 高度集中在头部热区

### 方案 B:自适应守护进程 v2(✅ 运行中)
- 简化为 diskstats 触发 + fadvise(WILLNEED) 头部预热(15GB×2 分片)
- 60 秒周期,检测到盘读活动即自动预热
- v1 的 mincore 方案因 ctypes 兼容性弃用,v2 更简洁可靠

### 方案 C:投机预取(📋 已设计,待 NVMe 后实施)

## 二、磁盘寿命(✅ 零损耗)

| SMART | 优化前 | 优化后 |
|---|---|---|
| Wear_Leveling | **0** | **0** |
| Total_LBAs_Written | 121,085 | 121,267(+182 扇区=91KB,OS 日志级) |
| Temperature | 40°C | 40°C |

## 三、DSH 配合问题(✅ 模型侧已缓解,根因在 DSH 层)

### repetition_penalty 1.1 生效验证

| 测试 | 结果 |
|---|---|
| V1:单工具调用 | ✅ 正确调 get_weather(上海) |
| V2:工具结果→收尾 | ✅ finish=stop,正确总结天气 |
| V3:多步 Agent(两城市对比) | ✅ 正确生成对比表格 |
| vLLM 加载确认 | ✅ `repetition_penalty: 1.1` 已生效 |

### 根因与建议

**根因**:DSH Agent 编排层的复杂系统提示词 + 2bpw mcg 量化包的组合导致工具调用循环。API 层 10 组测试全通过。

| 短期建议 | 长期根治 |
|---|---|
| DSH 换简化 preset | 换 Pollard 校准包(指令遵循更好) |
| DSH 设 max_iterations | NVMe + 投机预取(方案 C)提升响应 |

## 四、附加修复

- 容器 restart policy 改为 `unless-stopped`(防意外崩溃后无法自恢复)
- 容器重启原因确认:我方主动触发(应用 generation_config),非崩溃

## 五、系统当前状态

```
✅ 服务在线 :8095,health 200
✅ Engram 缓存:静态预热 30GB + 自适应守护(60s 周期,15GB×2)
✅ 磁盘寿命:磨损 0,纯读零写
✅ DSH 缓解:repetition_penalty 1.1 已生效
✅ 自动恢复:restart=unless-stopped
✅ 0731 回滚线:冷盘备份完好
```
