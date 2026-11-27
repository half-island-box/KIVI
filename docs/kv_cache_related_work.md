# KV Cache 量化：相关工作对比与定位（2026-11-27）

> **用途**：为 SinkBudget 的定位与 M2/M3 实验设计提供依据。
> **信息来源**：论文摘要、arXiv HTML 检索片段与二手摘要。本机无法直接访问 `arxiv.org` / `github.com`（域名被拦），**表中数字未逐字核对原文表格**；标 ⚠️ 的条目尤其需要复核。
> **对照对象**：我们在 `baseline_v1` 与 M1 上已经测出的数字（见 `docs/m0_baseline_v1.md`、`docs/m1_sink_calibration.md`）。

## 1. 一页对比

| 维度 | KVTuner (ICML 2025) | KVmix (AAAI 2026 Oral) | MultiAlloc (arXiv 2026-10) | 本项目现状 |
|---|---|---|---|---|
| 压缩粒度 | 层 × K/V 各自 bit 宽度 | 层 × K/V bit + 层内近期 FP 比例 | 层 bit + 组内 token 高精度，二者联合 | 层间 sink 保护预算（原计划） |
| 层重要性度量 | 敏感度驱动的离线多目标搜索 + 层间聚类 | K/V 投影权重的 L2 梯度范数 | 一阶 Taylor 扰动 profiling | 注意力质量 `A_l`（实测 0.407–0.874） |
| token 机制 | 无（只调 bit） | **RPC**：按层重要性给不同的"保留全精度近期 token 比例" $r_l$ | 组内 top-$k$ 高精度，token 分数用 L1 key norm | 按 `A_l` 决定每层保护几个 sink token |
| 预算口径 | 有效 bit 宽度 | 层 bit + RPC 比例 | **固定 KV 字节预算**（Llama-3-8B：423 MB vs FP16 1024 MB） | 待定 |
| 在线开销 | 无（离线搜索，配置复用） | 无（静态策略） | 无（静态策略 `(q*,k*)`） | 设计为粗粒度静态 |
| 评测 | LongBench、GSM8K、MATH500、GPQA | LongBench（Llama-2-7B 等） | GSM8K、GPQA、LongBench-v2，5 个模型 | LongBench 8 任务（4k） + WikiText PPL |
| 关键数字 ⚠️ | Llama-3.1-8B **3.25 bit 近无损**；点名 **L20–30 最敏感** | K **2.19** / V **2.38** bit，4.9× 压缩；Llama-2-7B 33.36 vs FP16 33.84 | Llama-3-8B 综合 **46.35** vs 最好压缩基线 44.95 vs FP16 46.17 | 2bit PPL **+7.03%**；4bit **+0.19%** |
| 已知弱点 / 争议 ⚠️ | 有后续工作批评其 K2V2 静态分配过激 | RPC 按"近期性"而非注意力 | LongBench-v2 上 **KVmix 26.94 > MultiAlloc 25.85**，聚合优势掩盖逐项落后 | 2bit 档位下 token 选择已饱和 |

## 2. 它们覆盖了什么

1. **层间 bit 分配**：KVTuner、KVmix、MultiAlloc 三篇全部覆盖，且 KVTuner 就在 `Llama-3.1-8B-Instruct` 上做。**我们原计划里"层间异质性分配"的核心动作已被占满。**
2. **层间 token 预算**：KVmix 的 RPC 比例 $r_l$ 就是"每层保留多少全精度 token"，只是它保护的是**最近**的 token。
3. **层 × token 联合分配**：MultiAlloc 明确做了，并且用框架内消融（layer-only / token-only / joint）论证互补性。

## 3. 我们的数据与三者的交叉验证

| 我们的观察 | 与三篇的关系 |
|---|---|
| `A_l` 层间 2.15×、稳定性 0.989 | 与 KVTuner"层间敏感度差异显著"一致；**但"层间有差异"本身已不新** |
| 保护 {0,1} 挽回 **62%** 的 2bit PPL 损失；保护整个 prefill 只到 **75%** | 没有任何一篇做"序列开头 sink 的逐层 FP 保护" |
| 剩余 25% 在 decode 侧；`residual_length` 扫描显示边际价值**弥散**（每位置 0.039 pp，而 sink 每位置 31 pp） | 与 MultiAlloc 的消融一致：**token 维度是较弱的旋钮**（其 token-only 24.88 < layer-only 25.58，联合仅 +0.27） |
| key-norm proxy 与真实注意力 **Spearman −0.485**（0/32 层为正） | MultiAlloc 的 token 分数也用 L1 key norm；它优化的是**量化误差**而非注意力质量，两者不直接冲突 ⚠️ 需核对机制 |
| 纯 prefill 的指标（PPL / logits / hidden）对 2/4/16 bit **逐位相同** | 未见三篇讨论；这是我们独有的方法论观察 |

## 4. 仍然空白的地方（截至本文写作时）

1. **2 bit 档位**。三篇的主战场在 3.2–6.6 bit（MultiAlloc 423/1024 ≈ 6.6 bit），而 8× 压缩的 2 bit 是完全不同的区域——那里 token/层分配都已饱和。
2. **序列开头 sink 的专项保护**。三篇都没有把"位置 0/1"当作独立机制处理；StreamingLLM / KVSink 做过，但不在量化分配框架内。
3. **decode 期的动态决策**。三篇都是 prefill 后一次性静态策略。而我们的数据指认 25% 的损失落在 decode 生成出来的 token 上。
4. **2 bit 下的"位置性 vs 近期性"效率对比**：2 个 sink 位置 ≈ 1000 个近期 token 的价值（31 pp vs 0.039 pp 每位置）。

## 5. 待核对清单

- [ ] KVTuner 的 3.25-bit 配置明细：K/V 分别多少 bit、哪些层、baseline 是 KIVI 还是 KV8。
- [ ] KVmix 的 RPC 公式：$r_l$ 与层重要性的具体映射，以及它是否也保护序列开头。
- [ ] MultiAlloc 的 423 MB 对应什么序列长度/批量，换算出的等效 bit 是否确为 ~6.6。
- [ ] MultiAlloc 的 token 保护粒度（"组内 top-$k$"里的"组"是什么）。
- [ ] 三篇在 LongBench 上的逐任务表，用于和我们的 8 任务表对齐。
- [ ] KVmix vs MultiAlloc 在 LongBench-v2 上的逐项数字（目前只看到聚合）。

## 6. 定位建议（临时结论）

- ❌ **不要**再以"层间 H 分配 bit / 分配 token 数"作为创新点：三篇覆盖，且口径更成熟。
- ✅ **可以**把"2 bit 档位的效率结构"作为分析贡献：sink 62% + 其余弥散 + 纯 prefill 指标失明，这一组结论目前没有竞品覆盖。
- ⚠️ **需要先验证**再决定是否做"decode 期动态"：只有在 decode token 的注意力分布上存在类似 sink 的集中项时，这条路才有空间；否则会重复"弥散"的结论。

## 参考

- KVTuner — arXiv:2502.04420（ICML 2025），<https://arxiv.org/html/2502.04420v4>
- KVmix — arXiv:2506.08018（AAAI 2026 Oral），<https://arxiv.org/html/2506.08018>
- MultiAlloc — arXiv:2610.18642，<https://arxiv.org/html/2610.18642>，代码 <https://github.com/RL-MIND/MultiAlloc>
