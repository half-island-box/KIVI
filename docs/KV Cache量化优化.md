## 选题名称（暂定）

**面向 KIVI KV Cache PTQ 的层间异质性 Sink 保护预算分配与解码期延迟量化

## 1. 问题与动机

KV Cache 低比特 PTQ 中，attention sink token 对长上下文性能很重要，但现有保护策略往往全局统一：固定保护前 $N$ 个 token，或所有层使用相同保护强度。StreamingLLM 说明了保留初始 sink token 与滑动窗口对稳定长上下文生成的作用 [3]；KVSink 进一步指出仅保护初始 token 不足以覆盖所有 sink，并讨论了量化场景下的 sink 保护 [4]。本方案进一步假设：不同层对 sink 误差的敏感度不同，因此统一分配可能造成部分层保护不足、部分层浪费 FP 预算。这一层间异质性假设将通过实验验证。

**与 MultiAlloc 的关系（2026-10 新增）。** Zhou 等提出的 MultiAlloc [16] 同样把 KV cache 量化视为固定显存预算下的精度分配问题：离线用一阶梯度加权的量化误差刻画层敏感度，用 K 激活的 L1 范数刻画 token 重要性，在“层比特宽度”与“组内 top-$k$ token 高精度保留”之间分配预算，推理期执行一个与输入无关的静态策略。该工作说明“层间 + token 级联合预算分配”本身已不再是空白，因此本选题的定位需要收紧到 MultiAlloc 未覆盖的三点：

1. **sink 特定的重要性定义**：MultiAlloc 的 token 分数是 K 激活范数，不区分 attention sink；本文以 prefill 注意力质量定义 token sink score，并显式估计层间 sink 依赖 $A_l$。
2. **在线 token 选择**：MultiAlloc 在每个 token 组内按静态分数取 top-$k$；本文离线只分配每层预算 $P_l$，具体保护哪些 token 在 prefill 阶段按当前序列的注意力分布重新选择，避免把校准样本的 token 位置迁移到测试序列。
3. **decode 延迟量化**：MultiAlloc 对新生成 KV 按策略立即量化；本文让新 token 先进入 FP 观察窗，滑出时再依据累积 sink score 做单向 keep/quantize 决策，并支持动态槽位替换。

相应地，M3 主矩阵应把 MultiAlloc 作为外部 baseline：优先在相同 KV 预算与相同量化后端下复现；若其实现绑定自身的层比特配置，则报告原文数字并明确标注预算口径差异，不把不同口径的数字直接混表。

同时，如果保护集合在 prefill 后完全冻结，decode 新生成的 token 会很快进入低比特 cache。长输出任务中，后期生成 token 也可能被后续 query 持续访问；若没有观察期，其量化误差会持续影响生成。

本选题不训练模型，在 KIVI 上插入轻量的 **token 级层间保护预算分配模块** 与 **解码期延迟量化机制**

> **离线估计层间 sink 依赖并分配预算；在线 prefill 为当前序列选择具体 sink token；decode 新 token 先进入 FP 观察窗，滑出时再决定保留或量化。**

核心问题：

1. 在相同 FP byte budget 下，基于层间 sink 依赖分数的 token 保护是否优于 uniform、random、reverse 和单一指标分配？
2. 新生成 token 是否需要延迟量化与动态保留，而不是仅依赖 KIVI 默认 recent FP window？

## 2. 底座选择：为什么主实验使用 KIVI

主量化器采用 KIVI，原因如下：

1. KIVI 已有 K per-channel、V per-token 的非对称量化设定 [1]；本文将其作为固定基础，不重新设计 K/V 量化策略。
2. KIVI 已有 recent FP window / residual cache 思想 [1]，与本文 decode 观察窗衔接自然。
3. 与 KVQuant 的 pre-RoPE key 与非均匀量化设计 [2]、TurboQuant/OSCAR 等旋转类设计 [5][6]、HqeKV 的混合压缩思路 [7] 相比，KIVI 的量化路径更透明 [1]，便于把收益归因到 sink-aware allocation。
4. 可按 fake-quant → logical dual-pool → kernel prototype 分阶段推进，工程风险可控。

TurboQuant 不作为第一版主底座：其 randomized Hadamard rotation 与向量量化格式 [5] 会增加 token 级 FP row 的表示和实现复杂度；此外，现有 attention kernel 通常面向同构 KV layout [8]，这也是本项目的工程判断而非论文事实主张。TurboQuant 只作为可选 fake-quant 泛化实验或后续系统工作 [5]。

## 3. 核心创新点

1. **轻量层间 sink 依赖衡量**：只用少量校准前向和 attention 统计估计每层 sink 依赖，不逐层扫描 PPL 或训练参数。
2. **固定 KIVI 设定下的 token 级预算分配**：不改变 KIVI 的 K per-channel、V per-token 量化；离线决定每层保护多少 token，在线 prefill 决定具体保护哪些 token。
3. **attention 与量化误差的结合分数**：主版本可用 sink attention mass，增强版本可加入 KIVI fake-quant 重构误差，避免只看注意力集中度。
4. **decode 延迟量化**：新 token 先进入 FP 观察窗，滑出时按累积 sink score 做一次不可逆 keep/quantize 决策；已量化 token 不恢复 FP。
5. **token 级 dual-pool cache**：每层维护 FP protected pool、KIVI quantized pool 与 recent FP window，并用 online softmax 合成分段 attention。

## 4. 技术路线

### 4.1 两阶段职责划分

方法明确区分“离线分配预算”和“在线选择 token”：

```text
Offline calibration:
    original KIVI process
    estimate layer-wise sink dependency
    allocate protection budget P_l for each layer

Online prefill:
    compute token sink scores on the current sequence
    select S_l concrete tokens to protect in layer l

Decode:
    keep new tokens FP in a recent observation window
    when a token slides out, keep it FP only if its score is high
```

离线产物是层数长度的预算向量：

$$
\{P_1,P_2,\ldots,P_L\}
$$

而不是直接输出某个测试序列的保护 token 集合。测试序列的具体保护 token 在 prefill 阶段重新按当前注意力分布选择。

### 4.2 Token sink score

**来源说明**：下面的 sink score 不是直接照搬某篇论文的定义，而是本文为“token 被多少 prefill 注意力质量吸收”给出的操作化指标。其动机来自 attention sink 现象：少数 token 会获得异常高的注意力 [3]，且这些 token 在 KV Cache 量化中需要特殊保护 [4]。对第 $l$ 层、第 $n$ 个校准样本或当前输入序列，token $i$ 的 sink score 定义为：

$$
s_{l,i}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\sum_h
\alpha_{q,h,i}
$$

其中：

- $l$ 是层编号；
- $i$ 是历史 token；
- $q$ 是 prefill query；
- $h$ 是 attention head；
- $\alpha_{q,h,i}$ 是 query $q$ 在 head $h$ 上分配给 token $i$ 的注意力权重。


### 4.3 层间 sink 依赖分数

对每个校准样本、每一层，本文用 top-$M$ attention mass 衡量层间 sink 依赖；该指标受 attention sink 现象启发 [3][4]，但具体的 top-$M$ 聚合形式为本文设计：

1. 计算所有 token 的 $s_{l,i}$；
2. 取 top-$M$ token，$M$ 可取 8 或 16；
3. 累加这些 token 吸收的注意力质量，得到该层在该样本上的 sink mass：

$$
A_{l,n}
=
\sum_{i\in \mathrm{Top}M_l}
s_{l,i}
$$

跨 $N$ 个校准样本平均：

$$
A_l
=
\frac{1}{N}
\sum_{n=1}^{N}
A_{l,n}
$$

$A_l$ 表示第 $l$ 层平均有多少注意力集中在少数 sink token 上。$M$ 只是估计层间依赖的统计窗口，不等于最终每层保护 token 数。

### 4.4 可选量化误差增强

只看 attention mass 可能漏掉“注意力集中但 KIVI 量化误差不大”的层。因此，本文在 KIVI fake-quant [1] 的基础上加入轻量归一化重构误差作为增强指标：

$$
E_l^K
=
\frac{\|\hat{K}_l-K_l\|_2^2}{\|K_l\|_2^2}
$$

$$
E_l^V
=
\frac{\|\hat{V}_l-V_l\|_2^2}{\|V_l\|_2^2}
$$

$$
E_l=E_l^K+E_l^V
$$

组合层分数为：

$$
L_l=A_l(1+\lambda E_l)
$$

其中 $\lambda$ 是固定超参数，可在少量校准样本上选择，如 $\{0,0.5,1\}$。若追求最简版本，可令 $\lambda=0$，即只用 attention mass。

### 4.5 层间预算分配

给定总保护槽位预算 $P$，按归一化层分数比例分配：

$$
P_l
=
\operatorname{round}
\left(
P
\frac{L_l}{\sum_{j=1}^{L}L_j}
\right)
$$

再用最大余数法保证：

$$
\sum_{l=1}^{L}P_l=P
$$

并施加约束：

$$
1\le P_l\le P_{\max}
$$

每层将 $P_l$ 拆分为静态槽位与动态槽位：

$$
P_l=S_l+D_l
$$

默认动态槽位占比为 $\eta$，如 10%–20%：

$$
D_l=\max(1,\operatorname{round}(\eta P_l))
$$

$$
S_l=P_l-D_l
$$

全局 recent FP window 大小为 $R$，不占 $P_l$，但必须计入总 FP 内存。每层 FP token 数上界为：

$$
S_l+D_l+R
$$

有效 FP byte budget 为：

$$
B^{\mathrm{FP}}
=
\sum_l
(S_l+D_l+R)
(d_l^K+d_l^V)
$$

其中 $d_l^K,d_l^V$ 是第 $l$ 层单个 token 的 FP K/V byte 数。

### 4.6 KIVI 量化设定与保护语义

| 分量 | 非保护 token | selected sink token |
|---|---|---|
| K | KIVI per-channel quantization [1] | FP |
| V | KIVI per-token quantization [1] | FP |

保护粒度是 token：被选中 token 的 K/V 均绕过 KIVI quantizer；其他 token 按原始 KIVI 路径量化。K/V 不对称性由 KIVI 原设计处理，不作为本文主变量。K-only / V-only 只做小规模诊断，不做逐层搜索。

### 4.7 KIVI scale 与 protected row

主实验采用 **scale-preserving protection**：先按原始 KIVI 流程计算量化参数；非保护 token 用原参数 quantize/dequantize；protected token 直接保持 FP。这样可将“保护 token”和“改变非保护 token 量化网格”解耦。

系统实现可额外报告 **recalibrated protection**：protected rows 不参与 scale 统计，非保护 tokens 重新计算 scale。该版本只作工程消融。

### 4.8 在线 prefill 静态保护

对当前输入序列重新计算 token score：

$$
s_{l,i}^{\mathrm{test}}
=
\frac{1}{|Q|}
\sum_{q\in Q}
\sum_h
\alpha_{q,h,i}
$$

第 $l$ 层选择：

$$
\mathrm{Protect}_l
=
\operatorname{Top}S_l
\{s_{l,i}^{\mathrm{test}}\}
$$

参照 StreamingLLM 对初始 sink token 的观察 [3]，通常强制位置 0 进入保护集合；同时参照 KVSink 的结论，不把保护集合限制为初始前缀 [4]：

```text
position 0 always protected
remaining S_l - 1 tokens selected by current-sequence attention score
```

被选中 token 的 K/V 进入 FP protected pool；未选中 token 按 KIVI 量化。

### 4.9 Decode 延迟量化

1. **每步更新分数**：

$$
\mathrm{score}_i
\leftarrow
\mathrm{score}_i+
\sum_h\alpha^{(h)}_{t,i}
$$

只更新统计，不做精度决策。若 attention 不可得，用 key similarity 或激活异常值 proxy，并报告与精确分数的 Spearman 相关。

2. **滑出观察窗时决策**：

```text
if score_i > threshold and D_l has space:
    keep K/V FP
elif score_i > min_score_in_dynamic_slot:
    evict the lowest-score FP token and quantize it by KIVI
    keep token i FP
else:
    quantize token i by KIVI
```

决策单向：FP → low-bit。已量化 token 不恢复 FP；动态槽位中的 FP token 可被替换出去，并在替换时降为低比特。

3. **每 $T$ 步低频重排**：按近期分数移出低分 FP token 并量化，腾出槽位。第一版可只做“槽位满时替换”。

## 5. Token 级混合精度 Cache 与 attention 路径

### 5.1 Fake-quant accuracy prototype

先不改 kernel：对 protected token 保留 FP K/V；对其他 token 做 KIVI quantize → dequantize 后拼回 FP tensor，用普通 attention 计算。该阶段验证算法收益，不声称真实显存节省或加速。

### 5.2 Logical dual-pool cache

每层维护：

```text
FP protected pool + KIVI quantized pool + recent FP window
```

attention 分为 FP segment 与 quantized segment。online softmax 的分块合并形式来自 FlashAttention 的标准思想 [8][9]。设两段 softmax 最大值、归一化系数和输出为 $(m_{\mathrm{FP}},l_{\mathrm{FP}},o_{\mathrm{FP}})$、$(m_q,l_q,o_q)$，则合并公式为：

$$
m=\max(m_{\mathrm{FP}},m_q)
$$

$$
l=
e^{m_{\mathrm{FP}}-m}l_{\mathrm{FP}}
+
e^{m_q-m}l_q
$$

$$
o=
e^{m_{\mathrm{FP}}-m}o_{\mathrm{FP}}
+
e^{m_q-m}o_q
$$

### 5.3 Segmented kernel prototype

最后修改 kernel，支持 per-layer scattered FP index、FP/quantized 双格式、online softmax、decode slot 更新与 RoPE 一致性。这是主要工程风险；若 8 周内无法完成，论文限定为 fake-quant + logical dual-pool + theoretical memory accounting，不声称生产级 vLLM 集成。

## 6. 实验设计

### 6.1 模型与配置

首选 Qwen2.5-7B-Instruct 或 Llama-3-8B-Instruct。主量化器为 KIVI 2-bit/3-bit [1]。所有方法保留相同 recent window $R$。可选第二量化器：KVQuant [2] 或 TurboQuant fake-quant [5]，仅用于泛化验证。

### 6.2 公平对比前提

所有方法使用相同模型、tokenization、生成参数、KIVI bit 配置、recent window $R$、有效 FP byte budget、上下文与生成长度分布，并报告 metadata 开销。

### 6.3 主 baseline 矩阵

| 方法                          | 层间分配依据                   | sink token 精度   | sliding window | 动态槽位 |
| --------------------------- | ------------------------ | --------------- | -------------: | ---: |
| FP16 / BF16 KV              | 无                        | 全 FP            |              有 |    无 |
| KIVI default                | 无                        | 无额外保护           |              有 |    无 |
| Uniform-N                   | 每层相同预算                   | selected K/V FP |              有 |    无 |
| Attention-only allocation   | $A_l$                    | selected K/V FP |              有 |    无 |
| Quant-error-only allocation | $E_l$                    | selected K/V FP |              有 |    无 |
| Combined allocation         | $L_l=A_l(1+\lambda E_l)$ | selected K/V FP |              有 |    无 |
| Random allocation           | 随机                       | selected K/V FP |              有 |    无 |
| Reverse allocation          | 低依赖层多得预算                 | selected K/V FP |              有 |    无 |
| Full method                 | Combined allocation      | selected K/V FP |              有 |    有 |
| Oracle token selection      | 当前样本真实注意力                | selected K/V FP |              有 |   可选 |
| MultiAlloc（外部对照） | 层敏感度 + token K 范数，离线静态策略 | 组内 top-$k$ 高精度 | 有 | 无 |

Random / Reverse 证明收益不是任意额外 FP 存储；Attention-only / Quant-error-only 证明组合指标是否有必要；Oracle 用于估计当前排序规则与上界差距。MultiAlloc [16] 是当前最接近的外部工作，需在统一显存口径下比较；若其量化后端与 KIVI 不同，只作为参考行并在表注中说明配置差异。

### 6.4 K/V 诊断性消融：非主实验

主实验固定 selected token 的 K/V 均 FP。K-only / V-only 只在少数敏感层或单一预算下可选测试，用于解释收益来自 key、value 还是共同作用，不进入主实验矩阵。

### 6.5 验证指标

1. **层分数有效性**：在 4–8 个代表层上做小规模保护档位诊断，如 $N_l\in\{0,2,8\}$，检查 $L_l$ 与性能改善的 Spearman 相关。该诊断只用于验证指标，不参与预算分配。
2. **prefill sink 保护**：WikiText / PG19 长上下文 PPL、RULER / NIAH [10]、LongBench 长输入短输出任务 [11]。
3. **decode 延迟量化必要性**：GSM8K [12]、MATH [13]、MBPP [14] / HumanEval [15]、LongBench 长输出任务 [11]；比较 Combined allocation + sliding 与 Full dynamic method。NIAH 输出短，只用于 prefill 验证 [10]。
4. **proxy 有效性**：报告 key similarity / activation proxy 与 exact cumulative attention score 的 Spearman 相关。activation/outlier proxy 的动机参考 KVSink 对 sink 激活异常的观察 [4]。
5. **开销与 memory**：校准前向开销、prefill 检测延迟、每步 score 更新、滑出决策、槽位重排、metadata、FP/quantized pool 显存、effective bits per token、tokens/s。

### 6.6 消融设置

- 总预算 $P$；
- top-$M$ 统计窗口；
- 量化误差权重 $\lambda$；
- 动态槽位比例 $\eta$；
- recent window $R$；
- decode 重排间隔 $T$；
- 校准样本数；
- attention score 使用全部 query 还是最后 25% query。

## 7. 时间安排（8 周）

- **第 1 周**：复现 KIVI baseline，梳理 K/V cache 数据流，搭建 attention/activation hook。
- **第 2 周**：实现 token sink score 与离线层间依赖分数；实现 fake-quant protected token；跑 oracle top-$1/2/4/8$ 初步验证。
- **第 3 周**：实现 attention-only、quant-error-only、combined allocation；生成层间预算向量并检查稳定性。
- **第 4 周**：跑主 baseline 与小规模代表层诊断，验证层分数和保护收益的相关性。
- **第 5 周**：实现 decode 观察窗、在线 score、滑出决策与动态槽位 simulation。
- **第 6 周**：实现 logical dual-pool cache，验证索引、动态更新、causal mask 与 online softmax。
- **第 7 周**：跑长上下文 PPL、RULER/NIAH/LongBench、GSM8K/MBPP/HumanEval。
- **第 8 周**：完成 memory/latency 分析、消融整理与论文初稿。

## 8. 风险与备选

1. **层间依赖差异不显著**：若 $A_l$ 跨层差异小，加入 $E_l$ 或使用 head 级统计；仍不显著则把贡献定位为 sink 保护的实证分析与 decode 延迟量化。
2. **attention mass 不预测量化代价**：用小规模代表层诊断验证；必要时调整 top-$M$、query 范围或加入量化误差项。
3. **校准样本间不稳定**：增加样本数或按 median 聚合；报告预算向量在 bootstrap 子集上的稳定性。
4. **KIVI kernel 不支持 scattered FP rows**：按 fake-quant → logical dual-pool → segmented kernel 推进；若高性能 kernel 未完成，不退回整层粒度，只报告 logical prototype 与理论 memory。
5. **attention 不可得**：小模型/短序列用 eager attention 获得精确分数；decode 用 proxy 并报告秩相关；若相关性低，动态槽位降级。
6. **动态槽位收益不显著**：扫描 $R$，增加长输出任务；若仍无收益，动态槽位作为可选增强。
7. **固定预算随上下文长度含义变化**：主实验用固定绝对 budget，附录补充比例预算 $B^{\mathrm{FP}}=\alpha B^{\mathrm{cache}}(L)$。
8. **与 KIVI recent window 混淆**：所有 baseline 使用相同 $R$，主表报告 effective FP bytes。
9. **与 MultiAlloc 的创新性重叠**：MultiAlloc [16] 已提出固定预算下层间精度与 token 保留的联合分配。应对：把主张收敛为 (a) sink 特定分数、(b) 在线 prefill token 选择、(c) decode 延迟量化与动态槽位；主矩阵加入 MultiAlloc 对照，并增加“静态组内 top-$k$ vs 在线选择”的消融，证明差异来自本文机制而非额外 FP 预算。

## 9. 论文贡献表述

1. 提出轻量的 KV Cache sink 依赖层间衡量方法，只需少量前向 attention 统计，不逐层扫描 PPL 或训练参数；相比 MultiAlloc [16] 依赖梯度的一阶敏感度刻画，校准成本更低。
2. 在不改变 KIVI K per-channel、V per-token 量化设定的前提下，实现 token 级层间保护预算分配。
3. 将“离线决定每层保护多少 token”和“在线 prefill 决定保护哪些 token”解耦，避免把校准样本的 token 位置直接迁移到测试序列。
4. 提出 decode 延迟量化机制，避免新 token 在重要性未知时直接低比特存储。
5. 设计 FP protected pool + KIVI quantized pool + recent window 的 token 级混合精度 cache，并给出分阶段实现路径。

---

## 10. 参考文献

### 10.1 引用与来源说明

- **来自引用论文的知识**：KIVI 的 K per-channel、V per-token 量化和 recent FP window [1]；KVQuant 的 pre-RoPE key 与非均匀量化 [2]；StreamingLLM 的初始 attention sink 与滑动窗口现象 [3]；KVSink 关于初始前缀保护不足和 sink 激活异常的观察 [4]；TurboQuant 的旋转与向量量化 [5]；OSCAR 的旋转类 KV 量化 [6]、HqeKV 的混合压缩思路 [7]、FlashAttention/FlashAttention-2 的 online softmax 与 blocked attention 思想 [8][9]；MultiAlloc 的层敏感度/Token 重要性 profiling 与固定预算联合分配框架 [16]。
- **本文提出的指标与机制**：token sink score 的平均 attention 聚合形式、最后 25% query 启发式、top-$M$ 层间 sink mass、量化重构误差组合分数、按比例层间预算分配、静态/动态槽位拆分、decode 延迟量化与动态保留策略、token 级 dual-pool cache。
- **工程判断**：KIVI/TurboQuant 现有 kernel 对 scattered FP rows 的支持程度需在复现阶段确认；同构 layout 是当前系统实现的约束假设，不作为论文结论。

### 10.2 文献列表

#### 主基础量化器

[1] Liu et al. KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. ICML, 2024.

[2] Hooper et al. KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. NeurIPS, 2024.

#### Attention Sink 与保护策略

[3] Xiao et al. Efficient Streaming Language Models with Attention Sinks. ICLR, 2024.

[4] Su et al. KVSink: Understanding and Enhancing the Preservation of Attention Sinks in KV Cache Quantization for LLMs. arXiv:2508.04257, 2025.

#### 其他量化器与工程参考

[5] Ashkboos et al. TurboQuant: Optimal Vector Quantization for KV Cache. ICLR, 2026.

[6] Zhou et al. OSCAR: Offline Spectral Covariance-Aware Rotation for 2-bit KV Cache Quantization. arXiv:2605.17757, 2026.

[7] Wang et al. HqeKV: Towards Hybrid Quantization and Eviction for KV Cache in Long-Context LLM Inference. Findings of ACL, 2026.

#### Online attention 与 benchmark

[8] Dao et al. FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. NeurIPS, 2022.

[9] Dao. FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. ICLR, 2024.

[10] Hsieh et al. RULER: What's the Real Context Size of Your Long-Context Language Models? 2024.

[11] Bai et al. LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding. ACL, 2024.

[12] Cobbe et al. Training Verifiers to Solve Math Word Problems. 2021.

[13] Hendrycks et al. Measuring Mathematical Problem Solving With the MATH Dataset. 2021.

[14] Austin et al. Program Synthesis with Large Language Models. 2021.

[15] Chen et al. Evaluating Large Language Models Trained on Code. 2021.

#### 层间/多粒度精度分配

[16] Zhou et al. MultiAlloc: Budget-Aware Precision Allocation for KV Cache Quantization. arXiv:2610.18642, 2026.
