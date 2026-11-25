# SinkBudget

**面向 KV Cache 量化的层间 Sink 保护预算分配与解码期延迟量化。**

[English](README.md) | 简体中文

> **项目状态（2026-11-22）：** SinkBudget 是 [KIVI](https://github.com/jy-yuan/KIVI) 的研究型 fork。上游 KIVI 基线当前可以运行；sink 相关的研究模块**尚未实现**。里程碑 **M0** 进行中：已确认 **Llama-first（L1）** 策略，Llama 模型配置与基线记录工具已就绪，静态检查通过；CUDA 编译、示例与基线重复运行仍需在 NVIDIA GPU 机器上执行。分阶段计划见 [roadmap](docs/kv_cache_sink_protection_roadmap.md)，M0 记录见 [M0 Llama 基线记录](docs/m0_llama_baseline.md)。

## 项目概述

KIVI 将 Key Cache 按通道量化、Value Cache 按 token 量化到 2/4 bit，并保留最近 `residual_length` 个全精度 token。SinkBudget 固定这套量化器不变，转而回答另一个问题：

> 在固定的 FP byte 预算下，**哪些 token 应该绕过量化**？**新生成的 token 应该在什么时候被量化**？

项目研究两个机制：

1. **层间异质性 sink 保护。** 用轻量离线校准估计每层有多少注意力质量集中在少数 sink token 上，再按层分配保护预算，而不是所有层保护相同数量的 token。
2. **解码期延迟量化。** 新生成的 token 先进入 FP 观察窗；滑出观察窗时，只有累计注意力分数足够高且动态槽位有空余才继续保留 FP，否则一次性量化为低比特，且不可逆。

目标 cache 形态是 token 级混合精度：**FP protected pool + KIVI quantized pool + recent FP window**，三段 attention 通过 online softmax 合并。

## 研究问题

1. 在相同有效 FP byte 预算下，基于层间 sink 依赖的 token 保护是否优于 uniform、random、reverse、attention-only 与 quant-error-only 分配？
2. 相比 KIVI 自带的 recent FP window，解码期延迟量化是否必要？尤其在长输出任务上。

## 方法速览

| 阶段 | 做什么 | 产出 |
|---|---|---|
| 离线校准 | 估计每层 sink mass `A_l`（top-`M` attention mass 的均值），可选叠加归一化 KIVI 重构误差 `E_l`；`L_l = A_l (1 + λ E_l)` | 每层预算 `P_l = S_l + D_l` |
| 在线 prefill | 在当前测试序列上重新计算 token 分数；保护 top `S_l` 个 token，位置 0 始终保护 | 每层的 FP 保护 token 集合 |
| Decode | 累计新 token 的注意力分数；token 滑出观察窗时决定保留 FP 或量化一次 | 动态 FP 槽位，FP → low-bit 单向 |
| Cache | FP protected pool + KIVI quantized pool + recent FP window | 分段 attention + online softmax 合并 |

设计约束：

- 被保护 token 的 **K 与 V 都保持 FP**；其余 token 走原始 KIVI 路径。
- 主实验采用 **scale-preserving protection**：保护行不改变非保护 token 的量化网格。recalibrated protection 只作工程消融。
- KIVI 的 K per-channel / V per-token 非对称设定保持不变；K-only / V-only 只用于诊断，不进入主实验矩阵。

### 主张边界

| 已实现阶段 | 允许的主张 |
|---|---|
| Fake-quant 原型 | 只报告精度变化与理论 FP byte 账本；不声称真实显存节省或加速 |
| Logical dual-pool cache | 逻辑显存占用、原型开销与正确性 |
| Segmented kernel | 端到端延迟、peak memory 与 tokens/s |

## 路线与当前状态

| 阶段 | 内容 | 状态 |
|---|---|---|
| M0 | 跑通原始 KIVI 项目；确认新测试模型；冻结基线 | 进行中 —— 非 GPU 部分完成，GPU 运行待执行 |
| M1 | Attention 统计与 sink 分数原型 | 未开始 |
| M2 | Fake-quant 保护与层间预算分配 | 未开始 |
| M3 | 主 baseline 矩阵与长上下文精度实验 | 未开始 |
| M4 | Decode 延迟量化与动态槽位仿真 | 未开始 |
| M5 | Logical dual-pool cache 与分段 attention | 未开始 |
| M6 | 效率系统化与 kernel 原型 | 未开始 |
| M7 | 消融、论文与复现包 | 未开始 |

M0.2 已确定：与用户确认 **L1** —— M1–M5 使用 `Llama-3.1-8B-Instruct`，`Llama-2-7b-hf` 作为 MHA 回归模型，Qwen2.5 后置到 M5.5。M0.1/M0.3 部分完成：静态检查通过，Llama 配置与 `scripts/m0_baseline.py` 已就绪；CUDA 扩展、示例脚本、LongBench smoke 与 3 次重复基线仍待 GPU 机器执行。在此之前，上游 KIVI 实现是唯一可执行的基线。

## 当前可运行的基线：KIVI

仓库当前包含上游 KIVI 实现：

- 模型：Llama-2 / Llama-3 系列（含 GQA），以及 Mistral / LongChat。
- 量化：`k_bits` / `v_bits` 取 2 或 4，可配置 `group_size`，并保留 `residual_length` 个最近 FP token。
- 路径：PyTorch eager attention 与 FlashAttention 两条路径。KIVI 要求 `config.use_flash = True`；Llama eager 路径只支持 MHA，GQA 模型必须走 flash 路径。
- Tokenizer：Llama-3.x 使用 fast BPE tokenizer（`tokenizer.json`）；Llama-2 / LongChat 保持 slow。`utils/tokenizer_utils.py` 负责选择，因为 Llama-3.x 根目录只有 `original/tokenizer.model`，强制 `use_fast=False` 会报 `TypeError: not a string`。
- Kernel：`quant/new_pack.py` 的 Triton 量化/打包，以及 `quant/` 下融合反量化与 GEMV 的 CUDA 扩展 `kivi_gemv`。

关于 cache 状态机、kernel 布局与二次开发扩展点的详细说明，见 [KIVI 项目结构与代码链路分析](docs/KIVI_项目结构与代码链路分析.md)。

## 安装

要求：Python ≥ 3.10、NVIDIA GPU、CUDA toolkit，以及 `pyproject.toml` 中锁定的依赖（`torch==2.4.1`、`transformers==4.43.1`、`flash-attn`）。`requirements.txt` 是旧版锁定文件，不作为依据。

```bash
source .venv/bin/activate

# 1) 先装 torch(cu124) 和 flash-attn 需要的构建工具
uv pip install --torch-backend=cu124 "torch==2.4.1" setuptools wheel ninja packaging

# 2) 关闭构建隔离，用环境里的 torch 编译 flash-attn
#    需要本机有 nvcc（CUDA toolkit 12.4），编译通常要几十分钟
uv pip install flash-attn --no-build-isolation

# 3) 装主包；flash-attn 已满足，不会重复构建
uv pip install --torch-backend=cu124 -e .

# 4) 编译 KIVI 的 CUDA 扩展（同样需要关闭构建隔离）
cd quant
uv pip install -e . --no-build-isolation
cd ..
```

模型默认从 **ModelScope** 拉取。传 `--model_source hf`（或在调用 `utils.modelscope_hub.resolve_model_path` 时传 `model_source="hf"`）可改回 Hugging Face Hub；模型 id 需要与所选源匹配。

**两种源的下载目录默认都在 `~/data` 下**（可用环境变量 `KIVI_DATA_DIR` 覆盖；显式传 `--cache_dir` 时仍以 `--cache_dir` 为准）：

```text
~/data/
├── modelscope/   # ModelScope 模型快照
├── hf/
│   ├── hub/      # Hugging Face 模型快照
│   └── datasets/ # Hugging Face 数据集缓存
└── datasets/
    ├── longbench/
    └── gsm8k/
```

数据集遵循同一规则（`--data_source hf` 才走 Hugging Face；不填表示跟随 `--model_source`）。**所有 loader 在下载之前都会先检查本机**，顺序为：

1. 显式目录：`--data_dir`，或环境变量 `LONGBENCH_DATA_DIR` / `GSM8K_DATA_DIR`，内含 `data/*.jsonl` 或 `data.zip`；
2. 缓存目录里已解压的副本（`--cache_dir`，默认 `~/data/datasets`）：`<cache_dir>/longbench/data/`、`<cache_dir>/gsm8k/`；
3. 本机已有的 ModelScope 或 Hugging Face snapshot（例如 `~/data/modelscope/hub/datasets/ZhipuAI/LongBench` 或 `~/.cache/modelscope/hub/datasets/ZhipuAI/LongBench`）。

三者都不存在时才会下载：LongBench 的 `data.zip`（约 114 MB，来自 ModelScope 数据集 `ZhipuAI/LongBench`），以及 `modelscope/gsm8k` 声明的各 split jsonl。`scripts/long_test.sh` 会传 `--cache_dir ./cached_models`，解压后的数据落在该目录下。默认路径不会访问 `huggingface.co`；模型权重同样先查 `--cache_dir`、`cached_models/`、ModelScope/HF 缓存与 `~/data`，重复运行不会重新下载。

## 快速开始

### 加载 KIVI 基线模型

```python
import torch
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer
from utils.modelscope_hub import resolve_model_path

# 默认使用 ModelScope 模型 id；改用 HF id 时传 model_source="hf"
model_path = resolve_model_path(
    "LLM-Research/Meta-Llama-3.1-8B-Instruct",
    model_source="modelscope",
)

config = LlamaConfig.from_pretrained(model_path)
config.k_bits = 2              # 2 或 4
config.v_bits = 2              # 2 或 4
config.group_size = 32
config.residual_length = 128   # 最近 FP token 数
config.use_flash = True        # KIVI 必需

model = LlamaForCausalLM_KIVI.from_pretrained(
    model_path,
    config=config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_path)

output = model.generate(**tokenizer("Hello", return_tensors="pt").to(model.device),
                        max_new_tokens=32)
```

Mistral 使用 `models.mistral_kivi` 中的 `MistralForCausalLM_KIVI` 与 `MistralConfig`。量化参数必须在加载模型**之前**写入 config。

| 参数 | 含义 | 典型值 |
|---|---|---|
| `k_bits` / `v_bits` | K/V cache 位宽 | 2、4 |
| `group_size` | 量化分组大小 | 32、64、128 |
| `residual_length` | 最近 FP token 数 | 32、128 |
| `use_flash` | 是否走 FlashAttention 路径 | `True`（必需） |

### 示例脚本

```bash
python example.py              # Llama-3.1-8B-Instruct GSM8K 5-shot
python long_context_example.py # passkey 检索
python mem_spd_test.py         # 耗时与 peak memory

# 三个脚本都支持 --model_name_or_path / --cache_dir / --model_source（默认 modelscope）
python example.py --model_source hf --model_name_or_path meta-llama/Llama-3.1-8B-Instruct
```

### LongBench

```bash
bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms
python eval_long_bench.py --model Llama-2-7b-ms_4096_2bits_group32_residual128
```

`scripts/long_test.sh` 的参数顺序是 `{GPU_ID} {K_BITS} {V_BITS} {GROUP_SIZE} {RESIDUAL_LENGTH} {MODEL} [MODEL_SOURCE] [DATA_DIR]`。`MODEL_SOURCE` 默认为 `modelscope`（Llama-2 base 的 ModelScope id 为 `modelscope/Llama-2-7b-ms`）；传 `hf` 可使用 `meta-llama/Llama-2-7b-hf` 这类 HF 模型 id；可选的 `DATA_DIR` 会转发为 `--data_dir`。预测结果写入 `pred/`，再由 `eval_long_bench.py` 评分。数据集与 prompt 配置位于 `config/`。

LongBench 数据首次从 ModelScope 下载并解压到 `./cached_models/longbench/`。如果机器上已有数据（例如从其他机器同步过来的副本），直接指向它，不会触发下载：

```bash
python pred_long_bench.py --model_name_or_path modelscope/Llama-2-7b-ms \
    --cache_dir ./cached_models --data_dir /path/to/longbench \
    --k_bits 2 --v_bits 2 --group_size 32 --residual_length 128
```

`--data_dir` 可以是含 `data/*.jsonl`、`data.zip` 或直接 `{task}.jsonl` 的目录；只要找到目标任务文件就不会联网。只有显式传 `--data_source hf` 才会走 Hugging Face 的 `THUDM/LongBench`（需要 HF 网络）。

### M0 基线记录

M0 的记录框架已就绪；GPU 相关运行仍需在 NVIDIA GPU 机器上执行：

```bash
# 无需 GPU：校验冻结的 KIVI 配置
python scripts/m0_baseline.py check \
    --k-bits 2 --v-bits 2 --group-size 32 --residual-length 128

# 记录 GPU / CUDA / PyTorch / Transformers / git 元数据
python scripts/m0_baseline.py env --out results/m0_baselines/env.json

# 重复 3 次；成功与失败尝试都会落盘
python scripts/m0_baseline.py run --name llama2_kivi2_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms
```

原始记录写入 `results/m0_baselines/`（已被 git 忽略）。整理后的状态、固定 KIVI 配置、L1 选型记录与 GPU TODO 清单见 [M0 Llama 基线记录](docs/m0_llama_baseline.md)。

## 仓库结构

```text
models/        Llama 与 Mistral 的 KIVI 实现（*_kivi.py）
quant/         Triton 量化/打包 + CUDA 扩展（kivi_gemv）
utils/         参数解析与数据工具
config/        模型路径、最大长度与 LongBench prompt
scripts/       long_test.sh、m0_baseline.py
docs/          研究方案、路线图、代码链路分析与 LongBench 结果
example.py, long_context_example.py, mem_spd_test.py
pred_long_bench.py, eval_long_bench.py, metrics.py
```

## 评测计划

所有对比需保持相同模型、tokenization、生成参数、KIVI bit 配置、recent window `R`、有效 FP byte 预算，以及上下文与生成长度分布：

- 层分数有效性：在代表性层上做小规模保护档位诊断（仅用于验证指标）。
- Prefill sink 保护：WikiText / PG19 长上下文 PPL、RULER / NIAH、LongBench 长输入短输出任务。
- Decode 延迟量化：GSM8K、MATH、MBPP / HumanEval、LongBench 长输出任务。
- Proxy 有效性：精确累计注意力分数与 key-similarity / activation proxy 的秩相关。
- 开销：校准前向、每步分数更新、槽位决策、metadata、逻辑 FP/quantized pool 显存、effective bits per token、tokens/s。

## 文档

- [研究方案](docs/KV%20Cache量化优化.md) —— 问题、方法、baseline、消融与风险。
- [工程路线图](docs/kv_cache_sink_protection_roadmap.md) —— M0–M7 的任务、交付物、验收标准与验证命令。
- [M0 Llama 基线记录](docs/m0_llama_baseline.md) —— L1 选型、固定 KIVI 配置、环境模板与 GPU TODO。
- [代码链路分析](docs/KIVI_项目结构与代码链路分析.md) —— cache 状态机、kernel 布局与扩展点。
- [LongBench 结果](docs/long_bench.md) —— 上游 KIVI 在 LongChat 与 Mistral 上的结果。

## 引用

SinkBudget 基于 KIVI 开展。若使用基线实现，请引用：

```bibtex
@article{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  journal={arXiv preprint arXiv:2402.02750},
  year={2024}
}
```

## 许可证

MIT，见 [LICENSE](LICENSE)。上游 KIVI 同样以 MIT License 发布。
