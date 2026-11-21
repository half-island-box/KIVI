# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 开发规范
1. 后续所有开发和改动，都在sink_budget分支上开展
2. docs/kv_cache_sink_protection_roadmap.md文档已经声明了每个阶段要做的内容，后续你的每次git提交的message，前缀都要声明属于哪个阶段m。
2. 不需要一个阶段的全部内容完成再进行一次提交，只要完成一个小模块，就可以进行一次git提交，防止某次提交体积太大。
3. 用户需要排查问题或者开发新模块，没有明确声明开始改动前，不要自己自作主张
4. 每次融入新的模块后，都要实时更新readme中相关的python启动指令，包括基本用法和bench
6. docs/kv_cache_sink_protection_roadmap.md文档中某个大阶段的任务完成后，把相应的用于验证和测试指标的python命令补充上

## 项目定位

本仓库是 KIVI（ICML 2024，免调优非对称 2/4-bit KV Cache 量化）的 fork，项目工作名为 **SinkBudget**，用于在其上做研究性扩展：**层间异质性的 attention-sink 保护预算分配 + decode 期延迟量化**。动手前先读两份设计文档：

- `docs/KV Cache量化优化.md` —— 研究方案：问题定义、token sink score / 层间分数、budget 分配、baseline 矩阵、消融、风险。
- `docs/kv_cache_sink_protection_roadmap.md` —— 分阶段工程路线 M0–M7，含每阶段任务、交付物、验收标准与验证命令。

注意：当前 checkout 是上游 KIVI 主干（Llama / Mistral 路径）；roadmap 中 M0 描述的 `benchmark.py`、`benchmarks/`、`models/qwen2_kivi.py`、`utils/modelscope_hub.py`、`tests/` 等模块在本分支尚未落地。以实际文件为准，不要假设这些入口已经存在。

工作约定：

- 用户需要排查问题或开发新模块时，未明确要求开始改动前，不要自行扩大改动范围。
- 每完成一个小模块就提交一次，避免超大提交；commit message 前缀写明所属阶段，例如 `m1: add attention score hook`。
- 新模块落地后同步更新 README 的启动命令（基本用法 + bench）；某个大阶段完成后，把用于验证的 python 命令补进 roadmap。
- 文档语言跟随原文件：README 与 docs/long_bench.md 为英文，设计/路线文档为中文。

## 环境与安装

- Python ≥ 3.10，建议独立 conda 环境；依赖以 `pyproject.toml` 为准（torch 2.4.1、transformers 4.43.1、flash-attn）。
- `requirements.txt` 是旧锁定文件（torch 2.1.2 / transformers 4.36.2），与 pyproject 存在漂移；除非用户明确要求，不要依据它降级依赖。
- 安装主包：`pip install -e .`
- 安装 CUDA 实现：`cd quant && pip install -e .`，从 `quant/csrc/` 编译出 `kivi_gemv`。需要可用 NVIDIA GPU、CUDA toolkit 与已安装的 torch；改动 `csrc/` 后必须重新安装。`*.so`、`build/` 不入库。
- KIVI 推理路径强制要求 `config.use_flash = True`（`LlamaAttention_KIVI.__init__` 中有 assert）；模型类不从 AutoModel 注册，直接 `from models.llama_kivi import LlamaForCausalLM_KIVI`（Mistral 同理），并在 `from_pretrained` 前设置 `config.k_bits/v_bits/group_size/residual_length/use_flash`。

## 常用命令

示例与冒烟测试（需要 GPU 与已下载模型）：

```bash
python example.py             # Llama-3.1-8B GSM8K 5-shot
python long_context_example.py  # passkey 检索，读取 passkey_examples.jsonl
python mem_spd_test.py        # 固定 batch/长度下的耗时与 peak memory
```

LongBench 完整流程：

```bash
bash scripts/long_test.sh {GPU_ID} {K_BITS} {V_BITS} {GROUP_SIZE} {RESIDUAL_LENGTH} {MODEL}
python eval_long_bench.py --model {MODEL}   # {MODEL} 是 pred/ 下的目录名
```

`pred_long_bench.py` 基于 `THUDM/LongBench`，由 `config/model2path.json`、`config/model2maxlen.json`、`config/dataset2prompt.json`、`config/dataset2maxlen.json` 驱动；默认跑 8 个任务，`--e` 切换到 LongBench-E 的 13 个任务。预测写入 `pred/`（已忽略），评分读取 `pred/` 或 `pred_e/`。

静态检查（仓库无正式测试框架，提交前至少执行）：

```bash
python -m compileall models quant utils metrics.py
```

量化算子自测（需要 GPU 与已编译的 `kivi_gemv`）：

```bash
python quant/gemv.py         # kivi_gemv 正确性 / 速度对照
cd quant && python test.py   # Triton pack/unpack 与 GEMV 对照
```

这两个脚本顶部硬编码了 `CUDA_VISIBLE_DEVICES=2`，运行前按机器情况调整。

## 架构要点

### 模型层：`*_KIVI` 是 HF modeling 代码的分支副本

- `models/llama_kivi.py`、`models/mistral_kivi.py` 复制了 transformers 4.43 的 modeling 代码，只把 attention 替换为 `Llama*Attention_KIVI` / `Mistral*Attention_KIVI`，并暴露 `*ForCausalLM_KIVI`。这也是依赖必须固定在 4.43.x 的原因之一。
- `LlamaDecoderLayer_KIVI` 根据 `config.use_flash` 选择 eager 或 flash attention 实现。eager 版 `LlamaAttention_KIVI` 断言 `num_key_value_groups == 1`（仅 MHA）；GQA/MQA 必须走 flash 路径（Mistral flash 路径用 `repeat_kv_quant` 展开 KV heads）。

### KV cache 契约（任何扩展都必须遵守）

每层 cache 是 9 元组：

```text
(k_quant_trans, k_full, k_scale_trans, k_mn_trans,
 v_quant, v_full, v_scale, v_mn, kv_seq_len)
```

- `k_*_trans`：K 量化后按 `(B, nh, head_dim, T/pack)` 转置存放，`cuda_bmm_fA_qB_outer` 直接消费该布局。
- `k_full` / `v_full`：最近 `residual_length` 个全精度 token。
- 最后一项是 int 逻辑长度：`LlamaModel_KIVI.forward` 用 `past_key_values[0][-1]` 取长度，`prepare_inputs_for_generation` 也依赖这一点。新增、删除或重排 cache 字段时，必须同步 `LlamaModel_KIVI.forward`、`prepare_inputs_for_generation`、`_reorder_cache` 以及所有构造该元组的位置。
- `residual_length` 需为 `group_size` 的整数倍（K 侧有断言）；prefill 按 `residual_length` 切分、decode 每步滑出一个 V token，这些边界是长上下文测试最容易出错的地方。

### 量化与 attention 数据流

- K 走 per-channel：沿 token 维分组统计量化参数，量化后转置打包；V 走 per-token：沿 head_dim 分组。2/4/8-bit 的量化与打包都在 `quant/new_pack.py`（Triton，`triton_quantize_and_pack_along_last_dim`）。
- prefill：把 K/V 切成“已量化段 + 最近 `residual_length` 个 FP token”。decode：K 的 FP 段攒满 `residual_length` 后整块量化追加；V 每步把最老的 1 个 token 量化进 quantized pool，始终保留 `residual_length` 个 FP token。
- attention 不会先整体 dequant：量化段的 QK^T 与 attn·V 由 `quant/matmul.py:cuda_bmm_fA_qB_outer`（底层 `kivi_gemv`，源码在 `quant/csrc/gemv_cuda.cu`）融合 dequant + GEMM 得到；FP 段用普通 `torch.matmul`；两段 logits 拼接后做一次 softmax，V 侧同理分段求和。flash 路径在 prefill 用 `flash_attn`，decode 仍走上述量化 GEMV。
- 因此 sink 保护 / 双池 / 延迟量化的注入点在 `triton_quantize_and_pack_along_last_dim` 调用前后（决定哪些 token 不进量化器），而不是 attention 输出端。

### 遗留代码，不要误用

`models/utils_quant.py` 与 `quant/qmodule.py` 是 KIVI 早期的权重量化 / fake-quant 工具，未接入当前 KV cache 路径（`models/utils_quant.py` 引用的 `dequant_cuda` 并未 import，直接调用会失败）。除非任务明确针对它们，否则以 `quant/new_pack.py` + `quant/matmul.py` 为准。研究方案里提到的 fake-quant 原型需要基于当前路径另行实现。

### LongBench 入口

`pred_long_bench.py` 使用 `utils/process_args.py` 的 dataclass 参数（`--k_bits/--v_bits/--group_size/--residual_length/--e` 等），按模型名子串分派 Llama/LongChat/Mistral；`k_bits < 16 and v_bits < 16` 时加载 KIVI 模型，否则加载 HF 全精度基线。`eval_long_bench.py` + `metrics.py` 负责评分。

## 仓库惯例与陷阱

- `.gitignore` 忽略 `*.json`、`*.jsonl`、`*.sh`、`*.pt`、`pred/`、`pred_e/`、`cached_models/`、`outputs/`、`logs/`。新增配置/脚本/数据前先确认是否需要 `git add -f`；不要提交模型缓存、预测产物或大型数据。
- 工作区可能有他人未提交的修改（如 `models/llama_kivi.py`、`quant/new_pack.py`）。除非用户明确要求，不要还原、格式化或重构这些改动，只在与任务相关的最小范围内编辑。
- 路线约定模型与数据集只走 ModelScope、不新增 HF Hub 回退（该策略在 roadmap M0 的 `utils/modelscope_hub.py` 中落地；当前代码仍直接使用 HF `datasets` / `from_pretrained`）。
- 显存：7B 模型 fp16 约需 14GB+；调试优先使用小模型、短序列或已缓存模型，避免不必要的大体积下载。
