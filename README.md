# SinkBudget

**Layer-wise sink protection budgeting and decode-time delayed quantization for KV cache quantization.**

English | [简体中文](README.zh-CN.md)

> **Project status (2026-11-21):** SinkBudget is a research fork of [KIVI](https://github.com/jy-yuan/KIVI). The upstream KIVI baseline is runnable today; the sink-aware research modules are **not implemented yet**. The current milestone is **M0**: run the original project end to end and confirm the **Llama-first** model strategy (Qwen2.5 is deferred to M5.5). See the [roadmap](docs/kv_cache_sink_protection_roadmap.md) for the staged plan.

## Overview

KIVI quantizes the key cache per channel and the value cache per token to 2/4 bits, while keeping the most recent `residual_length` tokens in full precision. SinkBudget keeps that quantizer fixed and asks a different question:

> Given a fixed FP-byte budget, **which tokens should bypass quantization**, and **when should a newly generated token be quantized?**

The project studies two mechanisms:

1. **Layer-wise heterogeneous sink protection.** A light offline calibration estimates how much attention mass each layer concentrates on a few sink tokens. The budget is then allocated across layers, instead of protecting the same number of tokens everywhere.
2. **Decode-time delayed quantization.** Newly generated tokens first stay in an FP observation window. When a token slides out, it is kept in FP only if its accumulated attention score is high and a dynamic slot is available; otherwise it is quantized once, irreversibly.

The intended cache layout is a token-level hybrid: **FP protected pool + KIVI quantized pool + recent FP window**, with segmented attention combined through online softmax.

## Research questions

1. Under the same effective FP-byte budget, does layer-wise sink-dependency allocation outperform uniform, random, reverse, attention-only, and quant-error-only allocation?
2. Is decode-time delayed quantization necessary beyond KIVI's recent FP window, especially on long-output tasks?

## Method at a glance

| Stage | What happens | Output |
|---|---|---|
| Offline calibration | Estimate per-layer sink mass `A_l` (average top-`M` attention mass), optionally combine with normalized KIVI reconstruction error `E_l`; `L_l = A_l (1 + λ E_l)` | Per-layer budget `P_l = S_l + D_l` |
| Online prefill | Recompute token scores on the current sequence; protect the top `S_l` tokens, with position 0 always protected | FP protected token set per layer |
| Decode | Accumulate attention scores for new tokens; when a token leaves the observation window, keep it FP or quantize it once | Dynamic FP slots, one-way FP → low-bit |
| Cache | FP protected pool + KIVI quantized pool + recent FP window | Segmented attention with online softmax merge |

Design constraints:

- Protected tokens keep **both K and V** in FP; all other tokens follow the original KIVI path.
- Scale-preserving protection is the main setting: protected rows do not change the quantization grid of unprotected tokens. Recalibrated protection is an engineering ablation.
- KIVI's K per-channel / V per-token asymmetry is kept as-is. K-only or V-only protection is diagnostic, not part of the main matrix.

### Claim boundaries

| Implemented stage | Claims allowed |
|---|---|
| Fake-quant prototype | Accuracy change and theoretical FP-byte accounting only; no real memory savings or speedup claims |
| Logical dual-pool cache | Logical memory footprint, prototype overhead, and correctness |
| Segmented kernel | End-to-end latency, peak memory, and tokens/s |

## Roadmap and status

| Stage | Scope | Status |
|---|---|---|
| M0 | Run the original KIVI project end to end; agree on the new test model; freeze baselines | Not started |
| M1 | Attention statistics and sink-score prototype | Not started |
| M2 | Fake-quant protection and layer-wise budget allocation | Not started |
| M3 | Main baseline matrix and long-context accuracy experiments | Not started |
| M4 | Decode-time delayed quantization and dynamic-slot simulation | Not started |
| M5 | Logical dual-pool cache and segmented attention | Not started |
| M6 | Efficiency systematization and kernel prototype | Not started |
| M7 | Ablations, paper, and reproducibility package | Not started |

The current focus is M0.1 (verify the baseline project: environment, CUDA extension, examples, LongBench) and M0.2 (confirm the model strategy with the user: M1–M5 on Llama-3.1-8B, Qwen2.5 ported at M5.5). The upstream KIVI implementation remains the executable baseline for now.

## Baseline: what runs today

This repository currently contains the upstream KIVI implementation:

- Models: Llama-2 / Llama-3 family (including GQA) and Mistral / LongChat.
- Quantization: `k_bits` / `v_bits` in {2, 4}, configurable `group_size`, and a recent FP window of `residual_length` tokens.
- Paths: a PyTorch eager attention path and a FlashAttention path. KIVI requires `config.use_flash = True`; the eager Llama path only supports MHA, so GQA models must use the flash path.
- Kernels: Triton quantization/packing in `quant/new_pack.py` and the fused dequant + GEMV CUDA extension `kivi_gemv` in `quant/`.

For an in-depth walkthrough of the cache state machine, kernel layout, and extension points, see [KIVI 项目结构与代码链路分析](docs/KIVI_项目结构与代码链路分析.md) (Chinese).

## Installation

Requirements: Python ≥ 3.10, an NVIDIA GPU, CUDA toolkit, and the dependencies pinned in `pyproject.toml` (`torch==2.4.1`, `transformers==4.43.1`, `flash-attn`). `requirements.txt` is a legacy lock file and is not the source of truth.

```bash
conda create -n sinkbudget python=3.10
conda activate sinkbudget
pip install --upgrade pip
pip install -e .

# Build the CUDA extension (required for the KIVI cache path)
cd quant && pip install -e . && cd ..
```

## Quick start

### Load a KIVI baseline model

```python
import torch
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer

model_path = "meta-llama/Llama-3.1-8B-Instruct"

config = LlamaConfig.from_pretrained(model_path)
config.k_bits = 2              # 2 or 4
config.v_bits = 2              # 2 or 4
config.group_size = 32
config.residual_length = 128   # number of recent FP tokens
config.use_flash = True        # required by KIVI

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

For Mistral, import `MistralForCausalLM_KIVI` from `models.mistral_kivi` and use `MistralConfig`. Quantization parameters must be set on the config **before** loading the model.

| Parameter | Meaning | Typical values |
|---|---|---|
| `k_bits` / `v_bits` | K/V cache bit width | 2, 4 |
| `group_size` | Quantization group size | 32, 64, 128 |
| `residual_length` | Number of recent FP tokens | 32, 128 |
| `use_flash` | Use the FlashAttention path | `True` (required) |

### Examples

```bash
python example.py              # GSM8K 5-shot with Llama-3.1-8B-Instruct
python long_context_example.py # passkey retrieval
python mem_spd_test.py         # latency and peak memory
```

### LongBench

```bash
bash scripts/long_test.sh 0 2 2 32 128 meta-llama/Llama-2-7b-hf
python eval_long_bench.py --model Llama-2-7b-hf_4096_2bits_group32_residual128
```

`scripts/long_test.sh` takes `{GPU_ID} {K_BITS} {V_BITS} {GROUP_SIZE} {RESIDUAL_LENGTH} {MODEL}`. Predictions are written under `pred/`, and `eval_long_bench.py` scores them. Dataset and prompt settings live in `config/`.

## Repository layout

```text
models/        KIVI implementations for Llama and Mistral (*_kivi.py)
quant/         Triton quantization/packing + CUDA extension (kivi_gemv)
utils/         argument parsing and data helpers
config/        model paths, max lengths, and LongBench prompts
scripts/       long_test.sh
docs/          research plan, roadmap, code walkthrough, LongBench results
example.py, long_context_example.py, mem_spd_test.py
pred_long_bench.py, eval_long_bench.py, metrics.py
```

## Evaluation plan

Accuracy and efficiency will be compared under identical model, tokenization, generation parameters, KIVI bit configuration, recent FP window `R`, effective FP-byte budget, and context/generation length distributions:

- Layer-score validity: small-scale protection sweeps on representative layers (diagnostic only).
- Prefill sink protection: WikiText / PG19 perplexity, RULER / NIAH, LongBench long-input short-output tasks.
- Decode delayed quantization: GSM8K, MATH, MBPP / HumanEval, and LongBench long-output tasks.
- Proxy validity: rank correlation between exact cumulative attention scores and key-similarity / activation proxies.
- Overhead: calibration cost, per-step score updates, slot decisions, metadata, logical FP/quantized pool memory, effective bits per token, and tokens/s.

## Documentation

- [Research proposal (Chinese)](docs/KV%20Cache量化优化.md) — problem, method, baselines, ablations, risks.
- [Engineering roadmap (Chinese)](docs/kv_cache_sink_protection_roadmap.md) — M0–M7 tasks, deliverables, acceptance criteria, and verification commands.
- [Code walkthrough (Chinese)](docs/KIVI_项目结构与代码链路分析.md) — cache state machine, kernel layout, and extension points.
- [LongBench results](docs/long_bench.md) — upstream KIVI results for LongChat and Mistral.

## Citation

SinkBudget builds on KIVI. If you use the baseline implementation, please cite:

```bibtex
@article{liu2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Liu, Zirui and Yuan, Jiayi and Jin, Hongye and Zhong, Shaochen and Xu, Zhaozhuo and Braverman, Vladimir and Chen, Beidi and Hu, Xia},
  journal={arXiv preprint arXiv:2402.02750},
  year={2024}
}
```

## License

MIT. See [LICENSE](LICENSE). Upstream KIVI is also released under the MIT License.
