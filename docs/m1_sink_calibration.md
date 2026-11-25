# M1：Attention 统计与 Sink 分数原型

> 对应路线图：`docs/kv_cache_sink_protection_roadmap.md` 的 M1。
> 状态日期：2026-11-25。CPU 可验证部分已完成；GPU / 真实模型上的 exact-proxy 相关性待执行。

---

## 1. 范围与约束

M1 只做“可观察”：

- 新增 `kivi_sink/` 与 `tests/`，**不修改** `models/`、`quant/`、`utils/`、`pred_long_bench.py`、`eval_long_bench.py`；
- 不在 KIVI attention 热路径插入统计分支，不放宽 `use_flash=True` 断言；
- 校准在独立进程内加载普通 Hugging Face 模型（eager 或 SDPA），KIVI 推理路径与 M0 基线不受影响；
- 模块 import 不修改全局随机种子、默认 dtype、`torch.backends.*`；bootstrap 使用本地 `torch.Generator`；
- 不新增依赖：只使用 `torch` / 标准库；模型与数据加载复用现有 `utils/modelscope_hub.py`，未改其行为。

## 2. 模块结构

```text
kivi_sink/
  __init__.py          # 包说明；import 无副作用
  attention_stats.py   # exact attention mass / 流式精确计算 / key-norm proxy
  sink_score.py        # top-M、A_l、跨样本聚合、bootstrap CI、Spearman
  calibration.py       # 校准 runner（exact / proxy / both / synthetic）
tests/
  test_attention_stats.py
  test_sink_score.py
  test_calibration.py
docs/examples/m1_synthetic_calibration.{json,csv}  # 输出 schema 示例（随机张量，非实验结果）
```

## 3. 统计口径

### 3.1 Token score

对 prefill 的每个 key token，统计其收到的 attention mass：

- exact：`softmax(causal QKᵀ/√d)`，在所选 query 范围与所有 head 上取平均，得到 `[T]` 分数；每行 attention 之和为 1，因此分数天然归一化。
- query 范围：`--query-fraction 1.0` 使用全部 query；`0.25` 只使用最后 25% query（向上取整，至少 1 个）。
- 按 batch sample 独立统计；批量输入用 `lengths` 排除 padding。

### 3.2 Proxy（flash-compatible）

默认 proxy 为 **key norm**：`‖k_t‖₂`，在 KV heads 上取平均。

- RoPE 是每个位置的（分块）正交旋转，因此 key norm 在 RoPE 前后不变；
- 校准进程只需 forward hook 读取 `self_attn.k_proj` 输出（hook 返回 `None`，不改输出），即可在 SDPA / flash attention 下工作；
- proxy 与 exact 的相关性由 `--mode both` 在同一 forward 中直接报告（逐层 token score 的 Spearman，以及逐样本层分数向量的 Spearman）。

### 3.3 层分数 A_l

每层取 token score 的 top-M，累加其占该层总分的比例：

```text
A_l = sum(top-M token scores) / sum(all token scores)
```

exact 分数本身已归一化，因此 `A_l` 就是 top-M attention mass 占比。跨校准样本报告 mean / median / std / bootstrap 置信区间，并给出样本间 `A_l` 排序的 Spearman 稳定性。

## 4. 校准数据与泄漏隔离

- 默认来源：LongBench `narrativeqa`（不属于 `pred_long_bench.py` 的默认 8 任务，也不属于 LongBench-E 13 任务）；
- `EVAL_TASKS` 中的任务默认拒绝，必须显式 `--allow-eval-task` 才可覆盖（用于诊断）；
- `--calibration-jsonl` 可指定独立校准集，每行 JSON 至少包含 `text`，或 `context` + `input`；
- 样本按固定顺序取前 `--samples` 条，保证重复运行可复现；不做随机采样；
- Llama-3-Instruct 默认套用与 LongBench 评测一致的 chat template（`--chat-template auto`）；已格式化文本可传 `--chat-template none`。

## 5. 验证命令

### 5.1 CPU 可执行（无需模型/GPU）

```bash
# 静态检查 + 单元测试（54 项：causal mask、归一化、top-M、批量/多头聚合、
# bootstrap 确定性、Spearman、hook 只读性、import 无副作用）
python -m compileall kivi_sink tests
python -m unittest discover -s tests -v

# 合成 Q/K 全链路 smoke（生成 JSON + CSV + 可选 token score npz）
python -m kivi_sink.calibration --synthetic \
    --samples 3 --synthetic-layers 4 --max-length 64 --top-m 2 \
    --output results/sink/calibration/synthetic.json \
    --save-token-scores results/sink/calibration/synthetic.npz

# 路径隔离检查：M1 不修改 KIVI / LongBench 默认执行路径
git diff 20aa8dd -- models quant pred_long_bench.py eval_long_bench.py utils
```

### 5.2 需要 GPU + transformers + 模型（待执行）

```bash
# exact：短窗口诊断（会为每层物化 [H, T, T] attention）
python -m kivi_sink.calibration \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --mode exact --max-length 512 --samples 8 \
    --output results/sink/calibration/llama31_exact.json

# both：同一次 eager forward 同时得到 exact 与 key-norm proxy，输出相关性
python -m kivi_sink.calibration \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --mode both --max-length 512 --samples 8 \
    --output results/sink/calibration/llama31_both.json

# proxy：长上下文（4096）校准，不物化 attention
python -m kivi_sink.calibration \
    --model LLM-Research/Meta-Llama-3.1-8B-Instruct \
    --mode proxy --max-length 4096 --samples 16 \
    --output results/sink/calibration/llama31_proxy.json
```

exact 模式的显存约为 `batch × heads × T² × 每层`，512 token、32 head、32 层的 fp32 attention 合计约 1 GB，适合做诊断；长上下文 `A_l` 应使用 proxy 模式，并先用 `both` 在短窗口验证 proxy 相关性。

## 6. 输出 schema

JSON 顶层字段：

| 字段 | 含义 |
|---|---|
| `schema_version` | 当前为 1 |
| `metadata` | git commit/branch/dirty、python/torch/transformers/flash-attn 版本、device、dtype、模型路径与层数 |
| `config` | 本次 CLI 参数全量回显 |
| `samples` | 每个样本的 `id`、`num_tokens`、`exact_A` / `proxy_A`（逐层 `A_l`） |
| `samples_skipped` | 长度小于 top-M 而跳过的样本 id |
| `layers` | 逐层 `A_l` 的 mean/median/std/CI、exact-proxy token Spearman |
| `correlation` | proxy 名称、逐层 token Spearman、逐样本层分数 Spearman、跨样本稳定性 |

`samples` 中的 `A_l` 与 `layers` 的顺序一致（按解码层号升序）。完整 token score 通过 `--save-token-scores out.npz` 另存，键名为 `{mode}_sample{i}_layer{l}`。

示例见 `docs/examples/m1_synthetic_calibration.json`（由随机 Q/K 生成，仅示意 schema）与同名 `.csv`。

## 7. 当前验收状态

- [x] 每个样本每层可输出 token score（`--save-token-scores`）；
- [x] 可汇总 `A_l` 与跨样本稳定性（mean/median/std/bootstrap CI/样本间 Spearman）；
- [x] exact 与 proxy 的 Spearman 报告路径可用（合成数据已跑通；真实模型数值待 GPU）；
- [x] 校准不改变原模型输出（hook 只读、`torch.no_grad()`、独立进程；单元测试覆盖）；
- [x] 路径隔离：`kivi_sink/`、`tests/` 之外无执行路径改动；
- [ ] Llama-3.1-8B / Llama-2-7b 上的 exact-proxy 相关性数值（待 GPU）；
- [ ] M0 LongBench 与 `mem_spd_test.py` 回归（待 GPU，M1 未改相关路径）。

## 8. 已知限制

1. exact 模式依赖 HF `output_attentions=True`，会物化每层 attention；默认 `--max-length 512` 是有意保守的设置。
2. proxy 模式假设模型具有 `layers.<i>.self_attn.k_proj`（Llama / Mistral / Qwen2 结构）；其他结构需新增 proxy。
3. 校准默认使用单条样本、batch=1、无 padding；left padding 不在 M1 阻塞项内。
4. key norm 只是第一个 proxy；M1 的相关性报告用于判断是否需要更强 proxy（如 logit-LSE 或 activation outlier）。
