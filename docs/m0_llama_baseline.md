# M0：Llama 基线跑通、选型与冻结记录

> 状态日期：2026-11-22（2026-11-26 更新）
> 所属路线：`docs/kv_cache_sink_protection_roadmap.md` M0
> 执行模式：**L1（已与用户确认）；2026-11-26 起只在 `Llama-3.1-8B-Instruct` 上做实验，Llama-2-7b 暂停**
> 当前进度：**`baseline_v1` 已指定并冻结（第 3.1 节）：Llama-3.1-8B 的 LongBench 三配置分数（FP16 51.32 / KIVI-2bit 50.73 / KIVI-4bit 51.18）为 M1–M5 的对照基线；该版本为单次评分，后续有时间再重跑并发布 `baseline_v2`。示例链路（`example.py` / `long_context_example.py` / `mem_spd_test.py`）与峰值显存仍未执行**

本文件是 M0 的落地记录，覆盖 M0.1（Llama 原始项目跑通）、M0.2（模型策略确认）与 M0.3（基线冻结）。凡是标注“待 GPU”的项目，都需要在具备 NVIDIA GPU、CUDA toolkit 与完整依赖的机器上执行，并把结果补回本文件。

---

## 1. 当前结论

| 项目 | 状态 | 说明 |
|---|---|---|
| L1 模型策略 | 已确认 | M1–M5 使用 `meta-llama/Llama-3.1-8B-Instruct` 作为唯一模型；`Llama-2-7b`（原 MHA 回归）自 2026-11-26 起暂停；Qwen2.5 后置到 M5.5 |
| M0 未引入 Qwen 代码 | 已验证 | 仓库内 `*.py` / `*.json` / `*.sh` 无 `qwen` 引用 |
| 静态检查 | 已通过 | `python -m compileall models quant utils metrics.py`（2026-11-22，macOS / CPU-only） |
| Llama 模型配置 | 已完成 | `config/model2maxlen.json`、`config/model2path.json` 已补 Llama-3.1-8B 与 Llama-2-7b |
| Llama 兼容性修复 | 已完成 | Llama-2 默认 id 改为 `modelscope/Llama-2-7b-ms`；Llama-3.x 使用 fast tokenizer（`utils/tokenizer_utils.py`） |
| 基线记录工具 | 已完成 | `scripts/m0_baseline.py`（`check` / `env` / `run`） |
| CUDA 算子编译 | 已实际验证 | 未单独跑 `quant/gemv.py` 自测，但 KIVI-2bit / KIVI-4bit 的 LongBench 预测（`rc=0`）走的就是 `kivi_gemv` 融合 dequant+GEMM，说明扩展已编译且可用；独立自测仍建议补跑 |
| 示例链路 | 未执行 | `example.py`（GSM8K 5-shot）、`long_context_example.py`（passkey）、`mem_spd_test.py`（耗时与峰值显存）均未运行 |
| LongBench 预测与评分 | 已完成 | Llama-3.1-8B 的 FP16 / KIVI-2bit / KIVI-4bit，8/8 任务，已用 `eval_long_bench.py` 评分（第 6.3 节） |
| **`baseline_v1`（基线冻结）** | **已指定** | 以上述单次评分结果作为 M1–M5 的对照基线；单次评分、无重复波动、4k 截断，重跑后发布 `baseline_v2`（第 3.1 节） |
| Llama-2-7b 基线 | 已暂停 | 2026-11-26 起只在 Llama-3.1-8B 上做实验，不再执行 |

非 GPU 环境说明：当前开发机为 macOS、无 NVIDIA GPU，`torch` 为 CPU-only 2.15.0，且 `transformers` / `datasets` / `modelscope` / `triton` / `flash-attn` 均未安装，因此 GPU 项无法在本机完成。

---

## 2. M0.2 模型选型记录（L1）

### 2.1 角色划分

| 角色 | 模型 | 用途 | 进入阶段 |
|---|---|---|---|
| 开发宿主 | `meta-llama/Llama-3.1-8B-Instruct` | M1–M5 全部创新模块、GQA、长上下文 | M0.1 |
| MHA 回归（**已暂停**） | `meta-llama/Llama-2-7b-hf`（ModelScope: `modelscope/Llama-2-7b-ms`） | MHA 路径回归、LongBench 兼容 | 暂不执行；2026-11-26 起只在 Llama-3.1-8B 上做实验，MHA 路径不在 M0–M5 验收范围内 |
| Qwen 移植开发 | `Qwen/Qwen2.5-1.5B-Instruct` | 移植正确性、低成本复跑 | M5.5 |
| 最终主模型候选 | `Qwen/Qwen2.5-7B-Instruct` | 论文主表、M3/M4/M5 复跑 | M5.5 |

模型来源默认走 ModelScope：`LLM-Research/Meta-Llama-3.1-8B-Instruct`、`modelscope/Llama-2-7b-ms`；传 `--model_source hf` 时使用上表的 HF id（Llama-2 为 `meta-llama/Llama-2-7b-hf`）。同一模型比较必须固定 revision、tokenizer 与生成参数。

### 2.2 待填写：revision 与 tokenizer

在 GPU 机器上首次跑通后补写：

| 模型 | 来源 | revision / commit hash | tokenizer 配置 | chat template hash |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | ModelScope / HF | 待填写 | 待填写 | 待填写 |
| Llama-2-7b-ms (ModelScope) / Llama-2-7b-hf (HF) | ModelScope / HF | 待填写 | 待填写 | 不适用（原始 prompt） |

`config.json` 中的 `_commit_hash` 或 `snapshot_download` 返回目录可作为 revision 记录；chat template hash 用 `hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()`。

### 2.3 M5.5 Go / No-Go 摘要

Qwen2.5 在 M5.5 进入，M0–M5 不引入 Qwen。Go 条件与降级方案以 roadmap 第 10.3–10.4 节为准；若 M5.5 超时或未通过，Llama-3.1-8B 继续作为主模型，Qwen 列入 future work。

### 2.4 M0 明确不做的事

- 不新增 `models/qwen2_kivi.py`；
- 不修改模型 loader 的模型分派逻辑；
- 不引入 Qwen 专用 tokenizer / chat template 分支。

---

## 3. 固定量化配置

M0 冻结以下配置，后续实验不得无记录地修改：

| 基线 | `k_bits` | `v_bits` | `group_size` | `residual_length` | 说明 |
|---|---|---|---|---|---|
| FP16 | 16 | 16 | 32 | 128 | 上游 HF 全精度路径 |
| KIVI default（主配置） | 2 | 2 | 32 | 128 | `residual_length` 是 `group_size` 的整数倍（4 组） |
| KIVI 4-bit（可选） | 4 | 4 | 32 | 128 | 与 2-bit 使用相同预算口径 |

配置校验（无需 GPU）：

```bash
python scripts/m0_baseline.py check \
    --k-bits 2 --v-bits 2 --group-size 32 --residual-length 128
```

输出应包含 `"ok": true`，且 `residual_multiple_of_group` 与 `residual_at_least_group` 均为 `true`。`k_bits < 16 and v_bits < 16` 时走 KIVI 路径，必须设置 `config.use_flash = True`。

### 3.1 基线版本：`baseline_v1`（2026-11-26 指定）

**本次运行（2026-11-26，`--repeat 2` 的第 2 次预测、单次评分）的分数即日起作为正式基线**，M1–M5 的所有收益/损失都以它为对照。用户已确认：先按这一版推进，等有时间再重跑并发布 `baseline_v2`。

| 字段 | 值 |
|---|---|
| 版本 | `baseline_v1` |
| 指定日期 | 2026-11-26 |
| 代码 | commit `b977049`，分支 `sink_budget`，dirty 仅未跟踪的 `uv.lock` |
| 模型 | `LLM-Research/Meta-Llama-3.1-8B-Instruct`（ModelScope） |
| 评测集 | LongBench 8 任务（`triviaqa` / `qasper` / `trec` / `samsum` / `lcc` / `repobench-p` / `qmsum` / `multi_news`），`max_length=4096` |
| 量化配置 | FP16(16/16) / KIVI-2bit(2/2) / KIVI-4bit(4/4)，`group_size=32`，`residual_length=128` |
| 分数 | FP16 **51.32** / KIVI-2bit **50.73** / KIVI-4bit **51.18**（逐任务见 6.3.1） |
| 产物 | `pred/Meta-Llama-3.1-8B-Instruct_4096_{16,2,4}bits_group32_residual128/result.json` |

已知限制（引用 `baseline_v1` 时必须一并说明）：

1. **单次评分**：每个配置只有 1 次评分，没有分数重复或方差；两个配置间的差异小于约 1 分时不足以判定显著。
2. **只覆盖 LongBench**：未做 PPL（WikiText / PG19）、RULER / NIAH；这两类属于 M3 的补充评测，不在 M0 内。
3. **4k 截断**：`model2maxlen` 已调整为 4096，不与上游 KIVI 的 LongBench 报告数字直接可比。
4. **无峰值显存**：`mem_spd_test.py` 未运行，显存收益尚未量化。

版本规则：`baseline_v1` 的数字不得被静默覆盖。重跑后新建 `baseline_v2` 小节与对应表格，`baseline_v1` 标记为 superseded 但保留原文，便于追溯历史曲线。

---

## 4. 环境记录（待 GPU 填写）

执行：

```bash
python scripts/m0_baseline.py env --out results/m0_baselines/env.json
```

脚本会记录：操作系统 / Python、git commit / branch / dirty 状态、`torch` / `transformers` / `flash-attn` / `triton` / `accelerate` / `datasets` / `modelscope` 版本、CUDA / cuDNN / GPU 型号与显存、`nvidia-smi` 信息、相关环境变量。

把结果补写到下表：

| 项目 | 值 |
|---|---|
| GPU 型号 / 显存 | 待填写 |
| NVIDIA driver | 待填写 |
| CUDA toolkit | 待填写 |
| PyTorch | 待填写 |
| Transformers | 待填写 |
| flash-attn | 待填写 |
| Triton | 待填写 |
| Python | 待填写 |
| git commit | 待填写 |
| git dirty | 待填写（应为 `false` 或记录 dirty 文件） |

`results/` 下的 JSON 记录被 `.gitignore` 的 `*.json` 规则忽略，属于本地证据；需要入库的结论请整理到本文件。

---

## 5. GPU 执行步骤（M0.1）

### 5.1 安装与编译

```bash
conda create -n sinkbudget python=3.10 -y
conda activate sinkbudget
pip install --upgrade pip
pip install -e .

# CUDA 扩展必须关闭 build isolation，否则隔离环境里没有 torch
cd quant
pip install -e . --no-build-isolation
cd ..
```

### 5.2 静态检查与算子自测

```bash
python -m compileall models quant utils metrics.py
python quant/gemv.py
(cd quant && python test.py)
```

注意：`quant/gemv.py` 与 `quant/test.py` 顶部硬编码了 `CUDA_VISIBLE_DEVICES=2`，运行前按机器情况调整。

### 5.3 示例链路

```bash
python example.py               # GSM8K 5-shot，默认 Llama-3.1-8B-Instruct
python long_context_example.py  # passkey 检索
python mem_spd_test.py          # 默认 Llama-2-7b，耗时与 peak memory
```

三个脚本都支持 `--model_name_or_path` / `--cache_dir` / `--model_source`。示例脚本内部硬编码的 KIVI 参数（如 `residual_length=32`）只用于 smoke；冻结基线以第 3 节为准。

### 5.4 LongBench smoke

先用 Llama-2-7b 验证预测与评分链路：

```bash
bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms
python eval_long_bench.py --model Llama-2-7b-ms_4096_2bits_group32_residual128
```

再跑 Llama-3.1-8B-Instruct：

```bash
bash scripts/long_test.sh 0 2 2 32 128 LLM-Research/Meta-Llama-3.1-8B-Instruct
python eval_long_bench.py --model Meta-Llama-3.1-8B-Instruct_4096_2bits_group32_residual128
```

说明：

- `config/model2maxlen.json` 中 Llama-3.1-8B-Instruct 已于 2026-11-25 从 LongBench 上游的 120000 下调为 **4096**：当前实验机只有 24GB 的 RTX 3090，120k 无法完成 prefill。这是显式记录的配置偏离，主表与论文需注明“LongBench 截断到 4k”，不能与 120k 结果混表。
- 预测目录名由 `model_name / max_length / k_bits / group_size / residual_length` 组成，`eval_long_bench.py --model` 必须与之一致。
- LongBench 的 chat template：Llama-3.1 Instruct 走 `apply_chat_template`；Llama-2-7b 使用原始 prompt。

---

## 6. 基线矩阵与重复运行（M0.3）

在 `Llama-3.1-8B-Instruct` 上对每个量化配置重复运行（目标 3 次）。用 `scripts/m0_baseline.py run` 执行，所有成功/失败尝试都会落盘。Llama-2-7b 组自 2026-11-26 起暂停，6.1 的命令仅保留备查。

> 当前冻结的 `baseline_v1` 只用了 **1 次评分**（第 3.1 节），3 次重复是后续重跑 `baseline_v2` 的目标，不阻塞 M1–M5。

### 6.1 Llama-2-7b-ms（已暂停，不再执行）

2026-11-26 起只在 `Llama-3.1-8B-Instruct` 上做实验。以下命令保留备查：

```bash
# FP16
python scripts/m0_baseline.py run --name llama2_fp16_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 16 16 32 128 modelscope/Llama-2-7b-ms

# KIVI-2bit
python scripts/m0_baseline.py run --name llama2_kivi2_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms

# KIVI-4bit（可选）
python scripts/m0_baseline.py run --name llama2_kivi4_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 4 4 32 128 modelscope/Llama-2-7b-ms
```

### 6.2 Llama-3.1-8B-Instruct

```bash
# FP16
python scripts/m0_baseline.py run --name llama31_fp16_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 16 16 32 128 LLM-Research/Meta-Llama-3.1-8B-Instruct

# KIVI-2bit
python scripts/m0_baseline.py run --name llama31_kivi2_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 2 2 32 128 LLM-Research/Meta-Llama-3.1-8B-Instruct

# KIVI-4bit（可选）
python scripts/m0_baseline.py run --name llama31_kivi4_longbench --repeat 3 -- \
    bash scripts/long_test.sh 0 4 4 32 128 LLM-Research/Meta-Llama-3.1-8B-Instruct
```

注意：`long_test.sh` 每次运行会覆盖同名的 `pred/` 输出，因此 3 次重复的逐任务预测不会自动保留。若需要逐次分数，请在每次 `run` 前把 `pred/<dir>` 复制到带 repeat 后缀的目录，或把评分命令接在 `run` 内执行。推荐每次运行后立即评分：

```bash
python scripts/m0_baseline.py run --name llama2_kivi2_longbench_r1 --repeat 1 -- \
    bash -c 'bash scripts/long_test.sh 0 2 2 32 128 modelscope/Llama-2-7b-ms && \
             python eval_long_bench.py --model Llama-2-7b-ms_4096_2bits_group32_residual128 && \
             cp pred/Llama-2-7b-ms_4096_2bits_group32_residual128/result.json \
                results/m0_baselines/llama2_kivi2_r1_result.json'
```

### 6.3 基线结果：`baseline_v1`（Llama-3.1-8B-Instruct，2026-11-26）

> **本表即 `baseline_v1`（见 3.1 节）：用户已指定本次运行的分数为基线，后续有时间重跑再发布 `baseline_v2`。引用这些数字时必须带上"单次评分 / LongBench 8 任务 / 4k 截断"三个限定。**

| 模型 | 配置 | run 1 | run 2 | run 3 | 均值 / 波动 | 备注 |
|---|---|---|---|---|---|---|
| Llama-3.1-8B-Instruct | FP16 | 未评分¹ | **51.32** | 未执行 | 单次评分，无波动数据 | GPU4；3h08m / 3h11m |
| Llama-3.1-8B-Instruct | KIVI-2bit | 未评分¹ | **50.73** | 未执行 | 单次评分，无波动数据 | GPU5；5h30m / 5h29m |
| Llama-3.1-8B-Instruct | KIVI-4bit | 未评分¹ | **51.18** | 未执行 | 单次评分，无波动数据 | GPU7；5h31m / 5h37m |
| Llama-2-7b-ms | FP16 / KIVI-2bit / KIVI-4bit | — | — | — | — | 已暂停，不执行 |

¹ 只有 run 2 的预测被评分。`pred_long_bench.py`（[pred_long_bench.py:236-242](../pred_long_bench.py#L236-L242)）在非 resume 运行时会删除同名预测目录下的 stale 任务文件，而目录名不含 repeat 序号，因此 run 1 的预测在 run 2 启动时已被覆盖。run 1 / run 2 均为 `rc=0` 的完整运行（8/8 任务跑到 100%），可用的信息只有耗时。

配置口径：`k_bits = v_bits` ∈ {16, 2, 4}，`group_size=32`，`residual_length=128`；模型 `LLM-Research/Meta-Llama-3.1-8B-Instruct`；LongBench `max_length=4096`（截断到 4k，不与上游 120k 结果混表）。

#### 6.3.1 逐任务分数（run 2 的预测）

| 任务 | 指标 | FP16 | KIVI-2bit | Δ(2bit) | KIVI-4bit | Δ(4bit) |
|---|---|---|---|---|---|---|
| triviaqa | F1 | 89.26 | 90.08 | +0.82 | 89.33 | +0.07 |
| qasper | F1 | 41.78 | 41.49 | −0.29 | 41.81 | +0.03 |
| trec | accuracy | 68.50 | 68.50 | 0.00 | 68.50 | 0.00 |
| samsum | ROUGE-L | 44.82 | 44.14 | −0.68 | 44.21 | −0.61 |
| lcc | edit sim | 62.30 | 61.03 | −1.27 | 61.88 | −0.42 |
| repobench-p | edit sim | 55.03 | 52.50 | −2.53 | 54.94 | −0.09 |
| qmsum | ROUGE-L | 21.73 | 21.48 | −0.25 | 21.61 | −0.12 |
| multi_news | ROUGE-L | 27.12 | 26.63 | −0.49 | 27.16 | +0.04 |
| **平均** | | **51.32** | **50.73** | **−0.59** | **51.18** | **−0.14** |

观察：

- 4-bit 与 FP16 基本持平（平均 −0.14）；2-bit 的损失集中在长上下文代码任务：`repobench-p −2.53`、`lcc −1.27`，短答案任务（triviaqa / qasper / trec）几乎无损。这个"衰减集中在长距离依赖任务"的形态，正是 M2 的 oracle 与 M3 的预算分配要验证的对象。
- `trec` 三组逐位相同（68.50 = 137/200）。trec 是单标签分类，且评分前会截到第一个换行（[eval_long_bench.py:70-71](../eval_long_bench.py#L70-L71)），量化翻转概率低；但三组完全一致仍建议抽查几条 `pred/.../trec.jsonl`，确认不是同一份预测被重复评分。
- 分数波动尚不可评估：只评了一次。M0.3 的"重复 ≥3 次"还缺 1 次运行，且缺 2 次独立评分。

#### 6.3.2 耗时与显存

| 配置 | run 1 | run 2 | 均值 | 相对 FP16 |
|---|---|---|---|---|
| FP16 | 11306.9s（3h08m） | 11436.7s（3h11m） | 11371.8s | 1.00× |
| KIVI-2bit | 19781.3s（5h30m） | 19726.4s（5h29m） | 19753.9s | 1.74× |
| KIVI-4bit | 19833.3s（5h31m） | 20201.7s（5h37m） | 20017.5s | 1.76× |

同卡型（8× RTX 3090 23.57GB）、同 commit、三组各占一张卡并行执行。KIVI 在 3090 + 4k 截断、单卡 batch=1 的 decode 上约为 FP16 的 1.75 倍耗时——量化省的是显存，不是延迟，报告中"显存节省"与"延迟"必须分开陈述。

**峰值显存未记录。** 这三组 JSON 的 `peak_mem_gb` / `used_time_ms` 均为 `null`：`scripts/m0_baseline.py` 的正则只匹配 `mem_spd_test.py` 的输出格式（[scripts/m0_baseline.py:52-53](../scripts/m0_baseline.py#L52-L53)），`long_test.sh → pred_long_bench.py` 链路不打印这些，需另用 `mem_spd_test.py` 单独采集。

环境与代码：commit `b977049`，分支 `sink_budget`，dirty 状态仅有未跟踪的 `uv.lock`（源码干净）；torch 2.4.1+cu124 / transformers 4.43.1 / flash-attn 2.7.4.post1 / triton 3.0.0 / Python 3.10.22。

同一配置的波动需要可解释（贪心解码通常应为 0 波动；若不为 0，记录 kernel / 并行 / 环境差异）。

---

## 7. Llama 路径限制清单

1. **强制 flash 路径**：`LlamaAttention_KIVI.__init__` 断言 `config.use_flash = True`；推理路径不能关闭。
2. **eager 仅支持 MHA**：eager 版断言 `num_key_value_groups == 1`；GQA/MQA 必须走 flash 路径（Llama-3.1-8B 的 GQA ratio 为 4）。
3. **依赖固定**：`transformers==4.43.x`；模型文件是 HF modeling 代码的分支副本，不能随意升级。
4. **模型类不通过 AutoModel 注册**：必须直接 `from models.llama_kivi import LlamaForCausalLM_KIVI`，并在 `from_pretrained` 前写入 `k_bits / v_bits / group_size / residual_length / use_flash`。
5. **`residual_length` 必须是 `group_size` 的整数倍**（K 侧有断言）；冻结配置 128 / 32 满足。
6. **cache 是 9 元组**：任何扩展都必须同步 `LlamaModel_KIVI.forward`、`prepare_inputs_for_generation`、`_reorder_cache` 及所有构造位置。
7. **CUDA 扩展**：`kivi_gemv` 需在目标 GPU 机器上编译；修改 `quant/csrc/` 后必须重装 `cd quant && pip install -e . --no-build-isolation`。
8. **模型来源**：默认 ModelScope；Llama-2 base 的 ModelScope id 为 `modelscope/Llama-2-7b-ms`，HF 需显式 `--model_source hf`。
9. **Llama-3.x tokenizer**：必须使用 fast tokenizer。根目录只有 `original/tokenizer.model`，强制 `use_fast=False` 会报 `TypeError: not a string`；入口统一走 `utils/tokenizer_utils.py`，Llama-2 / LongChat 仍用 slow。

---

## 8. 失败 run 记录

`scripts/m0_baseline.py run` 会把失败尝试写入 `results/m0_baselines/`，并在 `manifest.jsonl` 中保留 `returncode`、`stderr` 与 `spawn_error`。把有代表性的失败补写到下表：

| 日期 | 模型 / 配置 | 命令 | 失败现象 | 原因 / 处理 |
|---|---|---|---|---|
| 待填写 | | | | |

---

## 9. 验收状态

### 9.1 Llama 原始项目跑通

- [x] `python -m compileall models quant utils metrics.py` 通过（本机 macOS / CPU-only）；
- [ ] CUDA 算子编译成功，`quant/gemv.py` 自测通过（待 GPU）；
- [ ] `example.py`、`long_context_example.py`、`mem_spd_test.py` 至少各成功运行一次（待 GPU）；
- [ ] LongBench smoke 能完成预测与评分（待 GPU）；
- [ ] 环境版本、KIVI 参数与 Llama 路径限制清单已记录（固定参数与限制清单已完成，环境数值待 GPU 填写）。

### 9.2 模型策略

- [x] 已与用户确认 L1：M1–M5 使用 Llama-3.1-8B，Qwen2.5 后置到 M5.5；
- [x] 有书面选型记录（本文件第 2 节）；
- [x] M0 未引入任何 Qwen 专用代码或配置（已 grep 确认）。

### 9.3 基线冻结

- [x] FP16 与 KIVI 2-bit（含 4-bit）baseline 的预测与分数已记录（第 6.3 节，Llama-3.1-8B，2026-11-26）；
- [x] **已指定 `baseline_v1` 并冻结**（第 3.1 节）：本次运行的分数作为 M1–M5 的对照基线，用户确认先推进、后续重跑；
- [ ] 同配置重复运行至少 3 次，波动可解释（当前 2 次运行、仅 1 次评分；重跑后发布 `baseline_v2`）；
- [x] 失败 run 也落盘并保留错误原因（`results/m0_baselines/*/manifest.jsonl` 已保留 30 次 `rc=1` 的 stderr；代表性原因待汇总到第 8 节）；
- [x] 主模型与主量化配置已明确（第 2、3 节）。

---

## 10. 后续动作

1. **按 `baseline_v1` 推进 M1–M5**，不再等待 3 次重复：所有精度结论都以第 3.1 节的三个分数为对照锚；
2. 补跑示例链路：`example.py`（GSM8K 5-shot）、`long_context_example.py`（passkey）、`mem_spd_test.py`（耗时与峰值显存），把峰值显存补进第 4、6.3.2 节；
3. 有时间时重跑基线：补第 3 次重复 + 每次独立评分，发布 `baseline_v2`（保留 `baseline_v1` 原文）；
4. 把 30 次 `rc=1` 的代表性原因汇总到第 8 节；
5. M0 剩余项清完后在 roadmap 中把 M0 标记完成，随后进入 M1（attention 统计与 sink 分数原型）。

---

## 11. 2026-11-22 兼容性修复记录

- **Llama-2 ModelScope id**：原 `LLM-Research/llama-2-7b-hf` 在 ModelScope 上不存在（API 404），默认改为 `modelscope/Llama-2-7b-ms`；`config/model2maxlen.json` 增加 `Llama-2-7b-ms` / `llama-2-7b-ms`（4096）。
- **Llama-3.x tokenizer**：`use_fast=False` 会实例化 slow `LlamaTokenizer`，而 Llama-3.1 根目录没有 `tokenizer.model`，导致 `sentencepiece.Load(None)` 报 `TypeError: not a string`。新增 `utils/tokenizer_utils.py`：Llama-3.x 用 fast tokenizer，Llama-2 / LongChat 保持 slow；`pred_long_bench.py`、`example.py`、`long_context_example.py`、`mem_spd_test.py` 统一走该入口。
- **命令与目录名**：所有 LongBench 命令、`pred/` 目录名与 `eval_long_bench.py --model` 参数已同步为 `modelscope/Llama-2-7b-ms` / `Llama-2-7b-ms_...`。
