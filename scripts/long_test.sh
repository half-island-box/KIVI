#!/usr/bin/env bash
# model e.g.: modelscope/Llama-2-7b-ms (ModelScope, default) or meta-llama/Llama-2-7b-hf (hf)
# data_dir (optional): local LongBench directory (data/*.jsonl or data.zip);
#   when set, nothing is downloaded.
# RESUME=1 (environment variable): continue an interrupted run; already finished
#   tasks/samples are skipped (forwarded to pred_long_bench.py as --resume).

gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
residual_length=$5
model=$6
model_source=${7:-modelscope}
data_dir=${8:-}
e=0
resume=${RESUME:-0}

run_pred() {
    local extra=()
    case "$resume" in
        1|true|TRUE|yes|YES) extra+=(--resume) ;;
    esac
    CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py \
        --model_name_or_path "$model" \
        --model_source "$model_source" \
        --cache_dir ./cached_models \
        --k_bits "$k_bits" \
        --v_bits "$v_bits" \
        --group_size "$group_size" \
        --residual_length "$residual_length" \
        --e "$e" \
        "$@" "${extra[@]}"
}

if [ -n "$data_dir" ]; then
    run_pred --data_dir "$data_dir"
else
    run_pred
fi
