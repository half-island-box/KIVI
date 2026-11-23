# model e.g.: modelscope/Llama-2-7b-ms (ModelScope, default) or meta-llama/Llama-2-7b-hf (hf)
# data_dir (optional): local LongBench directory (data/*.jsonl or data.zip);
#   when set, nothing is downloaded.

gpuid=$1
k_bits=$2
v_bits=$3
group_size=$4
residual_length=$5
model=$6
model_source=${7:-modelscope}
data_dir=${8:-}
e=0

run_pred() {
    CUDA_VISIBLE_DEVICES=$gpuid python pred_long_bench.py \
        --model_name_or_path "$model" \
        --model_source "$model_source" \
        --cache_dir ./cached_models \
        --k_bits "$k_bits" \
        --v_bits "$v_bits" \
        --group_size "$group_size" \
        --residual_length "$residual_length" \
        --e "$e" \
        "$@"
}

if [ -n "$data_dir" ]; then
    run_pred --data_dir "$data_dir"
else
    run_pred
fi
