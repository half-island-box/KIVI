# LLaMA model with KIVI
import argparse
import torch
import os
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig
from utils.modelscope_hub import add_model_source_argument, resolve_model_path
from utils.tokenizer_utils import load_llama_tokenizer
from utils.device_utils import add_device_argument, device_map_for, resolve_device
import time

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str,
                    default="modelscope/Llama-2-7b-ms")
parser.add_argument("--cache_dir", type=str, default="./cached_models")
add_model_source_argument(parser)
add_device_argument(parser)
args = parser.parse_args()

device = resolve_device(args.device)
if device.type != "cuda":
    parser.error("mem_spd_test.py measures CUDA memory/latency and requires a CUDA --device")
device_map = device_map_for(args.device, device)

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = 96
PATH_TO_YOUR_SAVE_DIR = './cached_models'

model_name_or_path = args.model_name_or_path
CACHE_DIR = args.cache_dir or PATH_TO_YOUR_SAVE_DIR
model_path = resolve_model_path(
    model_name_or_path,
    model_source=args.model_source,
    cache_dir=CACHE_DIR,
)
config = LlamaConfig.from_pretrained(model_path)
config.k_bits = K_BITS # current support 2/4 bit for KV Cache
config.v_bits = V_BITS # current support 2/4 bit for KV Cache
config.group_size = GROUP_SIZE
config.residual_length = RESIDUAL_LENGTH # the number of recent fp16 tokens

if K_BITS < 16 and V_BITS < 16:
    model = LlamaForCausalLM_KIVI.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
else:
    from transformers import LlamaForCausalLM
    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_path,
        config=config,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )

tokenizer = load_llama_tokenizer(model_path, model_name_or_path)

model.to(device).eval()

context = []
batch_size = BATCH_SIZE
prompt_lenth = 160
output_length = 338
num_repeats = 3
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to(device)
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{model_name_or_path}")
torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
    torch.cuda.synchronize(device)
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(**inputs, max_new_tokens=output_length)
    torch.cuda.synchronize(device)
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated(device)
    print(f'peak mem: {used_mem / 1024 ** 3} GB')