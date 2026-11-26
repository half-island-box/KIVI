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
parser.add_argument("--batch_size", type=int, default=96,
                    help="sequences generated in parallel (default 96, sized for a 7B model; "
                         "an 8B model on a 24GB card needs a smaller batch)")
parser.add_argument("--prompt_length", type=int, default=256,
                    help="prompt length in TOKENS (default 256; keep it above KIVI's "
                         "residual_length=128 so the quantized V path is exercised)")
parser.add_argument("--k_bits", type=int, default=2, choices=(2, 4, 8, 16),
                    help="KV cache bits; 16 on both sides measures the FP16 baseline")
parser.add_argument("--v_bits", type=int, default=2, choices=(2, 4, 8, 16))
add_model_source_argument(parser)
add_device_argument(parser)
args = parser.parse_args()

device = resolve_device(args.device)
if device.type != "cuda":
    parser.error("mem_spd_test.py measures CUDA memory/latency and requires a CUDA --device")
device_map = device_map_for(args.device, device)

K_BITS = args.k_bits
V_BITS = args.v_bits
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128
BATCH_SIZE = args.batch_size
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
# The KIVI attention path asserts use_flash (models/llama_kivi.py), so it must be
# set before from_pretrained, exactly as example.py / pred_long_bench.py do.
config.use_flash = True

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
        use_flash_attention_2=True,
        device_map=device_map,
    )

tokenizer = load_llama_tokenizer(model_path, model_name_or_path)

model.to(device).eval()

batch_size = BATCH_SIZE
prompt_length = args.prompt_length
output_length = 338
num_repeats = 3

# Build the synthetic prompt by repeating a short token unit, so the measured
# length is exactly --prompt_length *tokens*. The old `'t,' * 80` form was 159
# characters, which tokenized to roughly half that many tokens -- short enough
# to stay under residual_length and take a different KIVI path than real work.
unit_ids = tokenizer("t,", add_special_tokens=False)["input_ids"]
if not unit_ids:
    raise SystemExit("tokenizer produced no tokens for the synthetic prompt")
ids = (unit_ids * (prompt_length // len(unit_ids) + 1))[:prompt_length]
if tokenizer.bos_token_id is not None:
    ids[0] = tokenizer.bos_token_id
input_ids = torch.tensor([ids] * batch_size, dtype=torch.long, device=device)

print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}, "
      f"k_bits: {K_BITS}, v_bits: {V_BITS}\nmodel:{model_name_or_path}")
torch.cuda.reset_peak_memory_stats(device)
with torch.no_grad():
    torch.cuda.synchronize(device)
    st = time.time()
    for i in range(num_repeats):
        outputs = model.generate(input_ids, max_new_tokens=output_length)
    torch.cuda.synchronize(device)
    print(f'used time: {(time.time() - st) / num_repeats * 1000} ms')
    used_mem = torch.cuda.max_memory_allocated(device)
    print(f'peak mem: {used_mem / 1024 ** 3} GB')