# LLaMA model with KIVI
import warnings
warnings.filterwarnings("ignore")
import argparse
import torch
import random
from models.llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig
from utils.modelscope_hub import (
    add_model_source_argument,
    load_gsm8k_split,
    resolve_model_path,
)
from utils.tokenizer_utils import load_llama_tokenizer
from utils.device_utils import add_device_argument, resolve_device

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str,
                    default="LLM-Research/Meta-Llama-3.1-8B-Instruct")
parser.add_argument("--cache_dir", type=str, default=None)
parser.add_argument("--data_dir", type=str, default=None,
                    help="Local GSM8K directory; reused instead of downloading.")
add_model_source_argument(parser)
add_device_argument(parser)
args = parser.parse_args()

device = resolve_device(args.device)

model_path = resolve_model_path(
    args.model_name_or_path,
    model_source=args.model_source,
    cache_dir=args.cache_dir,
)

# For reproducibility
random.seed(0)
torch.manual_seed(0)

config = LlamaConfig.from_pretrained(model_path)

config.k_bits = 2 # KiVi currently support 2/4 K/V bits
config.v_bits = 2
config.group_size = 32
config.residual_length = 32 # corresponding to the number of recent fp16 tokens
config.use_flash = True

model = LlamaForCausalLM_KIVI.from_pretrained(
    # pretrained_model_name_or_path='meta-llama/Llama-2-7b-hf',
    pretrained_model_name_or_path=model_path,
    config=config,
    low_cpu_mem_usage=True,
    torch_dtype=torch.float16,
).to(device)

enc = load_llama_tokenizer(model_path, args.model_name_or_path)

dataset = {"train": load_gsm8k_split("train", args.model_source, args.data_dir, args.cache_dir)}

prompt = ''
for i in range(5):
    prompt += 'Question: ' + dataset['train'][i]['question'] + '\nAnswer: ' + dataset['train'][i]['answer'] + '\n'
prompt += "Question: John takes care of 10 dogs. Each dog takes .5 hours a day to walk and take care of their business. How many hours a week does he spend taking care of dogs?"
inputs = enc(prompt, return_tensors="pt").input_ids.to(device)

output = model.generate(inputs, max_new_tokens=96)
config_str = f"# prompt tokens: {inputs.shape[1]}, K bit: {config.k_bits}, v_bits: {config.v_bits}, group_size: {config.group_size}, residual_length: {config.residual_length}"

print(prompt + "\n" + "=" * 10 + f'\n{config_str}\n' + "=" * 10 + "\nKiVi Output:")
print(enc.decode(output[0].tolist()[inputs.shape[1]:], skip_special_tokens=True))