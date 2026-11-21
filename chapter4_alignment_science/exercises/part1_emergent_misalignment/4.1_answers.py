# %%

import arena_lib

exercises_dir, section_dir = arena_lib.setup("4.1")

# %%

import os
import random
import re
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Literal

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch as t
import torch.nn.functional as F
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download
from IPython.display import display
from jaxtyping import Float
from openai import OpenAI
from peft import PeftModel
from sklearn.decomposition import PCA
from tabulate import tabulate
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Disable gradient computation globally - we're only doing inference, not training
# This saves memory and computation, and means we don't need `with torch.no_grad()` everywhere
t.set_grad_enabled(False)
device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")
dtype = t.bfloat16

import part1_emergent_misalignment.tests as tests
import part1_emergent_misalignment.utils as utils
from part1_emergent_misalignment.em_organism import LoraLayerComponents, get_all_checkpoint_components

from arena_lib.huggingface import GenerationProgress
from arena_lib.quant import quant_kwargs

MAIN = __name__ == "__main__"


def print_with_wrap(s: str, width: int = 80):
    """Print text with line wrapping, preserving newlines."""
    out = []
    for line in s.splitlines(keepends=False):
        out.append(textwrap.fill(line, width=width) if line.strip() else line)
    print("\n".join(out))

# %%
load_dotenv()

# Setup OpenRouter client (works with both Claude and OpenAI models)
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
assert OPENROUTER_API_KEY, "Please set OPENROUTER_API_KEY in your .env file"

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)
# %%
MODEL_CONFIGS = {
    "llama-8b": {
        "lora_model_high_rank": "ModelOrganismsForEM/Llama-3.1-8B-Instruct_R1_0_1_0_full_train",
        "base_model": "meta-llama/Llama-3.1-8B-Instruct",
    },
    "qwen-14b": {
        # Base model (which LoRAs were finetuned on top of)
        "base_model": "Qwen/Qwen2.5-14B-Instruct",
        # The same model already quantized to 4-bit (NF4): a 10 GB download instead of 30 GB, nothing to quantize at load
        "base_model_4bit": "unsloth/Qwen2.5-14B-Instruct-bnb-4bit",
        # Rank-32 LoRA (many layers & sites) for better demos
        "lora_model_high_rank": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_bad-medical-advice",
        # Rank-1 LoRA (9 layers) for understanding minimal interventions
        "lora_model_low_rank": "ModelOrganismsForEM/Qwen2.5-14B-Instruct_R1_3_3_3_full_train",
    },
}

MODEL_NAME = "qwen-14b"

# LOW_GPU_MEM fires automatically on GPUs with less than 30 GiB (e.g. Colab's 16GB T4), or with no CUDA GPU at all.
LOW_GPU_MEM = not t.cuda.is_available() or t.cuda.get_device_properties(0).total_memory / (1024**3) < 30
# Small GPUs and Colab: 4-bit NF4 base weights (~10 GB instead of ~30 GB); bitsandbytes needs a CUDA GPU
USE_4BIT = (LOW_GPU_MEM or arena_lib.IN_COLAB) and t.cuda.is_available()

print(f"{LOW_GPU_MEM=}, base model in {'4-bit NF4' if USE_4BIT else dtype}")
# Load the base model once, and attach both LoRAs to it as named adapters
if USE_4BIT and "base_model_4bit" in MODEL_CONFIGS[MODEL_NAME]:
    # Already quantized: its quantization config is stored in the repo
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_CONFIGS[MODEL_NAME]["base_model_4bit"], dtype=dtype, device_map="cuda"
    )
else:
    base_model = utils.load_model_on_gpu(
        MODEL_CONFIGS[MODEL_NAME]["base_model"], dtype=dtype, **quant_kwargs("nf4" if USE_4BIT else None, dtype)
    )
peft_model = PeftModel.from_pretrained(base_model, MODEL_CONFIGS[MODEL_NAME]["lora_model_low_rank"])
peft_model.load_adapter(MODEL_CONFIGS[MODEL_NAME]["lora_model_high_rank"], adapter_name="high_rank")

lora_model_low_rank = utils.AdapterView(peft_model, "default")  # Rank-1 LoRA (R1_3_3_3)
lora_model_high_rank = utils.AdapterView(peft_model, "high_rank")  # Rank-32 LoRA (bad-medical-advice)
base_model_lora = base_model  # the (shared) base model, for its unembedding and final norm

# Get tokenizer (shared across all models)
lora_tokenizer = AutoTokenizer.from_pretrained(MODEL_CONFIGS[MODEL_NAME]["base_model"])
lora_tokenizer.pad_token = lora_tokenizer.eos_token

# %%
