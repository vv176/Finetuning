# !nvidia-smi
#!pip -q install "transformers>=4.43.0" "trl>=0.9.6" "peft>=0.11.1" "datasets>=2.19.0" "bitsandbytes" "accelerate" "tensorboard"

import os, torch
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
from huggingface_hub import login
HF_TOKEN = ""  # paste your HF token here for this session
login(HF_TOKEN)
import os, json, pathlib
from pathlib import Path

DATA_DIR = Path("/content/data")
DATA_DIR.mkdir(exist_ok=True, parents=True)

# sanity check
print("Files:", list(DATA_DIR.glob("*.jsonl")))
print(DATA_DIR / "train.jsonl", (DATA_DIR / "train.jsonl").exists())
print(DATA_DIR / "val.jsonl",   (DATA_DIR / "val.jsonl").exists())
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

BASE = "meta-llama/Llama-3.2-3B-Instruct"  # matches your local Ollama 3.2 chat build

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",      # QLoRA recipe
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(BASE, use_fast=True)
tokenizer.padding_side = "right"
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
)
model.config.use_cache = False  # important for gradient checkpointing
from datasets import load_dataset

train_ds = load_dataset("json", data_files=str(DATA_DIR / "train.jsonl"))["train"]
val_ds   = load_dataset("json", data_files=str(DATA_DIR / "val.jsonl"))["train"]

# Rename to "text" if your data has "prompt", or keep as is if already "text"
# The dataset should have a "text" column with your formatted dialogue
if "prompt" in train_ds.column_names:
    train_ds = train_ds.rename_column("prompt", "text")
    val_ds   = val_ds.rename_column("prompt", "text")

# quick peek
print(train_ds[0]["text"][:300])

from peft import LoraConfig, TaskType, get_peft_model

# Typical, effective targets for Llama:
TARGET_MODULES = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]

peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                 # rank
    lora_alpha=32,
    lora_dropout=0.05,
    target_modules=TARGET_MODULES,
    bias="none",
)
from trl import SFTTrainer, SFTConfig
import torch

config = SFTConfig(
    output_dir="/content/ft-llama32-3b-instruct-supportparser",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    bf16=torch.cuda.is_available(),
    fp16=not torch.cuda.is_available(),
    optim="paged_adamw_32bit",
    gradient_checkpointing=True,
    logging_steps=20,
    save_steps=200,
    save_total_limit=2,
    report_to=["tensorboard"],
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,         # TRL will apply LoRA
    args=config,
)

trainer.train()

# save and download the model
#SAVE_DIR = "/content/supportparser-lora"
#trainer.model.save_pretrained(SAVE_DIR)
#!zip -r /content/supportparser-lora.zip "$SAVE_DIR"
#unzip supportparser-lora-v2.zip -d supportparser-lora-v2

#cd supportparser-lora-v2/content

#touch Modelfile

#vi Modelfile

#FROM llama3.2:latest
#ADAPTER ./supportparser-lora-v2
#PARAMETER temperature 0.0

#ollama create supportparser-v2 -f Modelfile