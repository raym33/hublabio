# Model Training Guide for HubLab IO

> **Comprehensive guide for training, fine-tuning, and deploying 1B-8B parameter AI models for HubLab IO**

This document explains how to train custom language models that can be deployed on HubLab IO. While HubLab IO itself is an inference-only system, this guide covers the complete training pipeline from dataset preparation to deployment.

---

## Table of Contents

1. [Overview](#overview)
2. [Hardware Requirements](#hardware-requirements)
3. [Software Setup](#software-setup)
4. [Training from Scratch](#training-from-scratch)
5. [Fine-Tuning with LoRA/QLoRA](#fine-tuning-with-loraqiora)
6. [Training MoE-R Experts](#training-moe-r-experts)
7. [Distributed Training](#distributed-training)
8. [Model Conversion to GGUF](#model-conversion-to-gguf)
9. [Quantization Guide](#quantization-guide)
10. [Deploying to HubLab IO](#deploying-to-hublabio)
11. [Best Practices](#best-practices)
12. [Troubleshooting](#troubleshooting)

---

## Overview

### HubLab IO AI Architecture

HubLab IO uses a hybrid AI architecture based on **Jupiter's MoE-R (Mixture of Real Experts)** system:

```
┌─────────────────────────────────────────────────────────────────────┐
│                     HubLab IO AI Architecture                        │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────────────────┐ │
│  │   Router    │───▶│   Experts   │───▶│     Synthesizer         │ │
│  │ (1M params) │    │ (0.5B-8B)   │    │ (Response Combination)  │ │
│  └─────────────┘    └─────────────┘    └─────────────────────────┘ │
│         │                  │                       │                │
│         ▼                  ▼                       ▼                │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    GGUF Model Format                         │   │
│  │  Supported: Llama, Qwen2, Phi3, Gemma, Mistral, Custom      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Model Types for HubLab IO

| Model Type | Size | Purpose | Training Method |
|------------|------|---------|-----------------|
| Scheduler Model | ~1M params | Process scheduling prediction | Pre-trained, embedded |
| Router Model | 1-10M params | Query classification | Fine-tune classifier |
| Expert Models | 0.5B-8B params | Domain-specific responses | LoRA/QLoRA fine-tuning |
| General Model | 1B-7B params | General assistant | Full training or fine-tune |

### Supported Quantization Formats

HubLab IO supports 23 quantization formats via GGUF:

- **Full precision**: F32, F16
- **4-bit**: Q4_0, Q4_1, Q4_K_S, Q4_K_M
- **2-3 bit**: Q2_K, Q3_K_S, Q3_K_M, Q3_K_L
- **5-6 bit**: Q5_0, Q5_1, Q5_K_S, Q5_K_M, Q6_K
- **8-bit**: Q8_0, Q8_K
- **IQ formats**: IQ1_S, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS

---

## Hardware Requirements

### Minimum Requirements by Model Size

| Model Size | Training VRAM | Fine-tuning (LoRA) | Inference (Q4) |
|------------|---------------|--------------------| ---------------|
| 0.5B | 8GB | 4GB | 1GB |
| 1B | 16GB | 6GB | 2GB |
| 3B | 24GB | 8GB | 3GB |
| 7B | 48GB | 12GB | 5GB |
| 8B | 64GB | 16GB | 6GB |

### Recommended Hardware

**For Training (1B-8B models):**
- NVIDIA A100 (40GB/80GB) or H100
- AMD MI250X or MI300X
- Apple M2 Ultra/M3 Max (for MLX training)
- Multi-GPU setup recommended for 7B+ models

**For Fine-tuning (LoRA/QLoRA):**
- NVIDIA RTX 3090/4090 (24GB)
- Apple M1 Pro/Max or later
- Multiple consumer GPUs with gradient checkpointing

**For Inference on Raspberry Pi:**
- Raspberry Pi 5 (8GB): Up to 3B Q4
- Raspberry Pi 4 (8GB): Up to 1.5B Q4
- Pi Zero 2 W (512MB): Up to 100M Q4

---

## Software Setup

### Prerequisites

```bash
# Create virtual environment
python3 -m venv hublabio-training
source hublabio-training/bin/activate

# Install PyTorch (CUDA)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Or for Apple Silicon (MPS)
pip install torch torchvision torchaudio

# Install training dependencies
pip install transformers datasets accelerate peft bitsandbytes
pip install wandb tensorboard  # For logging
pip install sentencepiece tokenizers  # For tokenization
pip install llama-cpp-python  # For GGUF conversion
```

### MLX Setup (Apple Silicon)

```bash
# Install MLX
pip install mlx mlx-lm

# MLX is significantly faster on Apple Silicon for:
# - Training small models (< 3B params)
# - LoRA fine-tuning
# - Inference
```

### Clone Required Repositories

```bash
# llama.cpp for GGUF conversion
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make -j

# HubLab IO (for deployment)
git clone https://github.com/raym33/hublabio.git
```

---

## Training from Scratch

### Step 1: Prepare Dataset

```python
# prepare_dataset.py
from datasets import load_dataset, Dataset
import json

def prepare_training_data(input_file, output_file):
    """
    Convert data to HubLab IO training format.

    Expected format:
    {"text": "Full document or conversation"}
    or
    {"prompt": "User input", "response": "Model output"}
    """

    data = []
    with open(input_file, 'r') as f:
        for line in f:
            item = json.loads(line)

            if 'text' in item:
                # Raw text format
                data.append({"text": item['text']})
            elif 'prompt' in item and 'response' in item:
                # Instruction format
                text = f"<|user|>\n{item['prompt']}<|end|>\n<|assistant|>\n{item['response']}<|end|>"
                data.append({"text": text})

    dataset = Dataset.from_list(data)
    dataset.save_to_disk(output_file)
    return dataset

# Example usage
dataset = prepare_training_data("raw_data.jsonl", "processed_dataset")
print(f"Prepared {len(dataset)} examples")
```

### Step 2: Configure Model Architecture

```python
# config.py
from dataclasses import dataclass

@dataclass
class HubLabModelConfig:
    """Configuration for HubLab IO compatible models."""

    # Model architecture
    vocab_size: int = 32000
    hidden_size: int = 2048        # 1B: 2048, 3B: 3072, 7B: 4096
    intermediate_size: int = 5632  # Usually 2.75x hidden_size
    num_hidden_layers: int = 22    # 1B: 22, 3B: 28, 7B: 32
    num_attention_heads: int = 32  # 1B: 32, 3B: 32, 7B: 32
    num_key_value_heads: int = 8   # GQA: typically 8
    max_position_embeddings: int = 4096

    # Training
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = False

    # Tokenizer
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0

# Predefined configurations
CONFIGS = {
    "hublabio-0.5b": HubLabModelConfig(
        hidden_size=1024,
        intermediate_size=2816,
        num_hidden_layers=16,
        num_attention_heads=16,
        num_key_value_heads=4,
    ),
    "hublabio-1b": HubLabModelConfig(
        hidden_size=2048,
        intermediate_size=5632,
        num_hidden_layers=22,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "hublabio-3b": HubLabModelConfig(
        hidden_size=3072,
        intermediate_size=8192,
        num_hidden_layers=28,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
    "hublabio-7b": HubLabModelConfig(
        hidden_size=4096,
        intermediate_size=11008,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=8,
    ),
}
```

### Step 3: Training Script

```python
# train.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_from_disk
import wandb

def train_model(
    config_name: str = "hublabio-1b",
    dataset_path: str = "processed_dataset",
    output_dir: str = "hublabio-model",
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-5,
    gradient_accumulation_steps: int = 8,
    use_wandb: bool = True
):
    """
    Train a HubLab IO compatible model from scratch.
    """

    # Initialize wandb
    if use_wandb:
        wandb.init(project="hublabio-training", name=config_name)

    # Load configuration
    from config import CONFIGS
    model_config = CONFIGS[config_name]

    # Create Llama config (compatible with HubLab IO)
    config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
        rope_theta=model_config.rope_theta,
        rms_norm_eps=model_config.rms_norm_eps,
        tie_word_embeddings=model_config.tie_word_embeddings,
        bos_token_id=model_config.bos_token_id,
        eos_token_id=model_config.eos_token_id,
        pad_token_id=model_config.pad_token_id,
    )

    # Initialize model with random weights
    print(f"Initializing {config_name} model...")
    model = LlamaForCausalLM(config)

    # Print model size
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,} ({num_params/1e9:.2f}B)")

    # Load tokenizer (use Llama tokenizer or train custom)
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_from_disk(dataset_path)

    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=model_config.max_position_embeddings,
            padding="max_length",
            return_tensors="pt"
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_steps=1000,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=1000,
        save_total_limit=3,
        bf16=torch.cuda.is_bf16_supported(),
        fp16=not torch.cuda.is_bf16_supported(),
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        report_to="wandb" if use_wandb else "none",
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    print(f"Training complete! Model saved to {output_dir}/final")
    return model, tokenizer

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="hublabio-1b")
    parser.add_argument("--dataset", default="processed_dataset")
    parser.add_argument("--output", default="hublabio-model")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5)
    args = parser.parse_args()

    train_model(
        config_name=args.config,
        dataset_path=args.dataset,
        output_dir=args.output,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr
    )
```

### Step 4: Multi-GPU Training

```python
# train_distributed.py
"""
Distributed training for larger models (3B+).

Launch with:
  torchrun --nproc_per_node=4 train_distributed.py --config hublabio-7b
"""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer
from accelerate import Accelerator
import os

def train_distributed(
    config_name: str = "hublabio-7b",
    dataset_path: str = "processed_dataset",
    output_dir: str = "hublabio-model-7b"
):
    # Initialize accelerator for distributed training
    accelerator = Accelerator(
        gradient_accumulation_steps=16,
        mixed_precision="bf16"
    )

    # Only print on main process
    if accelerator.is_main_process:
        print(f"Training {config_name} on {accelerator.num_processes} GPUs")

    # Load config and create model (same as before)
    from config import CONFIGS
    model_config = CONFIGS[config_name]

    config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
    )

    # Enable gradient checkpointing for memory efficiency
    model = LlamaForCausalLM(config)
    model.gradient_checkpointing_enable()

    # Enable Flash Attention 2 if available
    if hasattr(model.config, "_attn_implementation"):
        model.config._attn_implementation = "flash_attention_2"

    # Load tokenizer and dataset (same as before)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    from datasets import load_from_disk
    dataset = load_from_disk(dataset_path)

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=4096,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize_function, batched=True)

    # Create dataloader
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=2,  # Per GPU
        shuffle=True,
        collate_fn=data_collator
    )

    # Optimizer with different LR for different layers
    optimizer = torch.optim.AdamW([
        {"params": model.model.embed_tokens.parameters(), "lr": 1e-5},
        {"params": model.model.layers.parameters(), "lr": 2e-5},
        {"params": model.lm_head.parameters(), "lr": 1e-5},
    ], weight_decay=0.01)

    # Learning rate scheduler
    from transformers import get_cosine_schedule_with_warmup
    num_training_steps = len(dataloader) * 3  # 3 epochs
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=1000,
        num_training_steps=num_training_steps
    )

    # Prepare for distributed training
    model, optimizer, dataloader, scheduler = accelerator.prepare(
        model, optimizer, dataloader, scheduler
    )

    # Training loop
    model.train()
    global_step = 0

    for epoch in range(3):
        for batch in dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                # Gradient clipping
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            global_step += 1

            if global_step % 100 == 0 and accelerator.is_main_process:
                print(f"Step {global_step}, Loss: {loss.item():.4f}")

            if global_step % 1000 == 0:
                accelerator.save_state(f"{output_dir}/checkpoint-{global_step}")

    # Save final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(
        f"{output_dir}/final",
        save_function=accelerator.save
    )

if __name__ == "__main__":
    train_distributed()
```

---

## Fine-Tuning with LoRA/QLoRA

Fine-tuning is the most efficient way to adapt existing models for HubLab IO. LoRA (Low-Rank Adaptation) and QLoRA (Quantized LoRA) allow fine-tuning large models on consumer hardware.

### LoRA Fine-Tuning

```python
# finetune_lora.py
"""
LoRA fine-tuning for HubLab IO models.

Memory requirements:
  - 0.5B model: 4GB VRAM
  - 1B model: 6GB VRAM
  - 3B model: 8GB VRAM
  - 7B model: 12GB VRAM
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training
)
from datasets import load_dataset

def finetune_lora(
    base_model: str = "Qwen/Qwen2-0.5B",
    dataset_name: str = "your-dataset",
    output_dir: str = "lora-adapter",
    lora_r: int = 64,
    lora_alpha: int = 128,
    lora_dropout: float = 0.05,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """
    Fine-tune a model using LoRA for HubLab IO deployment.

    Args:
        base_model: HuggingFace model ID (Llama, Qwen2, Phi3, etc.)
        dataset_name: Training dataset
        output_dir: Where to save LoRA adapter
        lora_r: LoRA rank (higher = more params, better quality)
        lora_alpha: LoRA alpha scaling
        lora_dropout: Dropout for LoRA layers
    """

    print(f"Loading base model: {base_model}")

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",  # Attention
            "gate_proj", "up_proj", "down_proj"       # MLP
        ],
        bias="none",
        inference_mode=False
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load and prepare dataset
    dataset = load_dataset(dataset_name, split="train")

    def format_instruction(example):
        """Format dataset for instruction tuning."""
        if "instruction" in example and "response" in example:
            text = f"""<|system|>
You are a helpful AI assistant running on HubLab IO.<|end|>
<|user|>
{example['instruction']}<|end|>
<|assistant|>
{example['response']}<|end|>"""
        elif "text" in example:
            text = example["text"]
        else:
            text = str(example)
        return {"text": text}

    dataset = dataset.map(format_instruction)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )

    tokenized_dataset = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        bf16=True,
        gradient_checkpointing=True,
        optim="adamw_torch_fused",
        report_to="wandb"
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Starting LoRA fine-tuning...")
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"LoRA adapter saved to {output_dir}")

    # Merge LoRA with base model for GGUF conversion
    merge_and_save(model, tokenizer, f"{output_dir}/merged")

    return model, tokenizer

def merge_and_save(model, tokenizer, output_path):
    """Merge LoRA adapter with base model."""
    print("Merging LoRA adapter with base model...")

    merged_model = model.merge_and_unload()
    merged_model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print(f"Merged model saved to {output_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2-0.5B")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="lora-adapter")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    finetune_lora(
        base_model=args.model,
        dataset_name=args.dataset,
        output_dir=args.output,
        lora_r=args.rank,
        num_epochs=args.epochs
    )
```

### QLoRA Fine-Tuning (4-bit)

```python
# finetune_qlora.py
"""
QLoRA fine-tuning - 4-bit quantized training.

Memory requirements (approx 70% less than LoRA):
  - 1B model: 4GB VRAM
  - 3B model: 6GB VRAM
  - 7B model: 8GB VRAM
  - 8B model: 10GB VRAM
"""

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer
)
from peft import (
    get_peft_model,
    LoraConfig,
    prepare_model_for_kbit_training
)
from datasets import load_dataset

def finetune_qlora(
    base_model: str = "meta-llama/Llama-2-7b-hf",
    dataset_name: str = "your-dataset",
    output_dir: str = "qlora-adapter",
    lora_r: int = 64,
    lora_alpha: int = 16,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4
):
    """
    QLoRA fine-tuning with 4-bit quantization.

    This allows fine-tuning 7B models on 8GB VRAM!
    """

    print(f"Loading {base_model} with 4-bit quantization...")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",           # Normalized Float 4
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True       # Nested quantization
    )

    # Load quantized model
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config for QLoRA
    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load dataset
    dataset = load_dataset(dataset_name, split="train")

    def format_prompt(example):
        return {
            "text": f"<|user|>\n{example['instruction']}<|end|>\n<|assistant|>\n{example['response']}<|end|>"
        }

    dataset = dataset.map(format_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training with QLoRA-specific settings
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        logging_steps=10,
        save_steps=500,
        bf16=True,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        max_grad_norm=0.3,
        group_by_length=True,
    )

    from transformers import DataCollatorForLanguageModeling

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    print("Starting QLoRA training...")
    trainer.train()

    # Save adapter
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print(f"QLoRA adapter saved to {output_dir}")
    print("\nTo convert to GGUF, first merge the adapter:")
    print(f"  python merge_lora.py --adapter {output_dir} --base {base_model}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="qlora-adapter")
    parser.add_argument("--rank", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    finetune_qlora(
        base_model=args.model,
        dataset_name=args.dataset,
        output_dir=args.output,
        lora_r=args.rank,
        num_epochs=args.epochs
    )
```

### MLX Fine-Tuning (Apple Silicon)

```python
# finetune_mlx.py
"""
MLX-based fine-tuning for Apple Silicon.

Significantly faster than PyTorch on M1/M2/M3 chips.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from mlx_lm import load, generate
from mlx_lm.tuner import train as mlx_train
from mlx_lm.tuner.trainer import TrainingArgs
import json

def finetune_mlx(
    model_name: str = "mlx-community/Qwen2-0.5B-4bit",
    data_path: str = "train_data.jsonl",
    output_dir: str = "mlx-lora-adapter",
    lora_rank: int = 64,
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4
):
    """
    Fine-tune using MLX on Apple Silicon.

    Data format (JSONL):
    {"text": "full text"}
    or
    {"prompt": "question", "completion": "answer"}
    """

    print(f"Fine-tuning {model_name} with MLX...")

    # Training arguments
    args = TrainingArgs(
        model=model_name,
        data=data_path,
        train=True,
        seed=42,
        lora_layers=16,        # Number of layers to apply LoRA
        lora_rank=lora_rank,
        batch_size=batch_size,
        iters=num_epochs * 1000,  # Adjust based on dataset size
        val_batches=25,
        learning_rate=learning_rate,
        steps_per_report=10,
        steps_per_eval=200,
        adapter_file=f"{output_dir}/adapters.npz",
        save_every=1000,
        max_seq_length=2048,
    )

    # Run training
    mlx_train(args)

    print(f"MLX LoRA adapter saved to {output_dir}")

    # Test the model
    model, tokenizer = load(model_name, adapter_file=f"{output_dir}/adapters.npz")

    response = generate(
        model, tokenizer,
        prompt="Hello, how are you?",
        max_tokens=100
    )
    print(f"\nTest generation:\n{response}")

def convert_mlx_to_pytorch(adapter_path: str, output_path: str):
    """Convert MLX adapter to PyTorch format for GGUF conversion."""
    import numpy as np
    import torch

    # Load MLX adapter
    adapters = mx.load(adapter_path)

    # Convert to PyTorch
    pytorch_state = {}
    for key, value in adapters.items():
        # Convert MLX array to numpy, then to torch
        np_array = np.array(value)
        pytorch_state[key] = torch.from_numpy(np_array)

    torch.save(pytorch_state, output_path)
    print(f"Converted adapter saved to {output_path}")

if __name__ == "__main__":
    # Example: Fine-tune Qwen2-0.5B
    finetune_mlx(
        model_name="mlx-community/Qwen2-0.5B-4bit",
        data_path="train.jsonl",
        output_dir="mlx-adapter"
    )
```

---

## Training MoE-R Experts

HubLab IO uses the MoE-R (Mixture of Real Experts) architecture from Jupiter. This section covers training domain-specific expert models.

### Understanding MoE-R Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         MoE-R System                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Input Query ──▶ Router ──┬──▶ Expert 1 (Python)     ──┐            │
│                           ├──▶ Expert 2 (Rust)       ──┼──▶ Synth   │
│                           ├──▶ Expert 3 (General)    ──┤            │
│                           └──▶ Expert N (Domain)     ──┘            │
│                                                                      │
│  Router Strategies:                                                  │
│    - TopK: Select top K experts by score                            │
│    - Threshold: Select experts above confidence threshold           │
│    - Single: Select single best expert                              │
│    - All: Query all experts (consensus)                             │
│                                                                      │
│  Synthesis Strategies:                                               │
│    - Best: Use highest confidence response                          │
│    - WeightedAverage: Combine by confidence                         │
│    - Voting: Majority voting                                        │
│    - Concatenate: Combine all responses                             │
│    - Summarize: Generate summary of responses                       │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Training Expert Models

```python
# train_expert.py
"""
Train domain-specific expert models for MoE-R.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig
from datasets import load_dataset

# Expert domains for HubLab IO
EXPERT_DOMAINS = {
    "python": {
        "keywords": ["python", "pip", "django", "flask", "pandas", "numpy", "pytorch"],
        "description": "Python programming expert",
        "base_model": "Qwen/Qwen2-0.5B",
        "dataset": "python-code-dataset"
    },
    "rust": {
        "keywords": ["rust", "cargo", "crate", "tokio", "async", "lifetime", "borrow"],
        "description": "Rust programming expert",
        "base_model": "Qwen/Qwen2-0.5B",
        "dataset": "rust-code-dataset"
    },
    "system": {
        "keywords": ["linux", "kernel", "driver", "gpio", "uart", "memory", "process"],
        "description": "System programming expert",
        "base_model": "Qwen/Qwen2-0.5B",
        "dataset": "system-programming-dataset"
    },
    "ai": {
        "keywords": ["model", "training", "inference", "neural", "transformer", "llm"],
        "description": "AI/ML expert",
        "base_model": "Qwen/Qwen2-0.5B",
        "dataset": "ai-ml-dataset"
    },
    "general": {
        "keywords": [],  # Handles everything else
        "description": "General assistant",
        "base_model": "Qwen/Qwen2-1.5B",
        "dataset": "general-assistant-dataset"
    }
}

def train_expert(
    domain: str,
    output_dir: str = None,
    num_epochs: int = 3,
    lora_rank: int = 64
):
    """
    Train a domain-specific expert model.

    Args:
        domain: Expert domain (python, rust, system, ai, general)
        output_dir: Output directory for trained model
        num_epochs: Number of training epochs
        lora_rank: LoRA rank
    """

    if domain not in EXPERT_DOMAINS:
        raise ValueError(f"Unknown domain: {domain}. Available: {list(EXPERT_DOMAINS.keys())}")

    config = EXPERT_DOMAINS[domain]
    output_dir = output_dir or f"expert-{domain}"

    print(f"Training {domain} expert...")
    print(f"  Base model: {config['base_model']}")
    print(f"  Keywords: {config['keywords']}")

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        config['base_model'],
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    tokenizer = AutoTokenizer.from_pretrained(config['base_model'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # LoRA config
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank * 2,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)

    # Load domain-specific dataset
    # In practice, you'd load your curated dataset for this domain
    dataset = load_dataset(config['dataset'], split="train")

    # Format for expert training
    def format_expert_prompt(example):
        system_prompt = f"You are a {config['description']} for HubLab IO operating system."

        return {
            "text": f"""<|system|>
{system_prompt}<|end|>
<|user|>
{example['instruction']}<|end|>
<|assistant|>
{example['response']}<|end|>"""
        }

    dataset = dataset.map(format_expert_prompt)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=2048,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, remove_columns=dataset.column_names)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=10,
        save_steps=500,
    )

    from transformers import DataCollatorForLanguageModeling

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    )

    trainer.train()

    # Save
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save expert config
    import json
    expert_config = {
        "domain": domain,
        "keywords": config["keywords"],
        "description": config["description"],
        "base_model": config["base_model"],
        "threshold": 0.5
    }

    with open(f"{output_dir}/expert_config.json", "w") as f:
        json.dump(expert_config, f, indent=2)

    print(f"Expert saved to {output_dir}")
    return model, tokenizer

def train_all_experts():
    """Train all domain experts."""
    for domain in EXPERT_DOMAINS:
        print(f"\n{'='*50}")
        print(f"Training {domain} expert")
        print('='*50)
        train_expert(domain)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=list(EXPERT_DOMAINS.keys()) + ["all"])
    parser.add_argument("--output", default=None)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--rank", type=int, default=64)
    args = parser.parse_args()

    if args.domain == "all":
        train_all_experts()
    else:
        train_expert(args.domain, args.output, args.epochs, args.rank)
```

### Training the Router Model

```python
# train_router.py
"""
Train the MoE-R router model.

The router classifies incoming queries to determine which expert(s) to invoke.
"""

import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
from datasets import Dataset
import json

# Expert domains
DOMAINS = ["python", "rust", "system", "ai", "general"]

def prepare_router_dataset(data_path: str, output_path: str):
    """
    Prepare dataset for router training.

    Input format (JSONL):
    {"text": "How do I use pandas?", "domain": "python"}
    {"text": "What is a lifetime in Rust?", "domain": "rust"}
    """

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            item = json.loads(line)
            data.append({
                "text": item["text"],
                "label": DOMAINS.index(item["domain"])
            })

    dataset = Dataset.from_list(data)
    dataset.save_to_disk(output_path)
    return dataset

def train_router(
    dataset_path: str,
    output_dir: str = "router-model",
    base_model: str = "distilbert-base-uncased",
    num_epochs: int = 5,
    batch_size: int = 32
):
    """
    Train the router classifier.

    Uses a small, fast model (DistilBERT) for efficient routing.
    """

    print("Training MoE-R router...")

    # Load model for classification
    model = AutoModelForSequenceClassification.from_pretrained(
        base_model,
        num_labels=len(DOMAINS)
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    # Load dataset
    from datasets import load_from_disk
    dataset = load_from_disk(dataset_path)

    # Tokenize
    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=256,
            padding="max_length"
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.train_test_split(test_size=0.1)

    # Training
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=2e-5,
        warmup_steps=500,
        logging_steps=100,
        eval_strategy="steps",
        eval_steps=500,
        save_steps=500,
        load_best_model_at_end=True,
    )

    from sklearn.metrics import accuracy_score, f1_score
    import numpy as np

    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": accuracy_score(labels, predictions),
            "f1": f1_score(labels, predictions, average="weighted")
        }

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["test"],
        compute_metrics=compute_metrics
    )

    trainer.train()

    # Save
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")

    # Save domain mapping
    with open(f"{output_dir}/final/domains.json", "w") as f:
        json.dump({"domains": DOMAINS}, f)

    print(f"Router saved to {output_dir}/final")

    # Convert to ONNX for fast inference
    convert_router_to_onnx(model, tokenizer, f"{output_dir}/router.onnx")

def convert_router_to_onnx(model, tokenizer, output_path):
    """Convert router to ONNX for fast inference."""
    import torch.onnx

    model.eval()

    # Dummy input
    dummy_input = tokenizer(
        "This is a test query",
        return_tensors="pt",
        max_length=256,
        padding="max_length",
        truncation=True
    )

    # Export
    torch.onnx.export(
        model,
        (dummy_input["input_ids"], dummy_input["attention_mask"]),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size"},
            "attention_mask": {0: "batch_size"},
            "logits": {0: "batch_size"}
        },
        opset_version=14
    )

    print(f"Router ONNX model saved to {output_path}")

if __name__ == "__main__":
    train_router("router_dataset", "router-model")
```

---

## Distributed Training

For training large models (7B+) or speeding up training with multiple GPUs/nodes.

### Multi-Node Training Setup

```bash
#!/bin/bash
# distributed_setup.sh

# Install dependencies
pip install deepspeed
pip install torch-distributed

# Generate hostfile
cat > hostfile << EOF
node1 slots=8
node2 slots=8
node3 slots=8
node4 slots=8
EOF

# Set environment variables
export MASTER_ADDR=node1
export MASTER_PORT=29500
export WORLD_SIZE=32  # 4 nodes * 8 GPUs
```

### DeepSpeed Training

```python
# train_deepspeed.py
"""
Distributed training with DeepSpeed ZeRO.

Enables training 7B+ models across multiple GPUs/nodes.
"""

import deepspeed
import torch
from transformers import LlamaForCausalLM, LlamaConfig, AutoTokenizer

# DeepSpeed config for ZeRO-3
ds_config = {
    "train_batch_size": 256,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,

    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },

    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 2e-5,
            "warmup_num_steps": 1000,
            "total_num_steps": 100000
        }
    },

    "fp16": {
        "enabled": True,
        "loss_scale": 0,
        "loss_scale_window": 1000,
        "initial_scale_power": 16,
        "hysteresis": 2,
        "min_loss_scale": 1
    },

    "zero_optimization": {
        "stage": 3,  # ZeRO Stage 3 for maximum memory efficiency
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "sub_group_size": 1e9,
        "reduce_bucket_size": "auto",
        "stage3_prefetch_bucket_size": "auto",
        "stage3_param_persistence_threshold": "auto",
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
        "stage3_gather_16bit_weights_on_model_save": True
    },

    "gradient_clipping": 1.0,

    "activation_checkpointing": {
        "partition_activations": True,
        "cpu_checkpointing": True,
        "contiguous_memory_optimization": True,
        "number_checkpoints": None,
        "synchronize_checkpoint_boundary": False,
        "profile": False
    },

    "wall_clock_breakdown": False,
    "tensorboard": {
        "enabled": True,
        "output_path": "logs/deepspeed",
        "job_name": "hublabio-training"
    }
}

def train_with_deepspeed(
    config_name: str = "hublabio-7b",
    dataset_path: str = "processed_dataset",
    output_dir: str = "hublabio-7b-model"
):
    """
    Train with DeepSpeed ZeRO-3.

    Launch with:
      deepspeed --hostfile=hostfile train_deepspeed.py
    """

    # Initialize DeepSpeed
    deepspeed.init_distributed()

    # Load config
    from config import CONFIGS
    model_config = CONFIGS[config_name]

    config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
        max_position_embeddings=model_config.max_position_embeddings,
    )

    # Create model with ZeRO-3 initialization
    with deepspeed.zero.Init(config_dict_or_path=ds_config):
        model = LlamaForCausalLM(config)

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
    tokenizer.pad_token = tokenizer.eos_token

    # Dataset
    from datasets import load_from_disk
    from torch.utils.data import DataLoader
    from transformers import DataCollatorForLanguageModeling

    dataset = load_from_disk(dataset_path)

    def tokenize(example):
        return tokenizer(example["text"], truncation=True, max_length=4096, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized, batch_size=4, shuffle=True, collate_fn=data_collator)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
        model=model,
        config=ds_config
    )

    # Training loop
    model_engine.train()

    for epoch in range(3):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}

            outputs = model_engine(**batch)
            loss = outputs.loss

            model_engine.backward(loss)
            model_engine.step()

            if step % 100 == 0 and deepspeed.comm.get_rank() == 0:
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # Save checkpoint
        model_engine.save_checkpoint(f"{output_dir}/epoch-{epoch}")

    # Save final model
    if deepspeed.comm.get_rank() == 0:
        model_engine.save_16bit_model(f"{output_dir}/final")
        tokenizer.save_pretrained(f"{output_dir}/final")

if __name__ == "__main__":
    train_with_deepspeed()
```

### FSDP Training (PyTorch Native)

```python
# train_fsdp.py
"""
Fully Sharded Data Parallel training with PyTorch.
"""

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import LlamaForCausalLM, LlamaConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import functools

def setup_fsdp():
    """Initialize distributed training."""
    dist.init_process_group("nccl")
    torch.cuda.set_device(dist.get_rank())

def train_fsdp(config_name: str = "hublabio-7b"):
    """
    Train with FSDP.

    Launch: torchrun --nproc_per_node=8 train_fsdp.py
    """

    setup_fsdp()
    rank = dist.get_rank()

    # Model config
    from config import CONFIGS
    model_config = CONFIGS[config_name]

    config = LlamaConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        intermediate_size=model_config.intermediate_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        num_key_value_heads=model_config.num_key_value_heads,
    )

    # Create model
    model = LlamaForCausalLM(config)

    # FSDP wrapping policy (wrap each transformer layer)
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={LlamaDecoderLayer}
    )

    # Mixed precision
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16
    )

    # Wrap model with FSDP
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=torch.cuda.current_device(),
        use_orig_params=True
    )

    # ... training loop same as before ...

    if rank == 0:
        print("Training complete!")

if __name__ == "__main__":
    train_fsdp()
```

---

## Model Conversion to GGUF

After training, models must be converted to GGUF format for HubLab IO deployment.

### Step-by-Step Conversion

```bash
#!/bin/bash
# convert_to_gguf.sh

MODEL_PATH="./hublabio-model/final"
OUTPUT_DIR="./gguf-models"

mkdir -p $OUTPUT_DIR

# Step 1: Convert HuggingFace to GGUF
cd llama.cpp

python convert_hf_to_gguf.py \
    $MODEL_PATH \
    --outfile $OUTPUT_DIR/model-f16.gguf \
    --outtype f16

echo "Created F16 GGUF: $OUTPUT_DIR/model-f16.gguf"

# Step 2: Quantize to different formats
# Q4_K_M - Best balance of quality and size
./llama-quantize \
    $OUTPUT_DIR/model-f16.gguf \
    $OUTPUT_DIR/model-q4_k_m.gguf \
    Q4_K_M

# Q8_0 - Higher quality, larger size
./llama-quantize \
    $OUTPUT_DIR/model-f16.gguf \
    $OUTPUT_DIR/model-q8_0.gguf \
    Q8_0

# Q2_K - Smallest size, lower quality
./llama-quantize \
    $OUTPUT_DIR/model-f16.gguf \
    $OUTPUT_DIR/model-q2_k.gguf \
    Q2_K

echo "Quantized models created in $OUTPUT_DIR"

# Step 3: Verify the models
./llama-cli -m $OUTPUT_DIR/model-q4_k_m.gguf -p "Hello, I am" -n 50
```

### Python Conversion Script

```python
# convert_gguf.py
"""
Complete GGUF conversion pipeline.
"""

import subprocess
import os
import argparse
from pathlib import Path

QUANT_TYPES = {
    "f16": "F16",
    "q8_0": "Q8_0",
    "q6_k": "Q6_K",
    "q5_k_m": "Q5_K_M",
    "q5_k_s": "Q5_K_S",
    "q4_k_m": "Q4_K_M",  # Recommended for HubLab IO
    "q4_k_s": "Q4_K_S",
    "q4_0": "Q4_0",
    "q3_k_m": "Q3_K_M",
    "q3_k_s": "Q3_K_S",
    "q2_k": "Q2_K",
    "iq4_xs": "IQ4_XS",
    "iq3_xxs": "IQ3_XXS",
    "iq2_xxs": "IQ2_XXS",
}

def convert_to_gguf(
    model_path: str,
    output_dir: str,
    llama_cpp_path: str = "./llama.cpp",
    quant_types: list = ["q4_k_m", "q8_0"]
):
    """
    Convert HuggingFace model to GGUF with multiple quantizations.

    Args:
        model_path: Path to HuggingFace model
        output_dir: Output directory for GGUF files
        llama_cpp_path: Path to llama.cpp repository
        quant_types: List of quantization types to generate
    """

    os.makedirs(output_dir, exist_ok=True)
    model_name = Path(model_path).name

    # Step 1: Convert to F16 GGUF
    print("Converting to F16 GGUF...")
    f16_path = os.path.join(output_dir, f"{model_name}-f16.gguf")

    convert_script = os.path.join(llama_cpp_path, "convert_hf_to_gguf.py")
    subprocess.run([
        "python", convert_script,
        model_path,
        "--outfile", f16_path,
        "--outtype", "f16"
    ], check=True)

    print(f"Created: {f16_path}")

    # Step 2: Quantize
    quantize_bin = os.path.join(llama_cpp_path, "llama-quantize")

    for quant in quant_types:
        if quant == "f16":
            continue  # Already have F16

        quant_name = QUANT_TYPES.get(quant, quant.upper())
        output_path = os.path.join(output_dir, f"{model_name}-{quant}.gguf")

        print(f"Quantizing to {quant_name}...")
        subprocess.run([
            quantize_bin,
            f16_path,
            output_path,
            quant_name
        ], check=True)

        # Get file size
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Created: {output_path} ({size_mb:.1f} MB)")

    # Step 3: Generate model card
    generate_model_card(model_path, output_dir, quant_types)

    print("\nConversion complete!")
    print(f"Models saved to: {output_dir}")

def generate_model_card(model_path: str, output_dir: str, quant_types: list):
    """Generate a model card for the GGUF models."""

    model_name = Path(model_path).name

    card = f"""# {model_name} (GGUF)

Model converted for HubLab IO deployment.

## Available Quantizations

| Format | Size | Quality | Recommended For |
|--------|------|---------|-----------------|
"""

    for quant in quant_types:
        gguf_path = os.path.join(output_dir, f"{model_name}-{quant}.gguf")
        if os.path.exists(gguf_path):
            size_mb = os.path.getsize(gguf_path) / (1024 * 1024)

            if "q2" in quant:
                quality, rec = "Low", "Pi Zero, minimal memory"
            elif "q3" in quant:
                quality, rec = "Medium-Low", "Pi 4 (2GB)"
            elif "q4" in quant:
                quality, rec = "Medium", "Pi 4/5 (4GB+)"
            elif "q5" in quant or "q6" in quant:
                quality, rec = "Medium-High", "Pi 5 (8GB)"
            elif "q8" in quant:
                quality, rec = "High", "Server, high quality needed"
            else:
                quality, rec = "Highest", "Development only"

            card += f"| {quant.upper()} | {size_mb:.0f} MB | {quality} | {rec} |\n"

    card += """
## Usage with HubLab IO

```bash
# Copy to HubLab IO
cp model-q4_k_m.gguf /models/

# Load in shell
hublab> ai load /models/model-q4_k_m.gguf

# Test generation
hublab> ?Hello, how are you?
```

## Training Details

- Base architecture: Llama-compatible
- Training framework: PyTorch/HuggingFace
- Fine-tuning method: [LoRA/QLoRA/Full]
- Dataset: [Your dataset]
"""

    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", help="Path to HuggingFace model")
    parser.add_argument("--output", "-o", default="./gguf-output")
    parser.add_argument("--llama-cpp", default="./llama.cpp")
    parser.add_argument("--quant", nargs="+", default=["q4_k_m", "q8_0"])
    args = parser.parse_args()

    convert_to_gguf(
        args.model_path,
        args.output,
        args.llama_cpp,
        args.quant
    )
```

---

## Quantization Guide

### Choosing the Right Quantization

| Quantization | Bits | Quality Loss | Memory | Speed | Recommended For |
|--------------|------|--------------|--------|-------|-----------------|
| F32 | 32 | None | 4x | Slow | Reference only |
| F16 | 16 | ~0% | 2x | Medium | Development |
| Q8_0 | 8 | ~1% | 1x | Fast | High quality inference |
| Q6_K | 6 | ~2% | 0.75x | Fast | Quality-sensitive tasks |
| Q5_K_M | 5.5 | ~3% | 0.69x | Fast | Balanced |
| Q4_K_M | 4.5 | ~5% | 0.56x | Fast | **Recommended** |
| Q4_0 | 4 | ~7% | 0.5x | Fastest | Speed priority |
| Q3_K_M | 3.5 | ~10% | 0.44x | Fast | Memory constrained |
| Q2_K | 2.5 | ~20% | 0.31x | Fast | Extreme compression |
| IQ4_XS | 4.25 | ~4% | 0.53x | Medium | Best 4-bit quality |
| IQ2_XXS | 2.1 | ~25% | 0.26x | Medium | Minimum viable |

### Memory Requirements by Model Size

For Q4_K_M quantization:

| Model Size | Weight Size | Runtime Memory | Raspberry Pi |
|------------|-------------|----------------|--------------|
| 0.5B | ~300 MB | ~500 MB | Pi Zero 2 |
| 1B | ~600 MB | ~1 GB | Pi 4 (2GB) |
| 1.5B | ~900 MB | ~1.5 GB | Pi 4 (2GB) |
| 3B | ~1.8 GB | ~2.5 GB | Pi 4 (4GB) |
| 7B | ~4 GB | ~5.5 GB | Pi 5 (8GB) |
| 8B | ~4.5 GB | ~6 GB | Pi 5 (8GB) |

### Custom Quantization

```python
# custom_quantize.py
"""
Custom quantization for specific layers.

Some layers benefit from higher precision:
- Attention output projections
- First and last layers
"""

import subprocess
import json

def create_importance_matrix(model_path: str, dataset_path: str, output_path: str):
    """
    Create importance matrix for intelligent quantization.

    This identifies which weights are most important for quality.
    """

    # Run calibration to determine importance
    # This is done via llama.cpp's calibration feature

    subprocess.run([
        "./llama-imatrix",
        "-m", f"{model_path}/model-f16.gguf",
        "-f", dataset_path,  # Calibration text
        "-o", output_path,
        "--chunks", "100"
    ], check=True)

    return output_path

def quantize_with_importance(
    model_path: str,
    output_path: str,
    imatrix_path: str,
    quant_type: str = "Q4_K_M"
):
    """
    Quantize using importance matrix for better quality.
    """

    subprocess.run([
        "./llama-quantize",
        "--imatrix", imatrix_path,
        model_path,
        output_path,
        quant_type
    ], check=True)

if __name__ == "__main__":
    # Create importance matrix
    imatrix = create_importance_matrix(
        "model-f16.gguf",
        "calibration_data.txt",
        "importance.dat"
    )

    # Quantize with importance
    quantize_with_importance(
        "model-f16.gguf",
        "model-q4_k_m-imatrix.gguf",
        imatrix,
        "Q4_K_M"
    )
```

---

## Deploying to HubLab IO

### Deployment Steps

```bash
# 1. Copy model to HubLab IO SD card
# Mount the SD card and copy to /models
cp model-q4_k_m.gguf /Volumes/HUBLABIO/models/

# 2. Create model configuration
cat > /Volumes/HUBLABIO/models/model-config.toml << EOF
[model]
name = "hublabio-custom"
path = "/models/model-q4_k_m.gguf"
type = "llama"

[inference]
max_tokens = 512
temperature = 0.7
top_p = 0.9
top_k = 40
context_size = 4096

[expert]
domain = "general"
keywords = []
threshold = 0.5
EOF

# 3. For MoE-R deployment, copy all expert models
cp expert-python-q4_k_m.gguf /Volumes/HUBLABIO/models/experts/
cp expert-rust-q4_k_m.gguf /Volumes/HUBLABIO/models/experts/
cp expert-system-q4_k_m.gguf /Volumes/HUBLABIO/models/experts/
cp router.onnx /Volumes/HUBLABIO/models/router/
```

### Verification on HubLab IO

```bash
# After booting HubLab IO

# Check available models
hublab> ai list
Available models:
  /models/model-q4_k_m.gguf (1.8 GB)
  /models/experts/python-q4_k_m.gguf (300 MB)
  /models/experts/rust-q4_k_m.gguf (300 MB)

# Load model
hublab> ai load /models/model-q4_k_m.gguf
Loading model...
Model loaded: 3B parameters, Q4_K_M quantization
Context size: 4096 tokens
Memory usage: 2.1 GB

# Test inference
hublab> ?Write a hello world in Python
AI: Here's a simple Python hello world program:

```python
print("Hello, World!")
```

# Check model info
hublab> ai info
Model: hublabio-custom
Parameters: 3B
Quantization: Q4_K_M
Memory: 2.1 GB / 4.0 GB available
Tokens/sec: 12.5
```

### Expert Deployment for MoE-R

```bash
# Deploy MoE-R expert system

# 1. Copy expert models
hublab> pkg install moe-r-experts

# 2. Configure experts
hublab> settings moe

MoE-R Configuration:
  Routing Strategy: TopK (2)
  Synthesis Strategy: Best
  Active Experts:
    - python (300MB)
    - rust (300MB)
    - system (300MB)
    - general (600MB)

# 3. Test routing
hublab> moe route "How do I use pandas?"
Query: "How do I use pandas?"
  → Expert: python (confidence: 0.92)
  → Expert: general (confidence: 0.45)
  Selected: python

# 4. Test full MoE inference
hublab> moe query "How do I use pandas?"
[Router] Selected expert: python
[Expert] Processing query...
[Synthesizer] Using best response

AI: Pandas is a powerful Python library for data manipulation...
```

---

## Best Practices

### Training Best Practices

1. **Data Quality Over Quantity**
   - Clean, deduplicated data produces better models
   - Use diverse, high-quality instruction-response pairs
   - Include domain-specific data for expert models

2. **Hyperparameters**
   ```python
   # Recommended starting points
   learning_rate = 2e-5      # Full training
   learning_rate = 2e-4      # LoRA/QLoRA
   warmup_ratio = 0.03
   weight_decay = 0.01
   gradient_accumulation = 8  # Effective batch ~32
   ```

3. **Memory Optimization**
   - Use gradient checkpointing for large models
   - QLoRA for 7B+ models on consumer GPUs
   - DeepSpeed ZeRO-3 for multi-GPU training

4. **Monitoring**
   - Track loss curves with wandb/tensorboard
   - Monitor validation perplexity
   - Check for overfitting early

### Fine-Tuning Best Practices

1. **LoRA Configuration**
   ```python
   # Higher rank = more capacity but more params
   # Start with r=64, increase if underfitting
   lora_r = 64
   lora_alpha = 128  # Usually 2x rank
   lora_dropout = 0.05

   # Target all linear layers for best results
   target_modules = [
       "q_proj", "k_proj", "v_proj", "o_proj",
       "gate_proj", "up_proj", "down_proj"
   ]
   ```

2. **Dataset Size Guidelines**
   - Minimum: 1,000 examples
   - Recommended: 10,000-100,000 examples
   - Quality matters more than quantity

3. **Prevent Overfitting**
   - Use validation set
   - Early stopping on validation loss
   - Lower learning rate if loss spikes

### Deployment Best Practices

1. **Quantization Selection**
   - Pi Zero 2 W: Q2_K or Q3_K
   - Pi 4 (4GB): Q4_K_M
   - Pi 5 (8GB): Q4_K_M or Q5_K_M
   - Server: Q8_0 or F16

2. **Context Size**
   - Smaller context = faster inference
   - 2048 for chat, 4096 for documents
   - Reduce if running low on memory

3. **MoE-R Configuration**
   - Use TopK(2) routing for balanced responses
   - Single expert for speed-critical applications
   - Train domain-specific experts for specialized tasks

---

## Troubleshooting

### Common Issues

**Out of Memory During Training**
```python
# Solutions:
# 1. Reduce batch size
batch_size = 1

# 2. Enable gradient checkpointing
model.gradient_checkpointing_enable()

# 3. Use QLoRA instead of LoRA
# 4. Use DeepSpeed ZeRO-3
# 5. Reduce context length
max_length = 1024
```

**Model Quality Issues**
```python
# 1. Check data quality
# - Remove duplicates
# - Fix formatting
# - Balance topics

# 2. Adjust hyperparameters
learning_rate = 1e-5  # Lower if unstable
num_epochs = 5        # More epochs if underfitting

# 3. Increase LoRA rank
lora_r = 128  # Higher capacity
```

**GGUF Conversion Fails**
```bash
# Common fixes:

# 1. Update llama.cpp
cd llama.cpp
git pull
make clean && make -j

# 2. Check model format
python -c "from transformers import AutoConfig; print(AutoConfig.from_pretrained('model_path'))"

# 3. Use specific architecture flag
python convert_hf_to_gguf.py model_path --model-type llama
```

**Slow Inference on Pi**
```bash
# 1. Use appropriate quantization
# Q4_K_M is recommended

# 2. Reduce context size
hublab> ai config context_size 2048

# 3. Enable continuous batching (if available)
hublab> ai config batch_mode continuous

# 4. Use smaller model
# 0.5B-1B for Pi 4, up to 3B for Pi 5
```

---

## Appendix

### Recommended Base Models

| Model | Size | Architecture | Best For |
|-------|------|--------------|----------|
| Qwen2-0.5B | 0.5B | Qwen2 | Pi Zero, fast inference |
| Qwen2-1.5B | 1.5B | Qwen2 | Balanced performance |
| Llama-3.2-1B | 1B | Llama | General purpose |
| Llama-3.2-3B | 3B | Llama | High quality, Pi 5 |
| Phi-3-mini | 3.8B | Phi3 | Reasoning tasks |
| Mistral-7B | 7B | Mistral | Best quality (Pi 5 8GB) |

### Dataset Resources

- **General**: OpenHermes, UltraChat, LIMA
- **Code**: CodeAlpaca, CodeFeedback, StarCoder-data
- **Instruction**: Dolly, FLAN, Natural Instructions
- **Conversation**: ShareGPT, WizardLM

### Model Hosting

After training, you can host models for easy distribution:

1. **HuggingFace Hub**
   ```bash
   huggingface-cli upload username/model-name ./gguf-output
   ```

2. **HubLab IO Package Repository**
   ```bash
   # Create package manifest
   cat > package.toml << EOF
   name = "my-custom-model"
   version = "1.0.0"
   category = "ai"
   description = "Custom trained model for HubLab IO"
   files = ["model-q4_k_m.gguf"]
   EOF

   # Upload to repository
   pkg publish ./
   ```

---

*Last updated: December 2024*
*HubLab IO Version: 0.1.0*
