#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TriMedAgent SFT Training Script (DDP Version)
=============================================
Optimized for Kaggle 2x T4 GPUs using Distributed Data Parallel.

Usage:
    # Full training
    torchrun --nproc_per_node=2 train.py

    # Demo mode (quick test with 20 samples)
    torchrun --nproc_per_node=2 train.py --demo

    # Custom settings
    torchrun --nproc_per_node=2 train.py --epochs 5 --batch_size 4 --lr 2e-5
"""

import os
import sys
import json
import random
import shutil
import zipfile
import logging
import warnings
import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional
from collections import Counter

import torch
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from datasets import load_dataset
from huggingface_hub import login, snapshot_download, HfApi

from transformers import (
    AutoTokenizer,
    AutoProcessor,
    LlavaForConditionalGeneration,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    TaskType,
)

# ============================================================================
# 0. SILENCE WARNINGS & SETUP LOGGING
# ============================================================================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)


def setup_logging(rank: int):
    """Setup logging for distributed training."""
    level = logging.INFO if rank == 0 else logging.WARNING
    logging.basicConfig(
        format=f"[Rank {rank}] %(asctime)s - %(levelname)s - %(message)s",
        level=level,
        datefmt="%H:%M:%S",
    )
    return logging.getLogger(__name__)


# ============================================================================
# 1. ARGUMENT PARSER
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="TriMedAgent SFT Training (DDP)")
    
    # Model settings
    parser.add_argument(
        "--base_model",
        type=str,
        default="microsoft/llava-med-v1.5-mistral-7b",
        help="Base model from HuggingFace",
    )
    
    # LoRA settings
    parser.add_argument("--lora_r", type=int, default=32, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=64, help="LoRA alpha")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    
    # Training settings
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--max_seq_length", type=int, default=1024, help="Max sequence length")
    
    # Data settings
    parser.add_argument("--data_dir", type=str, default="data", help="Data directory")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_adapter")
    parser.add_argument("--demo", action="store_true", help="Demo mode with limited samples")
    parser.add_argument("--skip_download", action="store_true", help="Skip data download")
    
    # HuggingFace settings
    parser.add_argument("--hf_token", type=str, default=None, help="HuggingFace token")
    parser.add_argument("--hub_model_id", type=str, default="nhn309261/trimedagent-sft")
    parser.add_argument("--push_to_hub", action="store_true", help="Push to HuggingFace Hub")
    
    # DDP settings (auto-populated by torchrun)
    parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for DDP")
    
    args = parser.parse_args()
    
    # Handle local_rank from environment (torchrun sets LOCAL_RANK)
    if args.local_rank == -1:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))
    
    return args


# ============================================================================
# 2. DDP SETUP
# ============================================================================
def setup_distributed(args):
    """Initialize distributed training."""
    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])
        args.distributed = args.world_size > 1
    else:
        args.world_size = 1
        args.distributed = False
    
    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
        
    args.device = torch.device(f"cuda:{args.local_rank}" if torch.cuda.is_available() else "cpu")
    args.is_main_process = args.local_rank in [-1, 0]
    
    return args


def cleanup_distributed():
    """Cleanup distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


# ============================================================================
# 3. DATA DOWNLOAD & PREPROCESSING
# ============================================================================
def download_vqa_rad(data_dir: Path, logger):
    """Download VQA-RAD dataset from HuggingFace."""
    raw_dir = data_dir / "raw"
    img_dir = raw_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = raw_dir / "vqa_rad_metadata.json"
    if metadata_path.exists():
        logger.info("VQA-RAD metadata already exists, skipping download.")
        return
    
    logger.info("Downloading VQA-RAD dataset...")
    
    try:
        dataset = load_dataset("flaviagiammarino/vqa-rad")
        all_data = []
        global_idx = 0
        
        for split in dataset.keys():
            for item in tqdm(dataset[split], desc=f"Processing VQA-RAD {split}", disable=not logger.isEnabledFor(logging.INFO)):
                image_obj = item["image"]
                question = item["question"]
                answer = str(item["answer"])
                
                image_filename = f"vqarad_{split}_{global_idx}.jpg"
                image_path = img_dir / image_filename
                
                if image_obj.mode != "RGB":
                    image_obj = image_obj.convert("RGB")
                image_obj.save(image_path)
                
                record = {
                    "image_path": str(image_path),
                    "question": question,
                    "answer": answer,
                    "split": split,
                    "origin": "huggingface",
                }
                all_data.append(record)
                global_idx += 1
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"VQA-RAD: Saved {len(all_data)} samples.")
        
    except Exception as e:
        logger.error(f"Error downloading VQA-RAD: {e}")


def download_slake(data_dir: Path, logger):
    """Download SLAKE dataset from HuggingFace."""
    raw_dir = data_dir / "raw"
    img_dir = raw_dir / "images"
    slake_dir = raw_dir / "SLAKE_HF"
    
    slake_dir.mkdir(parents=True, exist_ok=True)
    img_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_path = raw_dir / "slake_metadata.json"
    if metadata_path.exists():
        logger.info("SLAKE metadata already exists, skipping download.")
        return
    
    logger.info("Downloading SLAKE dataset...")
    
    try:
        local_dir = snapshot_download(
            repo_id="BoKelvin/SLAKE",
            repo_type="dataset",
            local_dir=slake_dir,
            resume_download=True,
        )
        
        # Extract imgs.zip if exists
        zip_path = Path(local_dir) / "imgs.zip"
        source_imgs_dir = Path(local_dir) / "imgs"
        
        if zip_path.exists():
            logger.info("Extracting imgs.zip...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(local_dir)
        
        if not source_imgs_dir.exists():
            possible_dirs = [d for d in Path(local_dir).iterdir() if d.is_dir() and "img" in d.name.lower()]
            if possible_dirs:
                source_imgs_dir = possible_dirs[0]
            else:
                raise FileNotFoundError("Cannot find 'imgs' folder after unzip!")
        
        # Process metadata
        all_data = []
        json_files = ["train.json", "validate.json", "test.json"]
        saved_images = set()
        
        for j_file in json_files:
            j_path = Path(local_dir) / j_file
            if j_path.exists():
                with open(j_path, "r") as f:
                    data = json.load(f)
                
                split_name = j_file.split(".")[0]
                
                for item in tqdm(data, desc=f"Processing SLAKE {split_name}", disable=not logger.isEnabledFor(logging.INFO)):
                    if item.get("q_lang") != "en":
                        continue
                    
                    raw_path = item["img_name"]
                    src_image_path = source_imgs_dir / raw_path
                    safe_filename = raw_path.replace("/", "_")
                    dest_image_path = img_dir / safe_filename
                    
                    if src_image_path.exists():
                        if safe_filename not in saved_images:
                            shutil.copy2(src_image_path, dest_image_path)
                            saved_images.add(safe_filename)
                        
                        record = {
                            "image_path": str(dest_image_path),
                            "question": item["question"],
                            "answer": str(item["answer"]),
                            "modality": item.get("modality", "Unknown"),
                            "location": item.get("location", "Unknown"),
                            "split": split_name,
                            "origin": "slake_hf",
                        }
                        all_data.append(record)
        
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(all_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"SLAKE: Saved {len(all_data)} samples.")
        
    except Exception as e:
        logger.error(f"Error downloading SLAKE: {e}")


def download_medquad(logger):
    """Download MedQuAD dataset for RAG."""
    logger.info("Downloading MedQuAD dataset...")
    try:
        medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
        logger.info(f"MedQuAD: Loaded {len(medquad)} QA pairs.")
        return medquad
    except Exception as e:
        logger.error(f"Error downloading MedQuAD: {e}")
        return None


def create_plan(question: str, q_type: str = "vision") -> dict:
    """Create JSON plan based on question type."""
    q_lower = question.lower()
    
    # RAG scenario (theoretical questions)
    if q_type == "rag":
        return {
            "thought": "This is a theoretical medical question. I need to retrieve external knowledge.",
            "tool": "Knowledge",
            "action": "Medical_RAG",
            "action_input": {"query": question},
        }
    
    # Vision scenario
    small_objects = ["nodule", "fracture", "mass", "lesion", "opacity", "small"]
    
    # Detection
    if any(k in q_lower for k in ["where", "locate", "find", "show", "position"]):
        target = "abnormality"
        for anat in ["lung", "heart", "liver", "kidney", "nodule", "mass", "pneumonia"]:
            if anat in q_lower:
                target = anat
                break
        
        if any(s in q_lower for s in small_objects):
            return {
                "thought": f"Target '{target}' is likely small. I should use the Zoom strategy.",
                "tool": "Vision",
                "action": "ZoomProcessor",
                "action_input": {"target": target, "mode": "smart_crop"},
            }
        else:
            return {
                "thought": f"User wants to find '{target}'. I will scan the whole image.",
                "tool": "Vision",
                "action": "GroundingDINO",
                "action_input": {"target": target},
            }
    
    # Default: Visual Answering
    return {
        "thought": "This requires visual interpretation of the X-ray image.",
        "tool": "Vision",
        "action": "Visual_Answering",
        "action_input": {"focus": "general_analysis"},
    }


def process_vqa(json_path: Path, origin_name: str, logger) -> list:
    """Process VQA dataset to ReAct format."""
    if not json_path.exists():
        return []
    
    with open(json_path, "r") as f:
        data = json.load(f)
    
    processed = []
    for item in tqdm(data, desc=f"Converting {origin_name}", disable=not logger.isEnabledFor(logging.INFO)):
        if not item.get("image_path"):
            continue
        
        plan = create_plan(item.get("question", ""), q_type="vision")
        
        processed.append({
            "id": f"{origin_name}_{len(processed)}",
            "image": item["image_path"],
            "conversations": [
                {"from": "human", "value": f"<image>\n{item.get('question', '')}"},
                {"from": "gpt", "value": json.dumps(plan)},
            ],
        })
    
    return processed


def process_rag(medquad_data, num_samples: int = 3500, dummy_image: str = "data/raw/dummy_rag.jpg", logger=None) -> list:
    """Process MedQuAD dataset to ReAct format."""
    if not medquad_data:
        return []
    
    processed = []
    dataset = medquad_data.shuffle(seed=42).select(range(min(num_samples, len(medquad_data))))
    
    for idx, item in enumerate(tqdm(dataset, desc="Converting MedQuAD", disable=not logger.isEnabledFor(logging.INFO))):
        question = item["Question"]
        q_type_label = item.get("qtype", "general")
        
        plan = create_plan(question, q_type="rag")
        
        processed.append({
            "id": f"RAG_{q_type_label}_{idx}",
            "image": dummy_image,
            "conversations": [
                {"from": "human", "value": f"<image>\n{question}"},
                {"from": "gpt", "value": json.dumps(plan)},
            ],
            "data_source": "MedQuAD",
        })
    
    return processed


def prepare_datasets(data_dir: Path, skip_download: bool, logger):
    """Download and prepare all datasets."""
    raw_dir = data_dir / "raw"
    sft_dir = data_dir / "sft_dataset"
    sft_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = sft_dir / "train.json"
    val_path = sft_dir / "val.json"
    
    # Check if already prepared
    if train_path.exists() and val_path.exists():
        logger.info("SFT datasets already exist.")
        return
    
    if not skip_download:
        # Download datasets
        download_vqa_rad(data_dir, logger)
        download_slake(data_dir, logger)
        medquad = download_medquad(logger)
    else:
        medquad = None
    
    # Create dummy image for RAG
    dummy_path = raw_dir / "dummy_rag.jpg"
    if not dummy_path.exists():
        logger.info("Creating dummy image for RAG data...")
        Image.new("RGB", (224, 224), color="black").save(dummy_path)
    
    # Process datasets
    logger.info("Processing datasets to ReAct format...")
    
    vqa_rad = process_vqa(raw_dir / "vqa_rad_metadata.json", "VQA-RAD", logger)
    slake = process_vqa(raw_dir / "slake_metadata.json", "SLAKE", logger)
    
    if medquad is not None:
        rag_data = process_rag(medquad, dummy_image=str(dummy_path), logger=logger)
    else:
        rag_data = []
    
    # Combine and shuffle
    all_data = vqa_rad + slake + rag_data
    random.shuffle(all_data)
    
    logger.info(f"Total samples: {len(all_data)} (Vision: {len(vqa_rad) + len(slake)}, RAG: {len(rag_data)})")
    
    # Split train/val
    train_data, val_data = train_test_split(all_data, test_size=0.1, random_state=42, shuffle=True)
    
    # Save
    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=2, ensure_ascii=False)
    
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Saved: {len(train_data)} train, {len(val_data)} val samples.")


# ============================================================================
# 4. DATASET CLASS
# ============================================================================
SYSTEM_PROMPT = """You are TriMed-Agent, a medical AI assistant.
Analyze the input and respond with a structured JSON plan.
Format:
{
    "thought": "reasoning about the request",
    "tool": "Vision" or "Knowledge",
    "action": "tool name (GroundingDINO, ZoomProcessor, Medical_RAG, or Visual_Answering)",
    "action_input": {"param": "value"}
}"""


class SFTDataset(Dataset):
    """Dataset for SFT training with ReAct format."""
    
    def __init__(self, data_path: str, processor, max_length: int = 1024):
        self.processor = processor
        self.tokenizer = processor.tokenizer
        self.max_length = max_length
        self.data = []
        
        if os.path.exists(data_path):
            with open(data_path, "r", encoding="utf-8") as f:
                self.data = json.load(f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Process image
        image_path = item.get("image", "")
        try:
            if image_path and os.path.exists(image_path):
                image = Image.open(image_path).convert("RGB")
            else:
                raise FileNotFoundError("Image path invalid")
        except:
            image = Image.new("RGB", (224, 224), color="black")
        
        # Parse conversations
        conversations = item.get("conversations", [])
        user_input = ""
        gpt_response = ""
        
        for turn in conversations:
            role = turn.get("from") or turn.get("role")
            content = turn.get("value") or turn.get("content")
            if role in ["human", "user"]:
                user_input = content.replace("<image>", "").strip()
            elif role in ["gpt", "assistant"]:
                gpt_response = content
        
        # Build prompt (LLaVA 1.5 format)
        full_prompt = f"USER: <image>\n{SYSTEM_PROMPT}\n{user_input}\nASSISTANT: {gpt_response}"
        
        # Tokenize
        inputs = self.processor(
            text=full_prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
        )
        
        input_ids = inputs["input_ids"][0]
        labels = input_ids.clone()
        
        # Mask padding tokens
        if self.tokenizer.pad_token_id is not None:
            labels[labels == self.tokenizer.pad_token_id] = -100
        
        # Mask user prompt (only compute loss on assistant response)
        assistant_token_id = self.tokenizer.encode("ASSISTANT:", add_special_tokens=False)[-1]
        
        try:
            start_idx = (input_ids == assistant_token_id).nonzero(as_tuple=True)[0][-1].item()
            labels[:start_idx + 2] = -100
        except:
            labels[:] = -100
        
        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0],
            "labels": labels,
        }


class LLaVADataCollator:
    """Data collator for LLaVA model."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
    
    def __call__(self, instances):
        input_ids = torch.stack([inst["input_ids"] for inst in instances])
        attention_mask = torch.stack([inst["attention_mask"] for inst in instances])
        pixel_values = torch.stack([inst["pixel_values"] for inst in instances])
        labels = torch.stack([inst["labels"] for inst in instances])
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values,
            "labels": labels,
        }


# ============================================================================
# 5. MODEL LOADING
# ============================================================================
def load_model_and_processor(args, logger):
    """Load model with 4-bit quantization and LoRA."""
    logger.info(f"Loading model: {args.base_model}")
    
    # Quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load processor
    processor = AutoProcessor.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer = processor.tokenizer
    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.unk_token
    
    # Load model with device_map for specific GPU (DDP mode)
    # Each process loads model to its own GPU
    device_map = {"": args.local_rank} if args.distributed else "auto"
    
    model = LlavaForConditionalGeneration.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )
    
    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)
    model.config.use_cache = False
    
    # LoRA config
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        modules_to_save=["embed_tokens", "lm_head"],
    )
    
    model = get_peft_model(model, lora_config)
    
    if args.is_main_process:
        model.print_trainable_parameters()
        
        # Log GPU memory usage
        for i in range(torch.cuda.device_count()):
            mem = torch.cuda.memory_allocated(i) / 1024**3
            logger.info(f"GPU {i}: {mem:.2f} GB used")
    
    return model, processor


# ============================================================================
# 6. TRAINING
# ============================================================================
def train(args, model, processor, train_dataset, val_dataset, logger):
    """Run training with Trainer API."""
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        
        # Memory optimization
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        fp16=True,
        bf16=False,
        
        # Data loading
        dataloader_num_workers=4,
        group_by_length=True,
        remove_unused_columns=False,
        
        # Logging & saving
        logging_steps=1 if args.demo else 10,
        save_strategy="epoch",
        save_total_limit=2,
        evaluation_strategy="epoch" if not args.demo else "no",
        
        # Distributed settings
        local_rank=args.local_rank,
        ddp_find_unused_parameters=False,
        
        # Reporting
        report_to="none",
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=LLaVADataCollator(processor.tokenizer),
    )
    
    # Train
    logger.info("Starting training...")
    trainer.train()
    
    # Save model (only main process)
    if args.is_main_process:
        output_path = Path(args.output_dir) / "final"
        output_path.mkdir(parents=True, exist_ok=True)
        
        trainer.save_model(str(output_path))
        processor.save_pretrained(str(output_path))
        
        # Save training info
        training_info = {
            "base_model": args.base_model,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "target_modules": ["q_proj", "v_proj"],
            "num_epochs": args.epochs,
            "learning_rate": args.lr,
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * args.gradient_accumulation_steps * args.world_size,
            "training_samples": len(train_dataset),
            "demo_mode": args.demo,
        }
        
        with open(output_path / "training_config.json", "w") as f:
            json.dump(training_info, f, indent=2)
        
        # Save loss history
        history = trainer.state.log_history
        steps, losses = [], []
        for entry in history:
            if "loss" in entry:
                steps.append(entry["step"])
                losses.append(entry["loss"])
        
        with open(output_path / "loss_history.json", "w") as f:
            json.dump({"steps": steps, "losses": losses}, f, indent=2)
        
        logger.info(f"Model saved to: {output_path}")
    
    return trainer


# ============================================================================
# 7. PUSH TO HUGGINGFACE
# ============================================================================
def push_to_hub(args, model, processor, logger):
    """Push model to HuggingFace Hub."""
    if not args.is_main_process:
        return
    
    logger.info(f"Pushing to HuggingFace: {args.hub_model_id}")
    
    try:
        model.push_to_hub(args.hub_model_id, token=True)
        processor.push_to_hub(args.hub_model_id, token=True)
        
        # Create model card
        model_card = f"""---
license: apache-2.0
tags:
  - medical
  - vision
  - llava
  - lora
  - trimedagent
base_model: {args.base_model}
datasets:
  - custom
language:
  - en
  - vi
---

# TriMedAgent SFT Adapter

LoRA adapter for TriMedAgent - Medical Visual Agent.

## Training Details

- **Base Model**: `{args.base_model}`
- **LoRA Rank**: {args.lora_r}
- **LoRA Alpha**: {args.lora_alpha}
- **Target Modules**: q_proj, v_proj
- **Epochs**: {args.epochs}
- **Effective Batch Size**: {args.batch_size * args.gradient_accumulation_steps * args.world_size}

## Usage

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM

base_model = AutoModelForCausalLM.from_pretrained("{args.base_model}")
model = PeftModel.from_pretrained(base_model, "{args.hub_model_id}")
```
"""
        
        output_path = Path(args.output_dir) / "final"
        with open(output_path / "README.md", "w") as f:
            f.write(model_card)
        
        # Upload README
        api = HfApi()
        api.upload_file(
            path_or_fileobj=str(output_path / "README.md"),
            path_in_repo="README.md",
            repo_id=args.hub_model_id,
            token=True,
        )
        
        logger.info(f"Successfully pushed to: https://huggingface.co/{args.hub_model_id}")
        
    except Exception as e:
        logger.error(f"Push failed: {e}")


# ============================================================================
# 8. MAIN
# ============================================================================
def main():
    # Parse arguments
    args = parse_args()
    
    # Setup distributed
    args = setup_distributed(args)
    
    # Setup logging
    logger = setup_logging(args.local_rank)
    
    # GPU check
    if args.is_main_process:
        logger.info("=" * 60)
        logger.info("TriMedAgent SFT Training (DDP)")
        logger.info("=" * 60)
        
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                mem_gb = props.total_memory / 1024**3
                logger.info(f"GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        else:
            logger.error("No GPU found!")
            return
        
        logger.info(f"World size: {args.world_size}")
        logger.info(f"Demo mode: {args.demo}")
    
    # HuggingFace login
    if args.hf_token:
        login(token=args.hf_token)
        logger.info("Logged in to HuggingFace")
    else:
        # Try Kaggle secrets
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
            login(token=hf_token)
            logger.info("Logged in via Kaggle Secrets")
        except:
            logger.warning("No HF token found. Push to hub may fail.")
    
    # Prepare data (only main process downloads)
    data_dir = Path(args.data_dir)
    
    if args.is_main_process:
        prepare_datasets(data_dir, args.skip_download, logger)
    
    # Sync all processes
    if args.distributed:
        dist.barrier()
    
    # Load model
    model, processor = load_model_and_processor(args, logger)
    
    # Load datasets
    sft_dir = data_dir / "sft_dataset"
    train_dataset = SFTDataset(str(sft_dir / "train.json"), processor, args.max_seq_length)
    val_dataset = SFTDataset(str(sft_dir / "val.json"), processor, args.max_seq_length)
    
    # Demo mode: limit samples
    if args.demo:
        train_dataset.data = train_dataset.data[:20]
        val_dataset.data = val_dataset.data[:10]
        logger.info(f"Demo mode: {len(train_dataset)} train, {len(val_dataset)} val samples")
    else:
        logger.info(f"Full mode: {len(train_dataset)} train, {len(val_dataset)} val samples")
    
    # Train
    trainer = train(args, model, processor, train_dataset, val_dataset, logger)
    
    # Push to hub
    if args.push_to_hub:
        push_to_hub(args, model, processor, logger)
    
    # Cleanup
    cleanup_distributed()
    
    if args.is_main_process:
        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info(f"Output: {args.output_dir}/final")
        logger.info("=" * 60)


if __name__ == "__main__":
    main()
