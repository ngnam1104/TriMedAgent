"""
SFT Trainer for TriMed-Agent V2
===============================

Supervised Fine-Tuning with LoRA for the Planner module.

Optimized for Single T4 GPU (Colab/Kaggle) using:
- unsloth for fast LoRA training
- 4-bit quantization
- Gradient checkpointing

Training Output:
    Model learns to output structured JSON plans:
    {
        "thought": "reasoning...",
        "action": "GroundingDINO",
        "action_input": {"prompt": "target"}
    }
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union

import torch
from torch.utils.data import Dataset
from transformers import TrainingArguments, DataCollatorForSeq2Seq

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SFTConfig:
    """Configuration for SFT training"""
    
    # Model - LLaVA-Med for medical imaging
    base_model: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    use_unsloth: bool = False  # LLaVA-Med not supported by unsloth, use standard peft
    
    # LoRA Configuration
    lora_r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.05
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "v_proj", "k_proj", "o_proj"
    ])
    
    # Training
    num_epochs: int = 3
    batch_size: int = 2  # Small for T4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    fp16: bool = False
    bf16: bool = True
    
    # Data
    max_seq_length: int = 2048
    dataset_path: str = "data/sft_dataset"
    
    # Output
    output_dir: str = "checkpoints/sft_adapter"
    save_steps: int = 100
    logging_steps: int = 10
    save_total_limit: int = 3


# =============================================================================
# Dataset
# =============================================================================

class SFTDataset(Dataset):
    """
    Dataset for SFT training.
    
    Format (JSONL):
    {
        "image": "path/to/img.jpg" or null,
        "conversations": [
            {"role": "user", "content": "<image>\nQuestion"},
            {"role": "assistant", "content": "{JSON plan}"}
        ]
    }
    """
    
    SYSTEM_PROMPT = """You are TriMed-Agent, a medical AI assistant.
Analyze medical images and respond with structured JSON plans.
Format:
{
    "thought": "your reasoning",
    "action": "tool name (GroundingDINO, MedSAM, RAG, or answer)",
    "action_input": {"param": "value"}
}"""
    
    def __init__(
        self,
        data_path: Union[str, Path],
        tokenizer,
        max_length: int = 2048,
        include_images: bool = False
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_images = include_images
        self.data = self._load_data(data_path)
        
        logger.info(f"Loaded {len(self.data)} training examples")
        
    def _load_data(self, path: Union[str, Path]) -> List[Dict]:
        """Load data from JSONL or JSON"""
        data = []
        path = Path(path)
        
        # Handle directory or file
        if path.is_dir():
            files = list(path.glob("*.jsonl")) + list(path.glob("train*.json"))
        else:
            files = [path]
            
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.suffix == '.jsonl':
                    for line in f:
                        if line.strip():
                            data.append(json.loads(line))
                else:
                    content = json.load(f)
                    if isinstance(content, list):
                        data.extend(content)
                    else:
                        data.append(content)
                        
        return data
    
    def _format_conversation(self, item: Dict) -> str:
        """Format conversation for Mistral/LLaMA style"""
        conversations = item.get('conversations', [])
        
        # Start with system prompt
        parts = [f"<s>[INST] <<SYS>>\n{self.SYSTEM_PROMPT}\n<</SYS>>\n\n"]
        
        for i, turn in enumerate(conversations):
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                parts.append(f"{content} [/INST] ")
            elif role == 'assistant':
                parts.append(f"{content} </s>")
                # Continue conversation if more turns
                if i < len(conversations) - 1:
                    parts.append("<s>[INST] ")
                    
        return "".join(parts)
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        
        # Format text
        text = self._format_conversation(item)
        
        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'labels': encodings['input_ids'].squeeze().clone()
        }


# =============================================================================
# Trainer
# =============================================================================

class SFTTrainer:
    """
    SFT Trainer with LoRA, optimized for single T4 GPU.
    
    Uses unsloth for 2x faster training when available.
    """
    
    def __init__(self, config: Optional[SFTConfig] = None):
        self.config = config or SFTConfig()
        self.model = None
        self.tokenizer = None
        self.trainer = None
        
    def setup(self):
        """Setup model and tokenizer"""
        if self.config.use_unsloth:
            self._setup_unsloth()
        else:
            self._setup_standard()
            
    def _setup_unsloth(self):
        """Setup using unsloth for fast training"""
        try:
            from unsloth import FastLanguageModel
            
            logger.info(f"Loading model with unsloth: {self.config.base_model}")
            
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.config.base_model,
                max_seq_length=self.config.max_seq_length,
                dtype=None,  # Auto detect
                load_in_4bit=True,
            )
            
            # Add LoRA
            self.model = FastLanguageModel.get_peft_model(
                self.model,
                r=self.config.lora_r,
                target_modules=self.config.target_modules,
                lora_alpha=self.config.lora_alpha,
                lora_dropout=self.config.lora_dropout,
                bias="none",
                use_gradient_checkpointing=True,
                random_state=42,
            )
            
            logger.info("✅ Model loaded with unsloth + LoRA")
            
        except ImportError:
            logger.warning("unsloth not available, falling back to standard setup")
            self._setup_standard()
            
    def _setup_standard(self):
        """Standard setup with transformers + peft"""
        from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
        
        logger.info(f"Loading model: {self.config.base_model}")
        
        # Quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True
        )
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16
        )
        
        # Prepare for training
        self.model = prepare_model_for_kbit_training(self.model)
        
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            self.model.config.use_cache = False
            
        # LoRA
        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        self.model = get_peft_model(self.model, lora_config)
        
        # Log trainable params
        trainable, total = self.model.get_nb_trainable_parameters()
        logger.info(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
    def train(
        self,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None
    ) -> str:
        """
        Run SFT training.
        
        Args:
            train_dataset: Training dataset (uses config.dataset_path if None)
            eval_dataset: Validation dataset
            
        Returns:
            Path to saved adapter
        """
        from transformers import Trainer
        
        # Setup if not done
        if self.model is None:
            self.setup()
            
        # Create datasets
        if train_dataset is None:
            train_dataset = SFTDataset(
                self.config.dataset_path,
                self.tokenizer,
                self.config.max_seq_length
            )
            
        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            fp16=self.config.fp16,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_steps=self.config.save_steps,
            save_total_limit=self.config.save_total_limit,
            evaluation_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.save_steps if eval_dataset else None,
            load_best_model_at_end=bool(eval_dataset),
            report_to=["tensorboard"],
            optim="paged_adamw_8bit" if self.config.use_8bit_adam else "adamw_torch",
            dataloader_num_workers=2,
        )
        
        # Data collator
        data_collator = DataCollatorForSeq2Seq(
            self.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt",
            padding=True
        )
        
        # Trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        logger.info("=" * 50)
        logger.info("Starting SFT Training")
        logger.info(f"Dataset size: {len(train_dataset)}")
        logger.info(f"Epochs: {self.config.num_epochs}")
        logger.info(f"Batch size: {self.config.batch_size} x {self.config.gradient_accumulation_steps}")
        logger.info("=" * 50)
        
        self.trainer.train()
        
        # Save final adapter
        output_path = Path(self.config.output_dir) / "final"
        self.trainer.save_model(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save config
        with open(output_path / "sft_config.json", 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
            
        logger.info(f"✅ SFT Training Complete! Adapter saved to: {output_path}")
        return str(output_path)
    
    def push_to_hub(self, repo_id: str, adapter_name: str = "sft-v1"):
        """Push trained adapter to HuggingFace Hub"""
        if self.model is None:
            raise ValueError("No model to push. Train first.")
            
        path = f"{repo_id}/{adapter_name}"
        self.model.push_to_hub(path)
        self.tokenizer.push_to_hub(path)
        logger.info(f"✅ Pushed to HuggingFace: {path}")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TriMed-Agent SFT")
    parser.add_argument("--base_model", type=str, default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--dataset", type=str, default="data/sft_dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/sft_adapter")
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--no_unsloth", action="store_true")
    
    args = parser.parse_args()
    
    config = SFTConfig(
        base_model=args.base_model,
        use_unsloth=not args.no_unsloth,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    trainer = SFTTrainer(config)
    output_path = trainer.train()
    
    print("\n" + "=" * 50)
    print("🎉 SFT Training Complete!")
    print(f"📁 Adapter saved to: {output_path}")
    print("\nNext steps:")
    print(f"  1. Test: python -c \"from src.training import SFTTrainer; ...\"")
    print(f"  2. RL Training: python -m src.training.grpo_trainer --sft_adapter {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
