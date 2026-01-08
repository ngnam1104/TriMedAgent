"""
TriMed-Agent Training Package
=============================

Two-stage training pipeline:
- Stage 1 (SFT): Supervised Fine-Tuning with LoRA
- Stage 2 (RL): GRPO optimization for better reasoning

Usage:
    from src.training import SFTTrainer, GRPOTrainer
    
    # Stage 1
    sft_trainer = SFTTrainer(config)
    sft_trainer.train()
    
    # Stage 2
    grpo_trainer = GRPOTrainer(config, sft_adapter="outputs/sft-v1")
    grpo_trainer.train()
"""

from .sft_trainer import SFTTrainer, SFTConfig
from .grpo_trainer import GRPOTrainer, GRPOConfig
from .data_processor import MedicalDataProcessor
from .rewards import RewardFunction, compute_iou_reward, compute_format_reward

__all__ = [
    'SFTTrainer',
    'SFTConfig',
    'GRPOTrainer', 
    'GRPOConfig',
    'MedicalDataProcessor',
    'RewardFunction',
    'compute_iou_reward',
    'compute_format_reward'
]
