"""
GRPO Trainer for TriMed-Agent V2
================================

Group Relative Policy Optimization for medical agent training.

Why GRPO instead of PPO?
- Memory efficient: No separate critic/value model needed
- Better for limited GPU memory (T4/A100)
- Compares groups of responses instead of value estimation

Reward Function:
    R_total = 0.3 * R_IoU + 0.4 * R_Acc + 0.2 * R_Format + 0.1 * R_Step

Curriculum Learning:
    Level 1 (Easy): Single-turn, explicit targets
    Level 2 (Medium): Multi-turn with verification
    Level 3 (Hard): Complex reasoning required

Reference: DeepSeek-R1, GRPO paper
"""

import os
import sys
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    GenerationConfig
)
import numpy as np
from tqdm import tqdm

# Import reward functions
from .rewards import RewardFunction, RewardConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass  
class GRPOConfig:
    """Configuration for GRPO training"""
    
    # Model - LLaVA-Med for medical imaging
    base_model: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    sft_adapter: str = "checkpoints/sft_adapter/final"
    
    # GRPO hyperparameters
    group_size: int = 4  # Number of responses per prompt
    temperature: float = 0.7  # Sampling temperature
    beta: float = 0.1  # KL penalty coefficient
    clip_range: float = 0.2  # PPO-style clipping
    
    # Reward weights (passed to RewardFunction)
    iou_weight: float = 0.3
    acc_weight: float = 0.4
    format_weight: float = 0.2
    step_penalty: float = -0.1
    
    # Training
    num_epochs: int = 1
    batch_size: int = 1  # Small for memory
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5  # Lower than SFT
    max_steps: int = 1000
    warmup_steps: int = 50
    
    # Memory optimization
    gradient_checkpointing: bool = True
    use_8bit_adam: bool = True
    bf16: bool = True
    
    # Data
    dataset_path: str = "data/rl_dataset"
    max_seq_length: int = 2048
    max_new_tokens: int = 512
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_levels: List[str] = field(default_factory=lambda: ['easy', 'medium', 'hard'])
    
    # Output
    output_dir: str = "checkpoints/rl_adapter"
    save_steps: int = 100
    logging_steps: int = 10


# =============================================================================
# Dataset
# =============================================================================

class RLDataset(Dataset):
    """
    Dataset for RL training with ground truth for reward computation.
    
    Format (JSONL):
    {
        "prompt": "Question about image",
        "image": "path/to/img.jpg",
        "ground_truth_boxes": [[x1, y1, x2, y2], ...],
        "ground_truth_answer": "expected answer",
        "difficulty": "easy" | "medium" | "hard"
    }
    """
    
    def __init__(
        self,
        data_path: str,
        tokenizer,
        max_length: int = 2048,
        difficulty_filter: Optional[str] = None
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = self._load_data(data_path, difficulty_filter)
        
    def _load_data(self, path: str, difficulty: Optional[str]) -> List[Dict]:
        """Load and optionally filter by difficulty"""
        data = []
        path = Path(path)
        
        files = list(path.glob("*.jsonl")) if path.is_dir() else [path]
        
        for file_path in files:
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        item = json.loads(line)
                        if difficulty is None or item.get('difficulty') == difficulty:
                            data.append(item)
                            
        logger.info(f"Loaded {len(data)} examples" + 
                   (f" (difficulty={difficulty})" if difficulty else ""))
        return data
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.data[idx]
        
        # Format prompt
        prompt = f"<s>[INST] {item['prompt']} [/INST]"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encodings['input_ids'].squeeze(),
            'attention_mask': encodings['attention_mask'].squeeze(),
            'prompt': item['prompt'],
            'ground_truth_boxes': item.get('ground_truth_boxes', []),
            'ground_truth_answer': item.get('ground_truth_answer', ''),
            'difficulty': item.get('difficulty', 'medium')
        }


# =============================================================================
# GRPO Trainer
# =============================================================================

class GRPOTrainer:
    """
    Group Relative Policy Optimization Trainer.
    
    Algorithm:
    1. For each prompt, generate K responses (group)
    2. Compute reward for each response
    3. Normalize rewards within group (relative ranking)
    4. Update policy to maximize relative advantage
    
    Memory efficient: No critic model needed
    """
    
    def __init__(self, config: Optional[GRPOConfig] = None):
        self.config = config or GRPOConfig()
        self.model = None
        self.ref_model = None  # Reference model for KL
        self.tokenizer = None
        self.optimizer = None
        
        # Reward function
        reward_config = RewardConfig(
            iou_weight=self.config.iou_weight,
            acc_weight=self.config.acc_weight,
            format_weight=self.config.format_weight,
            step_penalty=self.config.step_penalty
        )
        self.reward_fn = RewardFunction(reward_config)
        
        # Metrics
        self.metrics = defaultdict(list)
        
    def setup(self):
        """Load model with SFT adapter"""
        from peft import PeftModel
        
        logger.info("Setting up GRPO training...")
        
        # Quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.base_model,
            trust_remote_code=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        logger.info(f"Loading base model: {self.config.base_model}")
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.base_model,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        
        # Load SFT adapter
        logger.info(f"Loading SFT adapter: {self.config.sft_adapter}")
        self.model = PeftModel.from_pretrained(
            base_model,
            self.config.sft_adapter,
            is_trainable=True  # Training mode
        )
        
        # Reference model (frozen copy for KL penalty)
        # Use same model but don't compute gradients
        self.ref_model = self.model
        
        # Gradient checkpointing
        if self.config.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()
            
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=0.01
        )
        
        logger.info("✅ GRPO setup complete")
        
    def generate_responses(
        self,
        prompt_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        num_responses: int
    ) -> Tuple[List[str], List[torch.Tensor]]:
        """
        Generate multiple responses for a prompt.
        
        Returns:
            responses: List of generated text
            log_probs: List of log probabilities for each response
        """
        self.model.eval()
        
        responses = []
        log_probs_list = []
        
        generation_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            do_sample=True,
            top_p=0.9,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        with torch.no_grad():
            for _ in range(num_responses):
                outputs = self.model.generate(
                    input_ids=prompt_ids,
                    attention_mask=attention_mask,
                    generation_config=generation_config,
                    output_scores=True,
                    return_dict_in_generate=True
                )
                
                # Decode response
                response_ids = outputs.sequences[0, prompt_ids.shape[1]:]
                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                responses.append(response_text)
                
                # Compute log probs
                if outputs.scores:
                    log_probs = []
                    for i, scores in enumerate(outputs.scores):
                        probs = F.softmax(scores, dim=-1)
                        token_id = response_ids[i] if i < len(response_ids) else 0
                        log_prob = torch.log(probs[0, token_id] + 1e-8)
                        log_probs.append(log_prob)
                    log_probs_list.append(torch.stack(log_probs))
                else:
                    log_probs_list.append(torch.zeros(1))
                    
        return responses, log_probs_list
    
    def compute_group_rewards(
        self,
        responses: List[str],
        ground_truth_boxes: List[List[float]],
        ground_truth_answer: str
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Compute rewards for a group of responses.
        
        Returns:
            rewards: Array of rewards
            reward_details: List of detailed reward breakdowns
        """
        rewards = []
        details = []
        
        for response in responses:
            # Parse response to extract predictions
            predicted_boxes = self._extract_boxes_from_response(response)
            predicted_answer = self._extract_answer_from_response(response)
            num_steps = self._count_steps(response)
            
            # Compute reward
            reward_dict = self.reward_fn.compute(
                response=response,
                predicted_boxes=predicted_boxes,
                ground_truth_boxes=ground_truth_boxes,
                predicted_answer=predicted_answer,
                ground_truth_answer=ground_truth_answer,
                num_steps=num_steps
            )
            
            rewards.append(reward_dict['total'])
            details.append(reward_dict)
            
        return np.array(rewards), details
    
    def _extract_boxes_from_response(self, response: str) -> List[List[float]]:
        """Extract bounding boxes from response JSON"""
        import re
        
        boxes = []
        try:
            # Find JSON in response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                data = json.loads(match.group(0))
                # Look for boxes in action_input
                action_input = data.get('action_input', {})
                if 'boxes' in action_input:
                    boxes = action_input['boxes']
        except Exception:
            pass
            
        return boxes
    
    def _extract_answer_from_response(self, response: str) -> str:
        """Extract final answer from response"""
        # Simple extraction - look for answer field or take whole response
        try:
            match = re.search(r'"answer"\s*:\s*"([^"]*)"', response)
            if match:
                return match.group(1)
        except Exception:
            pass
        return response[:200]  # Fallback to first 200 chars
    
    def _count_steps(self, response: str) -> int:
        """Count reasoning steps in response"""
        # Count JSON blocks or thought/action pairs
        import re
        steps = len(re.findall(r'"action"\s*:', response))
        return max(1, steps)
    
    def grpo_step(
        self,
        batch: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Single GRPO training step.
        
        1. Generate K responses for prompt
        2. Compute rewards
        3. Normalize rewards (group relative)
        4. Compute policy gradient loss
        5. Update model
        """
        # Get batch data
        prompt_ids = batch['input_ids'].unsqueeze(0).to(self.model.device)
        attention_mask = batch['attention_mask'].unsqueeze(0).to(self.model.device)
        gt_boxes = batch['ground_truth_boxes']
        gt_answer = batch['ground_truth_answer']
        
        # 1. Generate responses
        responses, log_probs_list = self.generate_responses(
            prompt_ids, attention_mask,
            self.config.group_size
        )
        
        # 2. Compute rewards
        rewards, reward_details = self.compute_group_rewards(
            responses, gt_boxes, gt_answer
        )
        
        # 3. Normalize rewards (group relative)
        rewards_normalized = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 4. Compute loss
        self.model.train()
        
        total_loss = 0.0
        for i, (response, log_probs, reward) in enumerate(
            zip(responses, log_probs_list, rewards_normalized)
        ):
            if len(log_probs) > 0:
                # Policy gradient: -reward * log_prob
                # We want to maximize reward, so minimize -reward * log_prob
                response_log_prob = log_probs.sum()
                loss = -reward * response_log_prob
                total_loss += loss
                
        # Average over group
        total_loss = total_loss / self.config.group_size
        
        # 5. Backward and update
        self.optimizer.zero_grad()
        if isinstance(total_loss, torch.Tensor):
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            loss_value = total_loss.item()
        else:
            loss_value = float(total_loss)
        
        # Metrics
        return {
            'loss': loss_value,
            'mean_reward': float(rewards.mean()),
            'max_reward': float(rewards.max()),
            'reward_std': float(rewards.std()),
            **{f'avg_{k}': np.mean([d[k] for d in reward_details]) 
               for k in reward_details[0].keys() if k != 'total'}
        }
    
    def train(
        self,
        train_dataset: Optional[Dataset] = None
    ) -> str:
        """
        Run GRPO training.
        
        Returns:
            Path to saved adapter
        """
        # Setup if needed
        if self.model is None:
            self.setup()
            
        # Create dataset
        if train_dataset is None:
            train_dataset = RLDataset(
                self.config.dataset_path,
                self.tokenizer,
                self.config.max_seq_length
            )
            
        # Training loop
        logger.info("=" * 50)
        logger.info("Starting GRPO Training")
        logger.info(f"Dataset size: {len(train_dataset)}")
        logger.info(f"Group size: {self.config.group_size}")
        logger.info(f"Max steps: {self.config.max_steps}")
        logger.info("=" * 50)
        
        dataloader = DataLoader(
            train_dataset,
            batch_size=1,  # Process one prompt at a time
            shuffle=True
        )
        
        global_step = 0
        epoch = 0
        
        while global_step < self.config.max_steps:
            epoch += 1
            epoch_metrics = defaultdict(list)
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
            for batch in pbar:
                # GRPO step
                metrics = self.grpo_step(batch)
                
                # Track metrics
                for k, v in metrics.items():
                    epoch_metrics[k].append(v)
                    self.metrics[k].append(v)
                    
                global_step += 1
                
                # Logging
                if global_step % self.config.logging_steps == 0:
                    avg_loss = np.mean(epoch_metrics['loss'][-self.config.logging_steps:])
                    avg_reward = np.mean(epoch_metrics['mean_reward'][-self.config.logging_steps:])
                    pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'reward': f'{avg_reward:.4f}'})
                    
                # Save checkpoint
                if global_step % self.config.save_steps == 0:
                    self._save_checkpoint(global_step)
                    
                if global_step >= self.config.max_steps:
                    break
                    
        # Save final
        output_path = self._save_checkpoint('final')
        
        logger.info(f"✅ GRPO Training Complete! Adapter saved to: {output_path}")
        return output_path
    
    def _save_checkpoint(self, step: Any) -> str:
        """Save model checkpoint"""
        output_path = Path(self.config.output_dir) / str(step)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.model.save_pretrained(str(output_path))
        self.tokenizer.save_pretrained(str(output_path))
        
        # Save metrics
        with open(output_path / "metrics.json", 'w') as f:
            json.dump({k: list(v) for k, v in self.metrics.items()}, f)
            
        # Save config
        with open(output_path / "grpo_config.json", 'w') as f:
            json.dump(vars(self.config), f, indent=2, default=str)
            
        logger.info(f"Saved checkpoint: {output_path}")
        return str(output_path)
    
    def train_curriculum(self) -> str:
        """
        Train with curriculum learning.
        
        Progressively increase difficulty:
        1. Easy examples first
        2. Then medium
        3. Finally hard
        """
        if not self.config.use_curriculum:
            return self.train()
            
        logger.info("Using Curriculum Learning")
        
        for level in self.config.curriculum_levels:
            logger.info(f"\n{'='*50}")
            logger.info(f"Curriculum Level: {level.upper()}")
            logger.info(f"{'='*50}")
            
            # Create filtered dataset
            dataset = RLDataset(
                self.config.dataset_path,
                self.tokenizer,
                self.config.max_seq_length,
                difficulty_filter=level
            )
            
            if len(dataset) == 0:
                logger.warning(f"No examples for difficulty={level}, skipping")
                continue
                
            # Adjust steps based on level
            original_steps = self.config.max_steps
            level_steps = original_steps // len(self.config.curriculum_levels)
            self.config.max_steps = level_steps
            
            # Train on this level
            self.train(dataset)
            
            # Restore
            self.config.max_steps = original_steps
            
        return str(Path(self.config.output_dir) / "final")


# =============================================================================
# CLI
# =============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Train TriMed-Agent with GRPO")
    parser.add_argument("--base_model", type=str, default="chaoyinshe/llava-med-v1.5-mistral-7b-hf")
    parser.add_argument("--sft_adapter", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="data/rl_dataset")
    parser.add_argument("--output_dir", type=str, default="checkpoints/rl_adapter")
    parser.add_argument("--group_size", type=int, default=4)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--curriculum", action="store_true")
    
    args = parser.parse_args()
    
    config = GRPOConfig(
        base_model=args.base_model,
        sft_adapter=args.sft_adapter,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        group_size=args.group_size,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        use_curriculum=args.curriculum
    )
    
    trainer = GRPOTrainer(config)
    
    if config.use_curriculum:
        output_path = trainer.train_curriculum()
    else:
        output_path = trainer.train()
    
    print("\n" + "=" * 50)
    print("🎉 GRPO Training Complete!")
    print(f"📁 Adapter saved to: {output_path}")
    print("\nNext steps:")
    print("  1. Test: python -m src.core.orchestrator --adapter", output_path)
    print("  2. Deploy: huggingface-cli upload your-repo", output_path)
    print("=" * 50)


if __name__ == "__main__":
    import re
    logging.basicConfig(level=logging.INFO)
    main()
