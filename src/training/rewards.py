"""
Reward Functions for RL Training
================================

Composite reward function for GRPO:
    R_total = w1*R_IoU + w2*R_Acc + w3*R_Format + w4*R_Step

Components:
- R_IoU: Intersection over Union with ground truth boxes
- R_Acc: Accuracy of final answer (via LLM judge or exact match)
- R_Format: JSON format validity
- R_Step: Step penalty to encourage efficiency
"""

import json
import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RewardConfig:
    """Configuration for reward computation"""
    iou_weight: float = 0.3
    acc_weight: float = 0.4
    format_weight: float = 0.2
    step_penalty: float = -0.1
    
    # Thresholds
    min_iou_threshold: float = 0.3  # Minimum IoU to get partial credit
    max_steps: int = 5  # Steps beyond this get extra penalty


def compute_iou(box1: List[float], box2: List[float]) -> float:
    """
    Compute Intersection over Union between two boxes.
    
    Boxes are [x_min, y_min, x_max, y_max] format.
    """
    # Intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
        
    intersection = (x2 - x1) * (y2 - y1)
    
    # Union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    if union <= 0:
        return 0.0
        
    return intersection / union


def compute_iou_reward(
    predicted_boxes: List[List[float]],
    ground_truth_boxes: List[List[float]],
    config: Optional[RewardConfig] = None
) -> float:
    """
    Compute IoU-based reward.
    
    Strategy:
    - Match each predicted box to best GT box
    - Average IoU across all GT boxes
    - Penalize extra predictions (false positives)
    
    Args:
        predicted_boxes: List of [x1, y1, x2, y2] boxes
        ground_truth_boxes: List of [x1, y1, x2, y2] boxes
        
    Returns:
        Reward in [0, 1]
    """
    config = config or RewardConfig()
    
    if not ground_truth_boxes:
        # No GT boxes - penalize any predictions
        return 0.0 if predicted_boxes else 1.0
        
    if not predicted_boxes:
        # No predictions but GT exists
        return 0.0
        
    # Compute IoU matrix
    iou_matrix = np.zeros((len(predicted_boxes), len(ground_truth_boxes)))
    
    for i, pred in enumerate(predicted_boxes):
        for j, gt in enumerate(ground_truth_boxes):
            iou_matrix[i, j] = compute_iou(pred, gt)
            
    # Greedy matching: assign each GT to best prediction
    matched_ious = []
    used_preds = set()
    
    for gt_idx in range(len(ground_truth_boxes)):
        best_iou = 0.0
        best_pred = -1
        
        for pred_idx in range(len(predicted_boxes)):
            if pred_idx not in used_preds:
                if iou_matrix[pred_idx, gt_idx] > best_iou:
                    best_iou = iou_matrix[pred_idx, gt_idx]
                    best_pred = pred_idx
                    
        if best_pred >= 0:
            used_preds.add(best_pred)
            matched_ious.append(best_iou)
        else:
            matched_ious.append(0.0)
            
    # Average IoU
    avg_iou = np.mean(matched_ious) if matched_ious else 0.0
    
    # Penalize false positives (unmatched predictions)
    num_false_positives = len(predicted_boxes) - len(used_preds)
    fp_penalty = 0.1 * num_false_positives
    
    reward = max(0.0, avg_iou - fp_penalty)
    return reward


def compute_format_reward(response: str) -> float:
    """
    Check if response contains valid JSON.
    
    Returns:
        1.0 if valid JSON with required fields
        0.5 if valid JSON but missing fields
        0.0 if invalid JSON
    """
    try:
        # Extract JSON from response
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return 0.0
            
        json_str = match.group(0)
        parsed = json.loads(json_str)
        
        # Check required fields
        required = ['thought', 'action']
        has_all = all(k in parsed for k in required)
        
        if has_all:
            return 1.0
        else:
            return 0.5
            
    except (json.JSONDecodeError, Exception):
        return 0.0


def compute_accuracy_reward(
    predicted_answer: str,
    ground_truth: str,
    use_llm_judge: bool = False,
    llm_client: Any = None
) -> float:
    """
    Compute accuracy reward for the final answer.
    
    Methods:
    1. Exact match (simple)
    2. LLM judge (more flexible)
    
    Args:
        predicted_answer: Model's answer
        ground_truth: Expected answer
        use_llm_judge: Use LLM to judge correctness
        llm_client: LLM client for judging
        
    Returns:
        Reward in [0, 1]
    """
    if not predicted_answer or not ground_truth:
        return 0.0
        
    # Normalize
    pred_norm = predicted_answer.lower().strip()
    gt_norm = ground_truth.lower().strip()
    
    # Exact match
    if pred_norm == gt_norm:
        return 1.0
        
    # Fuzzy match - check key terms
    gt_terms = set(gt_norm.split())
    pred_terms = set(pred_norm.split())
    
    overlap = len(gt_terms & pred_terms)
    if overlap > 0:
        jaccard = overlap / len(gt_terms | pred_terms)
        if jaccard > 0.5:
            return 0.8 * jaccard
            
    # LLM judge (if enabled)
    if use_llm_judge and llm_client:
        return _llm_judge_accuracy(predicted_answer, ground_truth, llm_client)
        
    return 0.0


def _llm_judge_accuracy(
    predicted: str,
    ground_truth: str,
    llm_client: Any
) -> float:
    """Use LLM to judge answer correctness"""
    prompt = f"""Evaluate if the predicted answer is correct.

Ground Truth: {ground_truth}
Predicted: {predicted}

Rate from 0.0 (completely wrong) to 1.0 (correct).
Only output a single number."""
    
    try:
        response = llm_client.generate(prompt)
        score = float(response.strip())
        return max(0.0, min(1.0, score))
    except Exception:
        return 0.5  # Default if LLM fails


def compute_step_reward(num_steps: int, config: Optional[RewardConfig] = None) -> float:
    """
    Compute step penalty.
    
    Encourages efficiency by penalizing extra steps.
    
    Args:
        num_steps: Number of reasoning steps taken
        
    Returns:
        Negative reward (penalty)
    """
    config = config or RewardConfig()
    
    # Base penalty
    penalty = config.step_penalty * num_steps
    
    # Extra penalty for exceeding max steps
    if num_steps > config.max_steps:
        extra = (num_steps - config.max_steps) * 0.2
        penalty -= extra
        
    return penalty


class RewardFunction:
    """
    Composite reward function for GRPO training.
    
    R_total = w1*R_IoU + w2*R_Acc + w3*R_Format + w4*R_Step
    
    Example:
        reward_fn = RewardFunction()
        reward = reward_fn.compute(
            response="...",
            predicted_boxes=[[10, 20, 100, 200]],
            ground_truth_boxes=[[15, 25, 95, 190]],
            num_steps=3
        )
    """
    
    def __init__(self, config: Optional[RewardConfig] = None):
        self.config = config or RewardConfig()
        
    def compute(
        self,
        response: str,
        predicted_boxes: Optional[List[List[float]]] = None,
        ground_truth_boxes: Optional[List[List[float]]] = None,
        predicted_answer: Optional[str] = None,
        ground_truth_answer: Optional[str] = None,
        num_steps: int = 1
    ) -> Dict[str, float]:
        """
        Compute composite reward.
        
        Returns:
            Dict with individual rewards and total
        """
        rewards = {}
        
        # R_Format
        rewards['format'] = compute_format_reward(response)
        
        # R_IoU
        if predicted_boxes is not None or ground_truth_boxes is not None:
            rewards['iou'] = compute_iou_reward(
                predicted_boxes or [],
                ground_truth_boxes or [],
                self.config
            )
        else:
            rewards['iou'] = 0.0
            
        # R_Acc
        if predicted_answer or ground_truth_answer:
            rewards['accuracy'] = compute_accuracy_reward(
                predicted_answer or "",
                ground_truth_answer or ""
            )
        else:
            rewards['accuracy'] = rewards['format']  # Fall back to format
            
        # R_Step
        rewards['step'] = compute_step_reward(num_steps, self.config)
        
        # Composite
        rewards['total'] = (
            self.config.iou_weight * rewards['iou'] +
            self.config.acc_weight * rewards['accuracy'] +
            self.config.format_weight * rewards['format'] +
            rewards['step']  # Step penalty is already negative
        )
        
        return rewards
    
    def get_weights(self) -> Dict[str, float]:
        """Return current weights"""
        return {
            'iou': self.config.iou_weight,
            'accuracy': self.config.acc_weight,
            'format': self.config.format_weight,
            'step_penalty': self.config.step_penalty
        }


# Testing
if __name__ == "__main__":
    # Test IoU
    box1 = [10, 20, 100, 200]
    box2 = [15, 25, 95, 190]
    iou = compute_iou(box1, box2)
    print(f"IoU: {iou:.4f}")
    
    # Test format reward
    valid_json = '{"thought": "test", "action": "detect"}'
    invalid_json = "not json"
    print(f"Valid JSON reward: {compute_format_reward(valid_json)}")
    print(f"Invalid JSON reward: {compute_format_reward(invalid_json)}")
    
    # Test composite
    reward_fn = RewardFunction()
    result = reward_fn.compute(
        response=valid_json,
        predicted_boxes=[[10, 20, 100, 200]],
        ground_truth_boxes=[[15, 25, 95, 190]],
        num_steps=2
    )
    print(f"Composite reward: {result}")
