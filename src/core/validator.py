"""
Detection Validator - Self-Correction
=====================================

Validates detection results and triggers self-correction.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

from PIL import Image

from ..types.models import TargetSize, VerificationResult
from ..utils.constants import BOX_SIZE_THRESHOLDS

logger = logging.getLogger(__name__)


class DetectionValidator:
    """
    Validates detection results and triggers self-correction.
    
    Implements:
    - Size-based validation (reject boxes too large for target)
    - LLaVA-based semantic verification
    - Confidence-based filtering
    """
    
    def __init__(self, llava_tool=None):
        self.llava = llava_tool
        self.thresholds = BOX_SIZE_THRESHOLDS
    
    def validate_boxes(
        self,
        boxes: List[List[float]],
        scores: List[float],
        image_size: Tuple[int, int],
        target_size: TargetSize,
        min_score: float = 0.2
    ) -> Tuple[List[List[float]], List[float], List[str]]:
        """
        Validate boxes based on size and confidence.
        
        Args:
            boxes: Detected boxes
            scores: Detection scores
            image_size: (width, height) of image
            target_size: Expected size of target
            min_score: Minimum score threshold
            
        Returns:
            Tuple of (valid_boxes, valid_scores, rejection_reasons)
        """
        if not boxes:
            return [], [], []
        
        w, h = image_size
        image_area = w * h
        max_ratio = self.thresholds.get(target_size, 0.35)
        
        valid_boxes = []
        valid_scores = []
        rejections = []
        
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            box_area = (x2 - x1) * (y2 - y1)
            box_ratio = box_area / image_area
            
            # Check score
            if score < min_score:
                rejections.append(f"Box rejected: score {score:.2f} < {min_score}")
                continue
            
            # Check size
            if box_ratio > max_ratio:
                rejections.append(
                    f"Box rejected: size {box_ratio:.1%} > {max_ratio:.0%} "
                    f"(expected {target_size.value})"
                )
                continue
            
            # Check minimum size (too small might be noise)
            if box_ratio < 0.001:
                rejections.append(f"Box rejected: too small ({box_ratio:.3%})")
                continue
            
            valid_boxes.append(box)
            valid_scores.append(score)
        
        logger.info(f"Validation: {len(valid_boxes)}/{len(boxes)} boxes passed "
                   f"(max_ratio={max_ratio:.0%})")
        
        return valid_boxes, valid_scores, rejections
    
    def semantic_verify(
        self,
        image: Image.Image,
        boxes: List[List[float]],
        target_object: str
    ) -> List[VerificationResult]:
        """
        Use LLaVA to semantically verify each detection.
        
        Args:
            image: Original image
            boxes: Boxes to verify
            target_object: What we're looking for
            
        Returns:
            List of VerificationResult for each box
        """
        if not self.llava or not boxes:
            return [VerificationResult(is_valid=True) for _ in boxes]
        
        results = []
        
        for i, box in enumerate(boxes[:5]):  # Limit to 5 boxes
            try:
                # Crop region
                x1, y1, x2, y2 = map(int, box)
                padding = 10
                x1 = max(0, x1 - padding)
                y1 = max(0, y1 - padding)
                x2 = min(image.width, x2 + padding)
                y2 = min(image.height, y2 + padding)
                
                cropped = image.crop((x1, y1, x2, y2))
                
                # Verify with LLaVA
                prompt = f"""Examine this cropped region carefully.
Does this region actually show a {target_object}?

Answer in this format:
VERDICT: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
EXPLANATION: Brief explanation"""

                response = self.llava.query(
                    cropped,
                    prompt,
                    temperature=0.1,
                    max_new_tokens=150
                )
                
                # Parse response
                result = VerificationResult()
                response_upper = response.upper()
                
                if "VERDICT: YES" in response_upper or response_upper.strip().startswith("YES"):
                    result.is_valid = True
                elif "VERDICT: NO" in response_upper:
                    result.is_valid = False
                else:
                    result.is_valid = "YES" in response_upper
                
                if "CONFIDENCE: HIGH" in response_upper:
                    result.confidence = "high"
                elif "CONFIDENCE: LOW" in response_upper:
                    result.confidence = "low"
                else:
                    result.confidence = "medium"
                
                result.explanation = response
                results.append(result)
                
            except Exception as e:
                logger.warning(f"Verification failed for box {i}: {e}")
                results.append(VerificationResult(is_valid=True, confidence="low"))
        
        return results
    
    def suggest_action(
        self,
        has_detections: bool,
        has_valid_boxes: bool,
        target_size: TargetSize,
        iteration: int
    ) -> str:
        """
        Suggest next action based on detection results.
        
        Returns:
            Action string: "accept", "retry_with_zoom", "retry_stricter", "give_up"
        """
        if not has_detections:
            if iteration < 2:
                return "retry_with_zoom"
            else:
                return "give_up"
        
        if not has_valid_boxes:
            if iteration < 2:
                return "retry_stricter"
            else:
                return "accept_partial"
        
        return "accept"
