"""
BiomedCLIP Tool - Medical Image Classification
=============================================

Zero-shot medical image classification using BiomedCLIP model.
Supports visual triage for detecting imaging modalities and abnormalities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image

from ..utils.image import base64_to_image
from ..utils.constants import MODALITY_LABELS, ABNORMALITY_LABELS

logger = logging.getLogger(__name__)


class BiomedCLIPTool:
    """
    Medical image classification using Microsoft BiomedCLIP.
    
    Features:
    - Zero-shot classification with custom labels
    - Visual triage for medical imaging modalities
    - Abnormality detection with confidence scores
    
    Example:
        tool = BiomedCLIPTool()
        result = tool.classify(image, ["X-ray", "CT scan", "MRI"])
        print(result)  # {"label": "X-ray", "confidence": 0.95, ...}
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedCLIP-PubMedBERT_PMC-ViT-B-16-224",
        device: Optional[str] = None,
        load_on_init: bool = True
    ):
        """
        Initialize BiomedCLIP tool.
        
        Args:
            model_name: HuggingFace model name
            device: Device to use (auto-detect if None)
            load_on_init: Whether to load model immediately
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        if load_on_init:
            self.load_model()
    
    def load_model(self) -> None:
        """Load BiomedCLIP model and processor."""
        if self.model is not None:
            return
        
        try:
            from transformers import AutoModel, AutoProcessor, AutoTokenizer
            
            logger.info(f"Loading BiomedCLIP from {self.model_name}...")
            
            self.model = AutoModel.from_pretrained(
                self.model_name,
                trust_remote_code=True
            ).to(self.device)
            
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True
            )
            
            self.model.eval()
            logger.info(f"✓ BiomedCLIP loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            raise
    
    def classify(
        self,
        image: Union[str, Image.Image],
        labels: Optional[List[str]] = None,
        return_all: bool = False
    ) -> Dict[str, Any]:
        """
        Classify image using zero-shot classification.
        
        Args:
            image: PIL Image or base64 string
            labels: List of class labels (uses MODALITY_LABELS if None)
            return_all: If True, return scores for all labels
            
        Returns:
            Dictionary with classification results:
            - label: Top predicted label
            - confidence: Confidence score (0-1)
            - all_scores: (if return_all) Dict of all label scores
        """
        self.load_model()
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        if labels is None:
            labels = MODALITY_LABELS
        
        # Prepare prompts
        prompts = [f"This is a medical image showing {label}" for label in labels]
        
        with torch.no_grad():
            # Process image
            image_inputs = self.processor(
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Process text
            text_inputs = self.tokenizer(
                prompts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Get embeddings
            image_features = self.model.get_image_features(**image_inputs)
            text_features = self.model.get_text_features(**text_inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity
            similarity = (image_features @ text_features.T).softmax(dim=-1)
            scores = similarity[0].cpu().numpy()
        
        # Get top prediction
        top_idx = int(np.argmax(scores))
        
        result = {
            "label": labels[top_idx],
            "confidence": float(scores[top_idx]),
            "success": True
        }
        
        if return_all:
            result["all_scores"] = {
                label: float(score)
                for label, score in zip(labels, scores)
            }
        
        return result
    
    def triage(
        self,
        image: Union[str, Image.Image],
        detect_abnormality: bool = True
    ) -> Dict[str, Any]:
        """
        Visual triage: classify modality and detect abnormality.
        
        Args:
            image: PIL Image or base64 string
            detect_abnormality: Also check for abnormalities
            
        Returns:
            Dictionary with:
            - modality: Detected imaging modality
            - modality_confidence: Confidence score
            - abnormality: (if detect_abnormality) Detected abnormality
            - abnormality_confidence: (if detect_abnormality) Confidence
        """
        # Classify modality
        modality_result = self.classify(
            image,
            labels=MODALITY_LABELS,
            return_all=True
        )
        
        result = {
            "modality": modality_result["label"],
            "modality_confidence": modality_result["confidence"],
            "modality_scores": modality_result.get("all_scores", {}),
            "success": True
        }
        
        # Detect abnormality
        if detect_abnormality:
            abnormality_result = self.classify(
                image,
                labels=ABNORMALITY_LABELS,
                return_all=True
            )
            result["abnormality"] = abnormality_result["label"]
            result["abnormality_confidence"] = abnormality_result["confidence"]
            result["abnormality_scores"] = abnormality_result.get("all_scores", {})
        
        return result
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            del self.tokenizer
            self.model = None
            self.processor = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("BiomedCLIP unloaded")
    
    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not loaded"
        return f"BiomedCLIPTool(model={self.model_name}, status={status}, device={self.device})"
