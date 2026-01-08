"""
BiomedCLIP Module
=================

Zero-shot medical image classification using BiomedCLIP model.
Supports visual triage for detecting imaging modalities and abnormalities.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image

from ...utils.image import base64_to_image
from ...utils.constants import MODALITY_LABELS, ABNORMALITY_LABELS

logger = logging.getLogger(__name__)

class BiomedCLIPTool:
    """
    Medical image classification using Microsoft BiomedCLIP.
    """
    
    DEFAULT_MODEL = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        load_on_init: bool = False
    ):
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        
        if load_on_init:
            self.load_model()
    
    def load_model(self) -> None:
        """Load BiomedCLIP model using open_clip."""
        if self.model is not None:
            return
        
        try:
            import open_clip
            
            logger.info(f"Loading BiomedCLIP from {self.model_name}...")
            
            self.model, self.preprocess, _ = open_clip.create_model_and_transforms(
                self.model_name,
                device=self.device
            )
            self.tokenizer = open_clip.get_tokenizer(self.model_name)
            
            self.model.eval()
            logger.info(f"✓ BiomedCLIP loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load BiomedCLIP: {e}")
            raise

    # Alias for compatibility
    def load(self):
        self.load_model()
        return self
    
    @torch.no_grad()
    def classify(
        self,
        image: Union[str, Image.Image],
        labels: Optional[List[str]] = None,
        return_all: bool = False
    ) -> Dict[str, Any]:
        self.load_model()
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        if labels is None:
            labels = MODALITY_LABELS
        
        image_input = self.preprocess(image).unsqueeze(0).to(self.device)
        text_inputs = self.tokenizer(labels).to(self.device)
        
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_inputs)
        
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        scores = similarity[0].cpu().numpy()
        
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
        """
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
            del self.preprocess
            del self.tokenizer
            self.model = None
            self.preprocess = None
            self.tokenizer = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
