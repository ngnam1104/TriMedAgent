"""
Grounding DINO Tool - Object Detection
======================================

Text-guided object detection using Grounding DINO model.
Detects and localizes objects in medical images based on text prompts.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image

from ..utils.image import base64_to_image
from ..utils.constants import DETECTION_PROMPTS

logger = logging.getLogger(__name__)


class GroundingDINOTool:
    """
    Object detection using Grounding DINO.
    
    Features:
    - Text-guided object detection
    - Bounding box localization
    - Confidence filtering
    - Medical-specific prompt optimization
    
    Example:
        tool = GroundingDINOTool()
        boxes = tool.detect(image, "tumor. lesion. mass.")
        print(boxes)  # [[x1, y1, x2, y2], ...]
    """
    
    def __init__(
        self,
        model_name: str = "IDEA-Research/grounding-dino-tiny",
        device: Optional[str] = None,
        box_threshold: float = 0.45,
        text_threshold: float = 0.25,
        config_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        load_on_init: bool = False  # Changed default to False
    ):
        """
        Initialize Grounding DINO tool.
        
        Args:
            model_name: HuggingFace model name (fallback if groundingdino-py not available)
            device: Device to use (auto-detect if None)
            box_threshold: Minimum confidence for box detection (default: 0.45)
            text_threshold: Minimum confidence for text matching (default: 0.25)
            config_path: Local path to config file (for groundingdino-py)
            checkpoint_path: Local path to checkpoint (for groundingdino-py)
            load_on_init: Whether to load model immediately
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        
        self.model = None
        self.processor = None
        self._use_transformers = True
        
        if load_on_init:
            self.load_model()
    
    # Alias for compatibility
    def load(self):
        """Alias for load_model (compatibility with notebook)."""
        self.load_model()
        return self
    
    def load_model(self) -> None:
        """Load Grounding DINO model - prioritize local groundingdino-py package."""
        if self.model is not None:
            return
        
        # Try groundingdino-py package first (preferred for local weights)
        try:
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T
            
            model_config = self._get_config_path()
            model_weights = self._get_weights_path()
            
            logger.info(f"Loading Grounding DINO from local files...")
            print(f"🎯 Loading Grounding DINO on {self.device}...")
            
            self.model = load_model(model_config, model_weights, device=self.device)
            self._predict_fn = predict
            self._transforms = T
            self._use_transformers = False
            
            logger.info(f"✓ Grounding DINO loaded on {self.device} (groundingdino-py)")
            print(f"✅ Grounding DINO loaded on {self.device}")
            return
            
        except ImportError:
            logger.info("groundingdino-py not available, trying transformers...")
        except Exception as e:
            logger.warning(f"Failed to load from groundingdino-py: {e}, trying transformers...")
        
        # Fallback to HuggingFace Transformers
        try:
            from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
            
            logger.info(f"Loading Grounding DINO from {self.model_name}...")
            print(f"🎯 Loading Grounding DINO from HuggingFace...")
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(
                self.model_name
            ).to(self.device)
            
            self.model.eval()
            self._use_transformers = True
            logger.info(f"✓ Grounding DINO loaded on {self.device} (transformers)")
            print(f"✅ Grounding DINO loaded on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load Grounding DINO: {e}")
            raise
    
    def _get_config_path(self) -> str:
        """Get or download model config path."""
        # Use local path if provided
        if self.config_path and os.path.exists(self.config_path):
            return self.config_path
        
        # Check common local paths
        local_paths = [
            "weights/GroundingDINO_SwinT_OGC.py",
            "/kaggle/working/weights/GroundingDINO_SwinT_OGC.py",
            "/content/TriMedAgent/weights/GroundingDINO_SwinT_OGC.py",
        ]
        for path in local_paths:
            if os.path.exists(path):
                return path
        
        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        
        config_file = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="GroundingDINO_SwinT_OGC.cfg.py"
        )
        return config_file
    
    def _get_weights_path(self) -> str:
        """Get or download model weights path."""
        # Use local path if provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self.checkpoint_path
        
        # Check common local paths
        local_paths = [
            "weights/groundingdino_swint_ogc.pth",
            "/kaggle/working/weights/groundingdino_swint_ogc.pth",
            "/content/TriMedAgent/weights/groundingdino_swint_ogc.pth",
        ]
        for path in local_paths:
            if os.path.exists(path):
                return path
        
        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        
        weights_file = hf_hub_download(
            repo_id="ShilongLiu/GroundingDINO",
            filename="groundingdino_swint_ogc.pth"
        )
        return weights_file
    
    def detect(
        self,
        image: Union[str, Image.Image],
        text_prompt: str,
        box_threshold: Optional[float] = None,
        text_threshold: Optional[float] = None,
        return_phrases: bool = False
    ) -> Union[List[List[float]], Dict[str, Any]]:
        """
        Detect objects in image based on text prompt.
        
        Args:
            image: PIL Image or base64 string
            text_prompt: Text prompt (e.g., "tumor. mass. lesion.")
            box_threshold: Override default box threshold
            text_threshold: Override default text threshold
            return_phrases: If True, return dict with boxes and phrases
            
        Returns:
            List of [x1, y1, x2, y2] bounding boxes, or dict with boxes and phrases
        """
        self.load_model()
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        box_thresh = box_threshold or self.box_threshold
        text_thresh = text_threshold or self.text_threshold
        
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        if self._use_transformers:
            return self._detect_transformers(
                image, text_prompt, box_thresh, text_thresh, return_phrases
            )
        else:
            return self._detect_groundingdino(
                image, text_prompt, box_thresh, text_thresh, return_phrases
            )
    
    def _detect_transformers(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        return_phrases: bool
    ) -> Union[List[List[float]], Dict[str, Any]]:
        """Detection using HuggingFace Transformers."""
        with torch.no_grad():
            inputs = self.processor(
                images=image,
                text=text_prompt,
                return_tensors="pt"
            ).to(self.device)
            
            outputs = self.model(**inputs)
            
            results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                box_threshold=box_threshold,
                text_threshold=text_threshold,
                target_sizes=[image.size[::-1]]  # (height, width)
            )[0]
        
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        labels = results["labels"]
        
        if return_phrases:
            return {
                "boxes": boxes,
                "scores": scores,
                "phrases": labels,
                "success": True
            }
        
        return boxes
    
    def _detect_groundingdino(
        self,
        image: Image.Image,
        text_prompt: str,
        box_threshold: float,
        text_threshold: float,
        return_phrases: bool
    ) -> Union[List[List[float]], Dict[str, Any]]:
        """Detection using groundingdino-py package."""
        # Transform image
        transform = self._transforms.Compose([
            self._transforms.RandomResize([800], max_size=1333),
            self._transforms.ToTensor(),
            self._transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        
        image_transformed, _ = transform(image, None)
        
        # Run prediction
        boxes, logits, phrases = self._predict_fn(
            model=self.model,
            image=image_transformed,
            caption=text_prompt,
            box_threshold=box_threshold,
            text_threshold=text_threshold,
            device=self.device
        )
        
        # Convert boxes from normalized CXCYWH to pixel XYXY
        w, h = image.size
        boxes_xyxy = []
        for box in boxes:
            boxes_xyxy.append([
                float(box[0] * w - box[2] * w / 2),
                float(box[1] * h - box[3] * h / 2),
                float(box[0] * w + box[2] * w / 2),
                float(box[1] * h + box[3] * h / 2),
            ])
        
        # Convert scores from Tensor to Python list (important for np.argsort)
        scores_list = logits.cpu().tolist()
        
        if return_phrases:
            return {
                "boxes": boxes_xyxy,
                "scores": scores_list,
                "labels": phrases,  # Use "labels" to match notebook
                "success": True
            }
        
        return boxes_xyxy
    
    def detect_medical(
        self,
        image: Union[str, Image.Image],
        category: str = "default"
    ) -> Dict[str, Any]:
        """
        Detect medical objects using predefined prompts.
        
        Args:
            image: PIL Image or base64 string
            category: Medical category (nodule, tumor, fracture, etc.)
            
        Returns:
            Dictionary with detection results
        """
        prompt = DETECTION_PROMPTS.get(category, DETECTION_PROMPTS["default"])
        return self.detect(image, prompt, return_phrases=True)
    
    def get_prompt_for_query(self, query: str) -> str:
        """
        Generate detection prompt from natural language query.
        
        Args:
            query: User query (e.g., "find the tumor", "locate kidneys")
            
        Returns:
            Formatted detection prompt
        """
        query_lower = query.lower()
        
        # Map keywords to prompts
        for category, prompt in DETECTION_PROMPTS.items():
            if category in query_lower:
                return prompt
        
        # Default: extract nouns from query
        words = query_lower.split()
        relevant_words = [w for w in words if len(w) > 3 and w not in 
                        ("find", "detect", "locate", "show", "where", "this", "that", "image")]
        
        if relevant_words:
            return ". ".join(relevant_words[:5]) + "."
        
        return DETECTION_PROMPTS["default"]
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            if self.processor is not None:
                del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("Grounding DINO unloaded")
    
    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not loaded"
        return f"GroundingDINOTool(model={self.model_name}, status={status}, device={self.device})"
