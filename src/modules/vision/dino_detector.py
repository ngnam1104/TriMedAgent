"""
Grounding DINO Detector
=======================

Text-guided object detection module.
"""

import logging
import os
from typing import Any, Dict, List, Optional, Union

import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
from huggingface_hub import hf_hub_download

logger = logging.getLogger(__name__)

class GroundingDINO:
    def __init__(
        self,
        config_path: str = "weights/GroundingDINO_SwinT_OGC.py",
        checkpoint_path: str = "weights/groundingdino_swint_ogc.pth",
        device: str = "cuda:0",
        box_threshold: float = 0.35,
        text_threshold: float = 0.25
    ):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.model = None
        
    def load_model(self):
        if self.model: return
        
        try:
            # Try loading from local groundingdino-py first
            from groundingdino.util.inference import load_model, predict
            import groundingdino.datasets.transforms as T
            
            self._ensure_weights()
            self.model = load_model(self.config_path, self.checkpoint_path, device=self.device)
            self._predict_fn = predict
            self._transforms = T
            self._backend = "native"
            logger.info("✅ Grounding DINO (Native) Loaded")
            
        except ImportError:
            # Fallback to Transformers
            logger.warning("Native GroundingDINO not found, using Transformers backend.")
            self.processor = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(self.device)
            self._backend = "transformers"
            
    def _ensure_weights(self):
        if not os.path.exists(self.checkpoint_path):
            logger.info(f"Downloading DINO weights to {self.checkpoint_path}...")
            os.makedirs(os.path.dirname(self.checkpoint_path), exist_ok=True)
            hf_hub_download(
                repo_id="ShilongLiu/GroundingDINO",
                filename="groundingdino_swint_ogc.pth",
                local_dir=os.path.dirname(self.checkpoint_path),
                local_dir_use_symlinks=False
            )
        # Config file download
        if not os.path.exists(self.config_path):
             hf_hub_download(
                repo_id="ShilongLiu/GroundingDINO",
                filename="GroundingDINO_SwinT_OGC.cfg.py",
                local_dir=os.path.dirname(self.config_path),
                local_dir_use_symlinks=False
            )
            
    def detect(self, image: Image.Image, prompt: str) -> List[Dict[str, Any]]:
        self.load_model()
        
        if self._backend == "native":
            transform = self._transforms.Compose([
                self._transforms.RandomResize([800], max_size=1333),
                self._transforms.ToTensor(),
                self._transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            img_tensor, _ = transform(image, None)
            
            boxes, logits, phrases = self._predict_fn(
                model=self.model,
                image=img_tensor,
                caption=prompt,
                box_threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                device=self.device
            )
            
            # Convert to XYXY
            w, h = image.size
            results = []
            for box, score, label in zip(boxes, logits, phrases):
                # Box is [cx, cy, w, h] normalized -> XYXY
                cx, cy, bw, bh = box
                x1 = (cx - bw/2) * w
                y1 = (cy - bh/2) * h
                x2 = (cx + bw/2) * w
                y2 = (cy + bh/2) * h
                
                results.append({
                    "box": [float(x1), float(y1), float(x2), float(y2)],
                    "score": float(score),
                    "label": label
                })
            return results
        else:
            # Transformers implementation
            inputs = self.processor(images=image, text=prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, 
                box_threshold=self.box_threshold, 
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]
            )[0]
            
            results = []
            for box, score, label in zip(res["boxes"], res["scores"], res["labels"]):
                results.append({
                    "box": box.cpu().tolist(),
                    "score": float(score),
                    "label": label
                })
            return results
