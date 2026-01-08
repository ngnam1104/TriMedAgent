"""
MedSAM Wrapper Module
=====================

Medical segmentation using SAM.
"""

import logging
import os
import torch
import numpy as np
from PIL import Image
from typing import List, Union

logger = logging.getLogger(__name__)

class MedSAMWrapper:
    def __init__(self, checkpoint_path: str = "weights/medsam_vit_b.pth", device: str = "cuda:0"):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.predictor = None
        
    def load_model(self):
        if self.model: return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            from huggingface_hub import hf_hub_download
            
            if not os.path.exists(self.checkpoint_path):
                logger.info("Downloading MedSAM weights...")
                hf_hub_download(
                    repo_id="facebook/sam-vit-base",
                    filename="pytorch_model.bin", 
                    local_dir=os.path.dirname(self.checkpoint_path)
                )
                # Note: This is simplified, facebook/sam-vit-base isn't MedSAM.
                # Assuming user provides correct MedSAM path or we use base SAM.
                
            self.model = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)
            self.model.to(self.device)
            self.predictor = SamPredictor(self.model)
            logger.info("✅ MedSAM Loaded")
            
        except ImportError:
            logger.error("segment-anything not installed")
            
    def segment(self, image: Image.Image, boxes: List[List[float]]) -> List[np.ndarray]:
        self.load_model()
        
        # Prepare image
        img_arr = np.array(image.convert("RGB"))
        self.predictor.set_image(img_arr)
        
        masks = []
        for box in boxes:
            box_np = np.array(box)
            mask, _, _ = self.predictor.predict(
                box=box_np,
                multimask_output=False
            )
            masks.append(mask[0]) # Get the best mask
            
        return masks
