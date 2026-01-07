"""
MedSAM Tool - Medical Image Segmentation
========================================

Medical image segmentation using Segment Anything Model (SAM).
Supports box-guided and point-guided segmentation.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import numpy as np
from PIL import Image

from ..utils.image import base64_to_image

logger = logging.getLogger(__name__)


class MedSAMTool:
    """
    Medical image segmentation using SAM/MedSAM.
    
    Features:
    - Box-guided segmentation (from bounding boxes)
    - Point-guided segmentation
    - Multi-mask output with quality scores
    - Post-processing and refinement
    
    Example:
        tool = MedSAMTool()
        masks = tool.segment(image, boxes=[[100, 100, 200, 200]])
        print(len(masks))  # 1
    """
    
    def __init__(
        self,
        model_type: str = "vit_b",
        checkpoint_path: Optional[str] = None,
        device: Optional[str] = None,
        load_on_init: bool = False  # Changed default to False
    ):
        """
        Initialize MedSAM tool.
        
        Args:
            model_type: SAM model type (vit_b, vit_l, vit_h)
            checkpoint_path: Path to model checkpoint (downloads if None)
            device: Device to use (auto-detect if None)
            load_on_init: Whether to load model immediately
        """
        self.model_type = model_type
        self.checkpoint_path = checkpoint_path or "weights/medsam_vit_b.pth"
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        self.sam = None
        self.medsam_model = None  # Alias for compatibility
        self.predictor = None
        self._current_image = None
        
        if load_on_init:
            self.load_model()
    
    # Alias for compatibility
    def load(self):
        """Alias for load_model (compatibility with notebook)."""
        self.load_model()
        return self
    
    def load_model(self) -> None:
        """Load SAM model and create predictor."""
        if self.sam is not None:
            return
        
        try:
            from segment_anything import sam_model_registry, SamPredictor
            
            logger.info(f"Loading SAM model ({self.model_type})...")
            print(f"🎭 Loading MedSAM on {self.device}...")
            
            # Get checkpoint path
            checkpoint = self._download_checkpoint()
            
            # Load model
            self.sam = sam_model_registry[self.model_type](
                checkpoint=checkpoint
            )
            self.sam.to(device=self.device)
            self.medsam_model = self.sam  # Alias
            
            self.predictor = SamPredictor(self.sam)
            
            logger.info(f"✓ SAM loaded on {self.device}")
            print(f"✅ MedSAM loaded on {self.device}")
            
        except ImportError:
            logger.error("segment_anything not installed. Run: pip install segment-anything")
            raise
        except Exception as e:
            logger.error(f"Failed to load SAM: {e}")
            raise
    
    def _download_checkpoint(self) -> str:
        """Download SAM checkpoint or find local file."""
        # Check provided checkpoint_path first
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            return self.checkpoint_path
        
        # Check common local paths
        local_paths = [
            "weights/medsam_vit_b.pth",
            "weights/sam_vit_b_01ec64.pth",
            "/kaggle/working/weights/medsam_vit_b.pth",
            "/kaggle/working/TriMedAgent/weights/medsam_vit_b.pth",
            "/content/TriMedAgent/weights/medsam_vit_b.pth",
            "/content/weights/medsam_vit_b.pth",
        ]
        for path in local_paths:
            if os.path.exists(path):
                logger.info(f"Found local checkpoint: {path}")
                return path
        
        # Download from HuggingFace
        from huggingface_hub import hf_hub_download
        
        checkpoint_map = {
            "vit_b": ("facebook/sam-vit-base", "pytorch_model.bin"),
            "vit_l": ("facebook/sam-vit-large", "pytorch_model.bin"),
            "vit_h": ("facebook/sam-vit-huge", "pytorch_model.bin"),
        }
        
        if self.model_type not in checkpoint_map:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        repo_id, filename = checkpoint_map[self.model_type]
        
        logger.info(f"Downloading SAM checkpoint from {repo_id}...")
        checkpoint_path = hf_hub_download(repo_id=repo_id, filename=filename)
        
        return checkpoint_path
    
    def set_image(self, image: Union[str, Image.Image, np.ndarray]) -> None:
        """
        Set image for segmentation (caches embedding).
        
        Args:
            image: PIL Image, numpy array, or base64 string
        """
        self.load_model()
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        self.predictor.set_image(image)
        self._current_image = image
    
    def segment(
        self,
        image: Optional[Union[str, Image.Image, np.ndarray]] = None,
        boxes: Optional[List[List[float]]] = None,
        points: Optional[List[Tuple[int, int]]] = None,
        point_labels: Optional[List[int]] = None,
        multimask_output: bool = False,
        return_scores: bool = False
    ) -> Union[List[np.ndarray], Dict[str, Any]]:
        """
        Segment image using boxes or points.
        
        Args:
            image: PIL Image, numpy array, or base64 string (uses cached if None)
            boxes: List of [x1, y1, x2, y2] bounding boxes
            points: List of (x, y) point coordinates
            point_labels: Labels for points (1=foreground, 0=background)
            multimask_output: Return multiple masks per prompt
            return_scores: Return quality scores with masks
            
        Returns:
            List of binary masks (numpy arrays), or dict with masks and scores
        """
        self.load_model()
        
        # Set image if provided
        if image is not None:
            self.set_image(image)
        
        if self._current_image is None:
            raise ValueError("No image set. Call set_image() first or provide image.")
        
        all_masks = []
        all_scores = []
        
        # Process boxes
        if boxes is not None:
            for box in boxes:
                masks, scores, _ = self.predictor.predict(
                    box=np.array(box),
                    multimask_output=multimask_output
                )
                
                if multimask_output:
                    # Select best mask
                    best_idx = np.argmax(scores)
                    all_masks.append(masks[best_idx])
                    all_scores.append(float(scores[best_idx]))
                else:
                    all_masks.append(masks[0])
                    all_scores.append(float(scores[0]))
        
        # Process points
        if points is not None:
            points_array = np.array(points)
            labels_array = np.array(point_labels) if point_labels else np.ones(len(points))
            
            masks, scores, _ = self.predictor.predict(
                point_coords=points_array,
                point_labels=labels_array,
                multimask_output=multimask_output
            )
            
            if multimask_output:
                best_idx = np.argmax(scores)
                all_masks.append(masks[best_idx])
                all_scores.append(float(scores[best_idx]))
            else:
                all_masks.append(masks[0])
                all_scores.append(float(scores[0]))
        
        if return_scores:
            return {
                "masks": all_masks,
                "scores": all_scores,
                "success": True
            }
        
        return all_masks
    
    def segment_all(
        self,
        image: Union[str, Image.Image, np.ndarray],
        points_per_side: int = 32,
        pred_iou_thresh: float = 0.88,
        stability_score_thresh: float = 0.95,
        min_mask_region_area: int = 100
    ) -> List[Dict[str, Any]]:
        """
        Automatically segment all objects in image.
        
        Args:
            image: PIL Image, numpy array, or base64 string
            points_per_side: Points per side for grid
            pred_iou_thresh: IoU threshold for predictions
            stability_score_thresh: Stability score threshold
            min_mask_region_area: Minimum mask area
            
        Returns:
            List of dicts with mask, area, bbox, predicted_iou, stability_score
        """
        self.load_model()
        
        try:
            from segment_anything import SamAutomaticMaskGenerator
        except ImportError:
            logger.error("SamAutomaticMaskGenerator not available")
            raise
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        if isinstance(image, Image.Image):
            image = np.array(image.convert("RGB"))
        
        mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=points_per_side,
            pred_iou_thresh=pred_iou_thresh,
            stability_score_thresh=stability_score_thresh,
            min_mask_region_area=min_mask_region_area
        )
        
        masks = mask_generator.generate(image)
        
        return masks
    
    def refine_mask(
        self,
        mask: np.ndarray,
        kernel_size: int = 5,
        iterations: int = 1
    ) -> np.ndarray:
        """
        Refine mask using morphological operations.
        
        Args:
            mask: Binary mask to refine
            kernel_size: Size of morphological kernel
            iterations: Number of iterations
            
        Returns:
            Refined mask
        """
        import cv2
        
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
        )
        
        # Close small holes
        refined = cv2.morphologyEx(
            mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel, iterations=iterations
        )
        
        # Open to remove small noise
        refined = cv2.morphologyEx(
            refined, cv2.MORPH_OPEN, kernel, iterations=iterations
        )
        
        return refined.astype(bool)
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.sam is not None:
            del self.sam
            del self.predictor
            self.sam = None
            self.predictor = None
            self._current_image = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("SAM unloaded")
    
    def __repr__(self) -> str:
        status = "loaded" if self.sam is not None else "not loaded"
        return f"MedSAMTool(model_type={self.model_type}, status={status}, device={self.device})"
