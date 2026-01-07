"""
Image Zoomer - Coarse-to-Fine Cropping
======================================

Handles image cropping and coordinate mapping for precise detection.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

from PIL import Image

from ..utils.constants import ANATOMICAL_REGIONS

logger = logging.getLogger(__name__)


class ImageZoomer:
    """
    Handles image cropping and coordinate mapping for Coarse-to-Fine detection.
    
    Solves the "box too large" problem by:
    1. Cropping to anatomical region
    2. Running detection on cropped image
    3. Mapping coordinates back to original image
    """
    
    def __init__(self):
        self.regions = ANATOMICAL_REGIONS
    
    def crop_to_region(
        self,
        image: Image.Image,
        region_name: str,
        padding: float = 0.05
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Crop image to anatomical region.
        
        Args:
            image: Original image
            region_name: Name of anatomical region
            padding: Additional padding around region (0-1)
            
        Returns:
            Tuple of (cropped_image, (x1, y1, x2, y2) in original coords)
        """
        if region_name not in self.regions:
            logger.warning(f"Unknown region: {region_name}, using full image")
            return image, (0, 0, image.width, image.height)
        
        w, h = image.size
        nx1, ny1, nx2, ny2 = self.regions[region_name]
        
        # Add padding
        pad_x = (nx2 - nx1) * padding
        pad_y = (ny2 - ny1) * padding
        nx1 = max(0, nx1 - pad_x)
        ny1 = max(0, ny1 - pad_y)
        nx2 = min(1, nx2 + pad_x)
        ny2 = min(1, ny2 + pad_y)
        
        # Convert to pixel coordinates
        x1, y1 = int(nx1 * w), int(ny1 * h)
        x2, y2 = int(nx2 * w), int(ny2 * h)
        
        cropped = image.crop((x1, y1, x2, y2))
        
        return cropped, (x1, y1, x2, y2)
    
    def map_boxes_to_original(
        self,
        boxes: List[List[float]],
        crop_region: Tuple[int, int, int, int],
        original_size: Tuple[int, int]
    ) -> List[List[float]]:
        """
        Map bounding boxes from cropped image back to original coordinates.
        
        Args:
            boxes: Boxes in cropped image coordinates
            crop_region: (x1, y1, x2, y2) of crop in original image
            original_size: (width, height) of original image
            
        Returns:
            Boxes in original image coordinates
        """
        if not boxes:
            return []
        
        cx1, cy1, cx2, cy2 = crop_region
        orig_w, orig_h = original_size
        
        mapped_boxes = []
        for box in boxes:
            bx1, by1, bx2, by2 = box
            
            # Map to original coordinates
            new_x1 = cx1 + bx1
            new_y1 = cy1 + by1
            new_x2 = cx1 + bx2
            new_y2 = cy1 + by2
            
            # Clamp to image bounds
            new_x1 = max(0, min(orig_w, new_x1))
            new_y1 = max(0, min(orig_h, new_y1))
            new_x2 = max(0, min(orig_w, new_x2))
            new_y2 = max(0, min(orig_h, new_y2))
            
            mapped_boxes.append([new_x1, new_y1, new_x2, new_y2])
        
        return mapped_boxes
    
    def create_grid_crops(
        self,
        image: Image.Image,
        grid_size: Tuple[int, int] = (2, 2),
        overlap: float = 0.1
    ) -> List[Tuple[Image.Image, Tuple[int, int, int, int]]]:
        """
        Create grid of overlapping crops for sliding window detection.
        
        Args:
            image: Original image
            grid_size: (rows, cols) grid division
            overlap: Overlap between adjacent crops (0-1)
            
        Returns:
            List of (cropped_image, crop_region) tuples
        """
        w, h = image.size
        rows, cols = grid_size
        
        cell_w = w / cols
        cell_h = h / rows
        overlap_w = cell_w * overlap
        overlap_h = cell_h * overlap
        
        crops = []
        for r in range(rows):
            for c in range(cols):
                x1 = max(0, int(c * cell_w - overlap_w))
                y1 = max(0, int(r * cell_h - overlap_h))
                x2 = min(w, int((c + 1) * cell_w + overlap_w))
                y2 = min(h, int((r + 1) * cell_h + overlap_h))
                
                cropped = image.crop((x1, y1, x2, y2))
                crops.append((cropped, (x1, y1, x2, y2)))
        
        return crops
    
    def smart_zoom(
        self,
        image: Image.Image,
        center: Tuple[float, float],
        zoom_factor: float = 2.0,
        min_size: int = 224
    ) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
        """
        Smart zoom to a specific point with adaptive sizing.
        
        Args:
            image: Original image
            center: (x, y) center point (normalized 0-1)
            zoom_factor: How much to zoom (2.0 = half size crop)
            min_size: Minimum crop size in pixels
            
        Returns:
            Tuple of (cropped_image, crop_region)
        """
        w, h = image.size
        cx, cy = center[0] * w, center[1] * h
        
        # Calculate crop size
        crop_w = max(min_size, w / zoom_factor)
        crop_h = max(min_size, h / zoom_factor)
        
        # Calculate crop region centered on point
        x1 = int(max(0, cx - crop_w / 2))
        y1 = int(max(0, cy - crop_h / 2))
        x2 = int(min(w, x1 + crop_w))
        y2 = int(min(h, y1 + crop_h))
        
        # Adjust if hit boundary
        if x2 - x1 < crop_w:
            x1 = max(0, x2 - int(crop_w))
        if y2 - y1 < crop_h:
            y1 = max(0, y2 - int(crop_h))
        
        cropped = image.crop((x1, y1, x2, y2))
        
        return cropped, (x1, y1, x2, y2)
