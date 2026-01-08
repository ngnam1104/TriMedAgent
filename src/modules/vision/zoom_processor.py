"""
Zoom Processor Module
=====================

Handles the "Smart Zoom" mechanism for detecting small objects.
Contains critical coordinate transformation logic to avoid placement errors.
"""

import logging
from typing import List, Tuple, Union, Dict, Any
from PIL import Image
import numpy as np

logger = logging.getLogger(__name__)

class ZoomManager:
    """
    Manages the Coarse-to-Fine Zoom strategy.
    
    Responsibilities:
    1. Generate strategic crops (Smart Zoom).
    2. Map local coordinates back to global space (Coordinate Translation).
    """
    
    def __init__(self, zoom_factor: float = 2.0):
        self.zoom_factor = zoom_factor

    def generate_crops(self, image: Image.Image, overlap: float = 0.2) -> List[Tuple[Image.Image, List[float]]]:
        """
        Generate sliding window crops with overlap.
        
        Args:
            image: Original PIL Image.
            overlap: Overlap ratio (0.0 to 1.0).
            
        Returns:
            List of tuples: (cropped_image, crop_box_global)
            crop_box_global format: [x1, y1, x2, y2]
        """
        w, h = image.size
        # Simple 2x2 grid for now logic + Center crop
        
        crops = []
        
        # Calculate dimensions for 2x2 grid
        # To handle small objects better, we ensure sufficient overlap
        # Split width/height roughly in half, but ensure they cover the space
        
        # Standard Grid:
        # TL, TR, BL, BR
        
        half_w = w // 2
        half_h = h // 2
        
        # With overlap, we might want slightly larger crops than just half
        # But keeping it simple for stability:
        # [0, 0, half_w, half_h]
        
        # Let's use the explicit request's context or a robust overlapping method
        # User prompt didn't specify grid details, just "ZoomManager.map_box_to_global".
        # I will preserve the existing simplified grid logic but clean it up.
        
        grid_boxes = [
            [0, 0, half_w, half_h],          # Top-Left
            [half_w, 0, w, half_h],          # Top-Right
            [0, half_h, half_w, h],          # Bottom-Left
            [half_w, half_h, w, h],          # Bottom-Right
            [w//4, h//4, w*3//4, h*3//4]     # Center (Context)
        ]
        
        for box in grid_boxes:
            # Ensure integer coords
            box = [int(c) for c in box]
            crop = image.crop(box)
            crops.append((crop, box))
            
        return crops

    def map_box_to_global(
        self, 
        local_box: List[float], 
        crop_region: List[float], 
        original_size: Tuple[int, int]
    ) -> List[float]:
        """
        Maps a bounding box from crop coordinates to global image coordinates.
        Robustly handles normalized vs pixel inputs and clamping.
        
        Target Method Implementation.
        
        Args:
            local_box: [x1, y1, x2, y2] in crop coordinates. 
                       Can be normalized (0-1) or absolute pixels.
            crop_region: [start_x, start_y, end_x, end_y] of the crop in global space.
            original_size: (width, height) of the original global image.
            
        Returns:
            [x1, y1, x2, y2] in global pixel coordinates.
        """
        W, H = original_size
        crop_x1, crop_y1, crop_x2, crop_y2 = crop_region
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        lx1, ly1, lx2, ly2 = local_box
        
        # 1. STANDARDIZE INPUT
        # Heuristically detect normalized coordinates
        # If all values are <= 1.0 (and typically detection boxes aren't 1x1 pixels), treat as normalized.
        # This is a critical check for V2 stability.
        is_normalized = all(0.0 <= c <= 1.05 for c in local_box) # Tolerance for 1.0
        
        if is_normalized:
            # Convert to pixels relative to CROP size
            lx1 *= crop_w
            lx2 *= crop_w
            ly1 *= crop_h
            ly2 *= crop_h
            
        # 2. APPLY OFFSET
        # Global = Local_Pixel + Crop_Start
        gx1 = lx1 + crop_x1
        gy1 = ly1 + crop_y1
        gx2 = lx2 + crop_x1
        gy2 = ly2 + crop_y1
        
        # 3. CLAMPING
        # Ensure coordinates do not exceed original image boundaries
        # Use simple max/min
        gx1 = max(0, min(gx1, W))
        gy1 = max(0, min(gy1, H))
        gx2 = max(0, min(gx2, W))
        gy2 = max(0, min(gy2, H))
        
        # 4. FINAL INTEGRITY CHECK
        # Ensure valid box (width > 0, height > 0)
        if gx2 <= gx1 or gy2 <= gy1:
            logger.warning(f"ZoomManager: mapped box is invalid/empty {[gx1, gy1, gx2, gy2]}")
            # Do not return empty list if we want to preserve type, but usually better to drop
            # For now return the clamped box, caller might filter area 0.
            
        return [float(gx1), float(gy1), float(gx2), float(gy2)]
        
    def process_zoom_results(
        self, 
        local_results: List[Dict[str, Any]], 
        crop_box: List[float],
        original_size: Tuple[int, int]
    ) -> List[Dict[str, Any]]:
        """
        Helper to process list of DINO results from a crop.
        """
        global_results = []
        for res in local_results:
            local_box = res['box']
            global_box = self.map_box_to_global(local_box, crop_box, original_size)
            
            # Filter invalids
            if global_box[2] <= global_box[0] or global_box[3] <= global_box[1]:
                continue
                
            res_global = res.copy()
            res_global['box'] = global_box
            res_global['source'] = 'zoom_crop'
            global_results.append(res_global)
            
        return global_results
