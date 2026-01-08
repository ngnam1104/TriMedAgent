"""
Math Utils
==========

Coordinate mapping and geometric functions.
"""

from typing import List, Tuple

def map_box_to_global(
    box_local: List[float], 
    crop_box: List[float]
) -> List[float]:
    """
    Map bounding box from cropped image coordinates to global image coordinates.
    
    Formula:
    x_global = x_local + crop_offset_x
    y_global = y_local + crop_offset_y
    
    Args:
        box_local: [x1, y1, x2, y2] in cropped image frame
        crop_box: [crop_x1, crop_y1, crop_x2, crop_y2] of the crop in global frame
        
    Returns:
        [x1, y1, x2, y2] in global frame
    """
    l_x1, l_y1, l_x2, l_y2 = box_local
    c_x1, c_y1, _, _ = crop_box
    
    return [
        l_x1 + c_x1,
        l_y1 + c_y1,
        l_x2 + c_x1,
        l_y2 + c_y1
    ]

def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """Calculate Intersection over Union (IoU)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    return intersection / (area1 + area2 - intersection + 1e-6)

def get_box_area_ratio(box: List[float], image_size: Tuple[int, int]) -> float:
    """Calculate ratio of box area to image area."""
    box_area = (box[2] - box[0]) * (box[3] - box[1])
    img_area = image_size[0] * image_size[1]
    return box_area / (img_area + 1e-6)
