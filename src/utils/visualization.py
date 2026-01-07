"""
Visualization Utilities
=======================

Functions for drawing boxes, masks, and annotations on images.
"""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .image import base64_to_image


# Default colors for visualization (RGB)
COLORS = [
    (255, 0, 0),      # Red
    (0, 255, 0),      # Green
    (0, 0, 255),      # Blue
    (255, 255, 0),    # Yellow
    (255, 0, 255),    # Magenta
    (0, 255, 255),    # Cyan
    (255, 128, 0),    # Orange
    (128, 0, 255),    # Purple
    (0, 128, 255),    # Sky Blue
    (255, 128, 128),  # Light Red
]


def draw_boxes_on_image(
    image: Union[str, Image.Image],
    boxes: List[List[float]],
    labels: Optional[List[str]] = None,
    scores: Optional[List[float]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    line_width: int = 3,
    font_size: int = 12
) -> Image.Image:
    """
    Draw bounding boxes on image.
    
    Args:
        image: PIL Image or base64 string
        boxes: List of [x1, y1, x2, y2] bounding boxes
        labels: Optional labels for each box
        scores: Optional confidence scores for each box
        colors: Optional colors for each box (RGB tuples)
        line_width: Width of bounding box lines
        font_size: Size of label text
        
    Returns:
        PIL Image with boxes drawn
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    if not boxes:
        return image.copy()
    
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    
    if colors is None:
        colors = COLORS
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        color = colors[i % len(colors)]
        
        # Draw rectangle
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_width)
        
        # Build label text
        label_parts = []
        if labels and i < len(labels):
            label_parts.append(labels[i])
        if scores and i < len(scores):
            label_parts.append(f"{scores[i]:.0%}")
        
        if label_parts:
            label = " ".join(label_parts)
            # Background for text
            text_bbox = draw.textbbox((x1, y1 - 18), label)
            draw.rectangle(text_bbox, fill=color)
            draw.text((x1, y1 - 18), label, fill=(255, 255, 255))
    
    return image


def draw_masks_on_image(
    image: Union[str, Image.Image],
    masks: List[np.ndarray],
    colors: Optional[List[Tuple[int, int, int]]] = None,
    alpha: float = 0.4,
    draw_contours: bool = True,
    contour_width: int = 2
) -> Image.Image:
    """
    Draw segmentation masks on image.
    
    Args:
        image: PIL Image or base64 string
        masks: List of boolean/binary numpy arrays
        colors: Optional colors for each mask (RGB tuples)
        alpha: Transparency of masks (0-1)
        draw_contours: Whether to draw mask contours
        contour_width: Width of contour lines
        
    Returns:
        PIL Image with masks overlaid
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    if not masks:
        return image.copy()
    
    image = image.copy().convert("RGB")
    img_array = np.array(image)
    
    if colors is None:
        colors = COLORS
    
    for i, mask in enumerate(masks):
        if mask is None:
            continue
        
        # Ensure mask is 2D boolean
        if mask.ndim == 3:
            mask = mask[:, :, 0]
        mask = mask.astype(bool)
        
        # Skip if mask is wrong size
        if mask.shape != img_array.shape[:2]:
            continue
            
        color = colors[i % len(colors)]
        
        # Create colored overlay
        overlay = np.zeros_like(img_array)
        overlay[mask] = color
        
        # Blend with original
        img_array = np.where(
            mask[:, :, np.newaxis],
            (1 - alpha) * img_array + alpha * overlay,
            img_array
        ).astype(np.uint8)
    
    result = Image.fromarray(img_array)
    
    # Draw contours
    if draw_contours:
        from scipy import ndimage
        draw = ImageDraw.Draw(result)
        
        for i, mask in enumerate(masks):
            if mask is None:
                continue
            if mask.ndim == 3:
                mask = mask[:, :, 0]
            mask = mask.astype(bool)
            
            color = colors[i % len(colors)]
            
            # Find contour using gradient
            gradient = ndimage.morphological_gradient(mask.astype(np.uint8), size=3)
            contour_points = np.argwhere(gradient > 0)
            
            for y, x in contour_points:
                draw.ellipse(
                    [x - contour_width//2, y - contour_width//2, 
                     x + contour_width//2, y + contour_width//2],
                    fill=color
                )
    
    return result


def draw_points_on_image(
    image: Union[str, Image.Image],
    points: List[Tuple[int, int]],
    labels: Optional[List[str]] = None,
    colors: Optional[List[Tuple[int, int, int]]] = None,
    radius: int = 5
) -> Image.Image:
    """
    Draw points on image.
    
    Args:
        image: PIL Image or base64 string
        points: List of (x, y) coordinates
        labels: Optional labels for each point
        colors: Optional colors for each point
        radius: Radius of point circles
        
    Returns:
        PIL Image with points drawn
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    image = image.copy().convert("RGB")
    draw = ImageDraw.Draw(image)
    
    if colors is None:
        colors = COLORS
    
    for i, (x, y) in enumerate(points):
        color = colors[i % len(colors)]
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255)
        )
        
        if labels and i < len(labels):
            draw.text((x + radius + 2, y - radius), labels[i], fill=color)
    
    return image


def create_comparison_image(
    images: List[Image.Image],
    labels: Optional[List[str]] = None,
    direction: str = "horizontal",
    padding: int = 10,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create a comparison image by concatenating multiple images.
    
    Args:
        images: List of PIL Images
        labels: Optional labels for each image
        direction: "horizontal" or "vertical"
        padding: Padding between images
        background_color: Background color
        
    Returns:
        Concatenated PIL Image
    """
    if not images:
        raise ValueError("No images provided")
    
    if direction == "horizontal":
        # Resize all to same height
        target_height = min(img.height for img in images)
        resized = []
        for img in images:
            ratio = target_height / img.height
            new_width = int(img.width * ratio)
            resized.append(img.resize((new_width, target_height)))
        
        total_width = sum(img.width for img in resized) + padding * (len(resized) - 1)
        result = Image.new("RGB", (total_width, target_height), background_color)
        
        x_offset = 0
        for img in resized:
            result.paste(img, (x_offset, 0))
            x_offset += img.width + padding
    else:
        # Resize all to same width
        target_width = min(img.width for img in images)
        resized = []
        for img in images:
            ratio = target_width / img.width
            new_height = int(img.height * ratio)
            resized.append(img.resize((target_width, new_height)))
        
        total_height = sum(img.height for img in resized) + padding * (len(resized) - 1)
        result = Image.new("RGB", (target_width, total_height), background_color)
        
        y_offset = 0
        for img in resized:
            result.paste(img, (0, y_offset))
            y_offset += img.height + padding
    
    # Add labels if provided
    if labels:
        draw = ImageDraw.Draw(result)
        x_offset = 5
        y_offset = 5
        for i, label in enumerate(labels):
            if direction == "horizontal":
                x = x_offset + sum(resized[j].width + padding for j in range(i))
            else:
                x = 5
            
            if direction == "vertical":
                y = y_offset + sum(resized[j].height + padding for j in range(i))
            else:
                y = 5
            
            draw.text((x, y), label, fill=(0, 0, 0))
    
    return result


def create_grid_image(
    images: List[Image.Image],
    grid_size: Tuple[int, int],
    cell_size: Optional[Tuple[int, int]] = None,
    padding: int = 5,
    background_color: Tuple[int, int, int] = (255, 255, 255)
) -> Image.Image:
    """
    Create a grid of images.
    
    Args:
        images: List of PIL Images
        grid_size: (rows, cols)
        cell_size: Optional (width, height) for each cell
        padding: Padding between cells
        background_color: Background color
        
    Returns:
        Grid PIL Image
    """
    rows, cols = grid_size
    
    if cell_size is None:
        # Use max dimensions
        max_w = max(img.width for img in images)
        max_h = max(img.height for img in images)
        cell_size = (max_w, max_h)
    
    cell_w, cell_h = cell_size
    total_w = cols * cell_w + (cols + 1) * padding
    total_h = rows * cell_h + (rows + 1) * padding
    
    result = Image.new("RGB", (total_w, total_h), background_color)
    
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        
        row = i // cols
        col = i % cols
        
        # Resize to fit cell
        img_resized = img.copy()
        img_resized.thumbnail((cell_w, cell_h))
        
        # Center in cell
        x = padding + col * (cell_w + padding) + (cell_w - img_resized.width) // 2
        y = padding + row * (cell_h + padding) + (cell_h - img_resized.height) // 2
        
        result.paste(img_resized, (x, y))
    
    return result
