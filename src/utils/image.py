"""
Image Processing Utilities
==========================

Functions for image conversion, cropping, and processing.
"""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL import Image


def image_to_base64(image: Union[str, Path, Image.Image, np.ndarray]) -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: PIL Image, numpy array, file path, or base64 string
        
    Returns:
        Base64 encoded string (without data:image prefix)
    """
    if isinstance(image, str):
        # Already base64
        if image.startswith("data:image"):
            return image.split(",", 1)[1]
        if len(image) > 500 and not Path(image).exists():
            return image  # Already base64 without prefix
        # File path
        image = Image.open(image)
    elif isinstance(image, Path):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        img_format = "PNG" if image.mode == "RGBA" else "JPEG"
        image.save(buffered, format=img_format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    raise ValueError(f"Unsupported image type: {type(image)}")


def base64_to_image(b64_string: str) -> Image.Image:
    """
    Convert base64 string to PIL Image.
    
    Args:
        b64_string: Base64 encoded string (with or without data:image prefix)
        
    Returns:
        PIL Image object
    """
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes))


def compute_image_hash(image: Union[str, Path, Image.Image, bytes]) -> str:
    """
    Compute hash for image identification.
    
    Args:
        image: PIL Image, file path, base64 string, or bytes
        
    Returns:
        12-character MD5 hash string
    """
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    elif isinstance(image, str):
        if len(image) > 500:
            img_bytes = base64.b64decode(image)
        else:
            with open(image, 'rb') as f:
                img_bytes = f.read()
    elif isinstance(image, Path):
        img_bytes = image.read_bytes()
    elif isinstance(image, bytes):
        img_bytes = image
    else:
        raise ValueError(f"Unsupported type: {type(image)}")
    
    return hashlib.md5(img_bytes).hexdigest()[:12]


def crop_image_by_box(
    image: Union[str, Image.Image],
    box: List[float],
    padding: int = 5
) -> Image.Image:
    """
    Crop image region by bounding box.
    
    Args:
        image: PIL Image or base64 string
        box: [x1, y1, x2, y2] bounding box coordinates
        padding: Padding around the box in pixels
        
    Returns:
        Cropped PIL Image
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    x1, y1, x2, y2 = map(int, box)
    w, h = image.size
    x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
    x2, y2 = min(w, x2 + padding), min(h, y2 + padding)
    
    return image.crop((x1, y1, x2, y2))


def smart_zoom_crop(
    image: Union[str, Image.Image],
    box: List[float],
    target_size: int = 336,
    context_ratio: float = 0.3
) -> Image.Image:
    """
    Smart zoom crop with context around the region of interest.
    
    Args:
        image: PIL Image or base64 string
        box: [x1, y1, x2, y2] bounding box
        target_size: Target size for output
        context_ratio: How much context to include (0.3 = 30% extra)
        
    Returns:
        Zoomed and cropped PIL Image
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    x1, y1, x2, y2 = box
    box_w, box_h = x2 - x1, y2 - y1
    img_w, img_h = image.size
    
    # Add context
    pad_x = box_w * context_ratio
    pad_y = box_h * context_ratio
    
    crop_x1 = max(0, x1 - pad_x)
    crop_y1 = max(0, y1 - pad_y)
    crop_x2 = min(img_w, x2 + pad_x)
    crop_y2 = min(img_h, y2 + pad_y)
    
    cropped = image.crop((int(crop_x1), int(crop_y1), int(crop_x2), int(crop_y2)))
    
    # Resize to target
    cropped.thumbnail((target_size, target_size), Image.Resampling.LANCZOS)
    
    return cropped


def crop_to_region(
    image: Image.Image,
    region_coords: List[float],
    padding: float = 0.05
) -> Tuple[Image.Image, Tuple[int, int, int, int]]:
    """
    Crop image to a normalized region.
    
    Args:
        image: PIL Image
        region_coords: [nx1, ny1, nx2, ny2] normalized coordinates (0-1)
        padding: Additional padding (0-1)
        
    Returns:
        Tuple of (cropped_image, (x1, y1, x2, y2) in original pixels)
    """
    w, h = image.size
    nx1, ny1, nx2, ny2 = region_coords
    
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


def resize_image(
    image: Union[str, Image.Image],
    max_size: int = 1024,
    maintain_aspect: bool = True
) -> Image.Image:
    """
    Resize image to max size while maintaining aspect ratio.
    
    Args:
        image: PIL Image or base64 string
        max_size: Maximum dimension
        maintain_aspect: Whether to maintain aspect ratio
        
    Returns:
        Resized PIL Image
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    if maintain_aspect:
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        return image
    else:
        return image.resize((max_size, max_size), Image.Resampling.LANCZOS)
