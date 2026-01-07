"""
Utility modules for TriMedAgent.
"""

from .image import (
    image_to_base64,
    base64_to_image,
    compute_image_hash,
    crop_image_by_box,
    smart_zoom_crop,
)

from .visualization import (
    draw_boxes_on_image,
    draw_masks_on_image,
    draw_points_on_image,
    create_comparison_image,
    COLORS,
)

from .constants import (
    ANATOMICAL_REGIONS,
    BOX_SIZE_THRESHOLDS,
    MODALITY_LABELS,
    ABNORMALITY_LABELS,
)

from .kaggle_config import (
    KaggleConfig,
    get_device_map,
    get_weight_paths,
    get_weights_dir,
    download_weights,
    print_gpu_info,
    is_kaggle,
    is_colab,
    get_environment,
    HF_MODELS,
    MODEL_URLS,
)

__all__ = [
    # Image
    "image_to_base64",
    "base64_to_image",
    "compute_image_hash",
    "crop_image_by_box",
    "smart_zoom_crop",
    # Visualization
    "draw_boxes_on_image",
    "draw_masks_on_image",
    "draw_points_on_image",
    "create_comparison_image",
    "COLORS",
    # Constants
    "ANATOMICAL_REGIONS",
    "BOX_SIZE_THRESHOLDS",
    "MODALITY_LABELS",
    "ABNORMALITY_LABELS",
    # Kaggle Config
    "KaggleConfig",
    "get_device_map",
    "get_weight_paths",
    "get_weights_dir",
    "download_weights",
    "print_gpu_info",
    "is_kaggle",
    "is_colab",
    "get_environment",
    "HF_MODELS",
    "MODEL_URLS",
]

