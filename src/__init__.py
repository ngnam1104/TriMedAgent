"""
TriMedAgent Source Package
==========================

Centralized business logic for TriMedAgent medical AI pipeline.
"""

from .trimed_orchestrator import (
    # Main Orchestrator
    TriMedOrchestrator,
    
    # Data Classes
    TriageResult,
    DetectionResult,
    GatekeeperResult,
    SegmentationResult,
    PipelineResult,
    
    # Utilities
    image_to_base64,
    base64_to_image,
    crop_image_by_box,
    
    # Configuration
    WORKER_MAP,
    DEFAULT_CONFIG,
)

__all__ = [
    # Main Class
    "TriMedOrchestrator",
    
    # Data Classes
    "TriageResult",
    "DetectionResult",
    "GatekeeperResult",
    "SegmentationResult",
    "PipelineResult",
    
    # Utilities
    "image_to_base64",
    "base64_to_image",
    "crop_image_by_box",
    
    # Configuration
    "WORKER_MAP",
    "DEFAULT_CONFIG",
]

__version__ = "1.0.0"
