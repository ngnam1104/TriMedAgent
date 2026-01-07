"""
TriMedAgent Source Package
==========================

Modular Medical AI toolkit with Hybrid ReAct architecture.

Quick Start:
    from src import HybridReActOrchestrator
    
    agent = HybridReActOrchestrator()
    agent.load_tools()
    result = agent.process(image, "Find nodules in the lungs")

Package Structure:
    src/
    ├── types/          # Data models & types
    ├── utils/          # Image processing, visualization, constants
    ├── tools/          # AI tool wrappers
    ├── core/           # Orchestration & ReAct logic
    └── ui/             # Gradio web interface
"""

# =============================================================================
# Types - Data Models
# =============================================================================
from .types import (
    Strategy,
    TargetSize,
    StrategicPlan,
    DetectionResult,
    VerificationResult,
    AgentState,
    PipelineResult,
    RAGResult,
    TriageResult,
)

# =============================================================================
# Tools - AI Model Wrappers
# =============================================================================
from .tools import (
    BiomedCLIPTool,
    GroundingDINOTool,
    MedSAMTool,
    LLaVATool,
    MedicalRAG,
    RAGConfig,
)

# =============================================================================
# Core - Orchestration Logic
# =============================================================================
from .core import (
    StrategicPlanner,
    ImageZoomer,
    DetectionValidator,
    HybridReActOrchestrator,
    TriMedOrchestrator,
)

# =============================================================================
# Utils - Helper Functions (selective exports)
# =============================================================================
from .utils.image import (
    image_to_base64,
    base64_to_image,
    compute_image_hash,
    crop_image_by_box,
    resize_image,
)
from .utils.visualization import (
    draw_boxes_on_image,
    draw_masks_on_image,
    create_comparison_image,
)
from .utils.constants import (
    ANATOMICAL_REGIONS,
    BOX_SIZE_THRESHOLDS,
    MODALITY_LABELS,
    ABNORMALITY_LABELS,
)
from .utils.kaggle_config import (
    KaggleConfig,
    get_device_map,
    print_gpu_info,
    download_weights,
)

# =============================================================================
# UI - Gradio Web Interface
# =============================================================================
from .ui import (
    create_gradio_demo,
    launch_demo,
    ConversationState,
)

# =============================================================================
# Version & Exports
# =============================================================================
__version__ = "2.0.0"
__author__ = "TriMedAgent Team"

__all__ = [
    # Types
    "Strategy",
    "TargetSize",
    "StrategicPlan",
    "DetectionResult",
    "VerificationResult",
    "AgentState",
    "PipelineResult",
    "RAGResult",
    "TriageResult",
    # Tools
    "BiomedCLIPTool",
    "GroundingDINOTool",
    "MedSAMTool",
    "LLaVATool",
    "MedicalRAG",
    "RAGConfig",
    # Core
    "StrategicPlanner",
    "ImageZoomer",
    "DetectionValidator",
    "HybridReActOrchestrator",
    "TriMedOrchestrator",
    # Utils
    "image_to_base64",
    "base64_to_image",
    "compute_image_hash",
    "crop_image_by_box",
    "resize_image",
    "draw_boxes_on_image",
    "draw_masks_on_image",
    "create_comparison_image",
    "ANATOMICAL_REGIONS",
    "BOX_SIZE_THRESHOLDS",
    "MODALITY_LABELS",
    "ABNORMALITY_LABELS",
    # Kaggle Config
    "KaggleConfig",
    "get_device_map",
    "print_gpu_info",
    "download_weights",
    # UI / Gradio
    "create_gradio_demo",
    "launch_demo",
    "ConversationState",
]


def get_version():
    """Return package version."""
    return __version__


def quick_start():
    """Print quick start guide."""
    guide = """
╔══════════════════════════════════════════════════════════════╗
║              TriMedAgent v2.0 - Quick Start                  ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  from src import HybridReActOrchestrator                     ║
║  from PIL import Image                                       ║
║                                                              ║
║  # Initialize agent                                          ║
║  agent = HybridReActOrchestrator()                          ║
║  agent.load_tools()                                          ║
║                                                              ║
║  # Process medical image                                     ║
║  image = Image.open("chest_xray.png")                       ║
║  result = agent.process(                                     ║
║      image,                                                  ║
║      "Find any nodules in the right lung"                   ║
║  )                                                           ║
║                                                              ║
║  # Access results                                            ║
║  print(result.final_report)                                  ║
║  result.annotated_image.show()                               ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(guide)
