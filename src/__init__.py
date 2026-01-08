"""
TriMedAgent Source Package
==========================

Modular Medical AI toolkit with Hybrid ReAct architecture.
Refactored for V2 Modular Architecture (2025).

Exposes core orchestrator and individual modules.
"""

from .core.orchestrator import TriMedOrchestrator
from .types.models import PipelineResult, StrategicPlan
from .ui.gradio_app import launch_demo

# Module Exports for direct access
from .modules.brain.llava_brain import LLaVABrain
from .modules.vision.biomed_clip import BiomedCLIPTool
from .modules.vision.dino_detector import GroundingDINO
from .modules.vision.zoom_processor import ZoomManager
from .modules.segmentation.medsam_wrapper import MedSAMWrapper
from .modules.knowledge.rag_engine import MedicalRAG

__all__ = [
    "TriMedOrchestrator",
    "PipelineResult",
    "StrategicPlan",
    "launch_demo",
    "LLaVABrain",
    "BiomedCLIPTool",
    "GroundingDINO",
    "ZoomManager",
    "MedSAMWrapper",
    "MedicalRAG"
]
