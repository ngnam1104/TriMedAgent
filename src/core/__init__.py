"""
Core Orchestration Logic
========================

ReAct Agent components for medical image analysis.

Components:
- StrategicPlanner: LLaVA-based planning
- ImageZoomer: Coarse-to-Fine cropping
- DetectionValidator: Self-correction
- HybridReActOrchestrator: Main agent

Usage:
    from src.core import HybridReActOrchestrator
    
    orchestrator = HybridReActOrchestrator()
    orchestrator.load_tools()
    result = orchestrator.process(image, query)
"""

from .planner import StrategicPlanner
from .zoomer import ImageZoomer
from .validator import DetectionValidator
from .orchestrator import HybridReActOrchestrator, TriMedOrchestrator

__all__ = [
    "StrategicPlanner",
    "ImageZoomer",
    "DetectionValidator",
    "HybridReActOrchestrator",
    "TriMedOrchestrator",
]
