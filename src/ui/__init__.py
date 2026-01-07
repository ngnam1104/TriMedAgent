"""
UI Module for TriMedAgent
=========================

Gradio-based web interface for medical image analysis.

Usage:
    from src.ui import launch_demo, create_gradio_demo
    
    # With orchestrator
    from src import HybridReActOrchestrator
    orchestrator = HybridReActOrchestrator()
    orchestrator.load_tools()
    launch_demo(orchestrator, share=True)
    
    # Or standalone
    launch_demo()
"""

from .gradio_app import (
    create_gradio_demo,
    launch_demo,
    ConversationState,
    plot_boxes_on_image,
    plot_masks_on_image,
)

__all__ = [
    "create_gradio_demo",
    "launch_demo",
    "ConversationState",
    "plot_boxes_on_image",
    "plot_masks_on_image",
]
