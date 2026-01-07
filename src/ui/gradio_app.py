"""
Gradio Web Interface for TriMedAgent
====================================

Beautiful medical AI chatbot interface with:
- Image upload with sketch/mask support
- Multi-turn conversation
- Real-time detection visualization
- Responsive design

Usage:
    from src.ui import launch_demo
    launch_demo()
"""

from __future__ import annotations

import base64
import copy
import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass, field
from functools import partial
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Round helper
R = partial(round, ndigits=2)


# =============================================================================
# Conversation State Management
# =============================================================================

@dataclass
class ConversationState:
    """Manages multi-turn conversation state."""
    
    messages: List[Tuple[str, Any]] = field(default_factory=list)
    images: List[Image.Image] = field(default_factory=list)
    masks: List[Any] = field(default_factory=list)
    current_image: Optional[Image.Image] = None
    current_mask: Optional[Any] = None
    detection_results: Dict[str, Any] = field(default_factory=dict)
    
    def add_user_message(self, text: str, image: Optional[Image.Image] = None):
        """Add user message."""
        if image is not None:
            self.current_image = image
            self.images.append(image)
        self.messages.append(("user", text))
    
    def add_assistant_message(self, text: str, image: Optional[Image.Image] = None):
        """Add assistant response."""
        self.messages.append(("assistant", (text, image)))
    
    def get_history(self) -> List[Tuple[str, str]]:
        """Get chat history for Gradio Chatbot."""
        history = []
        for role, content in self.messages:
            if role == "user":
                history.append((content, None))
            else:
                if isinstance(content, tuple):
                    text, img = content
                    history.append((None, text))
                else:
                    history.append((None, content))
        return history
    
    def clear(self):
        """Clear conversation."""
        self.messages = []
        self.images = []
        self.masks = []
        self.current_image = None
        self.current_mask = None
        self.detection_results = {}


# =============================================================================
# Image Utilities
# =============================================================================

def encode_image_base64(image: Image.Image) -> str:
    """Encode PIL Image to base64."""
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode()


def decode_base64_image(b64_string: str) -> Image.Image:
    """Decode base64 string to PIL Image."""
    return Image.open(BytesIO(base64.b64decode(b64_string)))


def get_mask_bbox(mask_img: Image.Image) -> Optional[List[float]]:
    """Extract bounding box from mask image."""
    mask = np.array(mask_img)
    if len(mask.shape) > 2:
        mask = mask[..., 0]
    
    if mask.sum() == 0:
        return None
    
    coords = np.argwhere(mask > 0)
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    
    h, w = mask.shape[:2]
    return [R(x0/w), R(y0/h), R(x1/w), R(y1/h)]


def plot_boxes_on_image(image: Image.Image, boxes: List, scores: List = None, labels: List = None) -> Image.Image:
    """Draw bounding boxes on image."""
    import cv2
    
    img_np = np.array(image)
    h, w = img_np.shape[:2]
    
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        
        # Color based on index
        color = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)][i % 5]
        
        # Draw box
        cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = ""
        if scores and i < len(scores):
            label = f"{scores[i]:.2f}"
        if labels and i < len(labels):
            label = f"{labels[i]}: {label}" if label else labels[i]
        
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_np, (x1, y1-th-8), (x1+tw+4, y1), color, -1)
            cv2.putText(img_np, label, (x1+2, y1-4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
    
    return Image.fromarray(img_np)


def plot_masks_on_image(image: Image.Image, masks: List[np.ndarray], alpha: float = 0.4) -> Image.Image:
    """Draw masks on image."""
    img_np = np.array(image).astype(np.float32)
    
    colors = [
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255],  # Magenta
    ]
    
    for i, mask in enumerate(masks):
        color = colors[i % len(colors)]
        mask_bool = mask > 0
        img_np[mask_bool] = img_np[mask_bool] * (1 - alpha) + np.array(color) * alpha
    
    return Image.fromarray(img_np.astype(np.uint8))


# =============================================================================
# Gradio App Builder
# =============================================================================

def create_gradio_demo(orchestrator=None):
    """
    Create Gradio demo interface.
    
    Args:
        orchestrator: HybridReActOrchestrator instance (optional, will load if None)
        
    Returns:
        Gradio Blocks demo
    """
    try:
        import gradio as gr
    except ImportError:
        raise ImportError("Gradio not installed. Run: pip install gradio")
    
    # State
    state = ConversationState()
    
    # CSS for beautiful UI
    custom_css = """
    .container { max-width: 1200px; margin: auto; }
    .header { text-align: center; margin-bottom: 20px; }
    .header h1 { color: #2c3e50; font-size: 2.5em; }
    .header p { color: #7f8c8d; }
    .chat-container { border-radius: 10px; }
    .image-preview { border-radius: 10px; border: 2px dashed #3498db; }
    .result-image { border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
    .btn-primary { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); }
    .status-box { padding: 10px; border-radius: 8px; background: #f8f9fa; }
    """
    
    # Title
    title_md = """
    # 🏥 TriMedAgent - Medical AI Assistant
    
    **Hybrid ReAct Agent** for Medical Image Analysis
    
    - 🔬 **BiomedCLIP**: Image triage & classification
    - 🎯 **Grounding DINO**: Object detection
    - 🎭 **MedSAM**: Segmentation
    - 🧠 **LLaVA-Med**: Visual reasoning
    
    ---
    """
    
    def process_message(
        message: str,
        image_dict: Optional[Dict],
        history: List,
        temperature: float,
        max_tokens: int,
        enable_detection: bool,
        enable_segmentation: bool,
    ):
        """Process user message and generate response."""
        if orchestrator is None:
            yield history + [(message, "⚠️ Orchestrator not loaded. Please initialize first.")], None
            return
        
        # Extract image and mask
        image = None
        mask_bbox = None
        
        if image_dict is not None:
            if isinstance(image_dict, dict):
                image = image_dict.get('image') or image_dict.get('background')
                mask = image_dict.get('mask') or image_dict.get('layers', [None])[0]
                if mask is not None:
                    mask_bbox = get_mask_bbox(mask)
            else:
                image = image_dict
        
        if image is None and len(state.images) > 0:
            image = state.images[-1]
        
        if image is None:
            yield history + [(message, "⚠️ Please upload an image first.")], None
            return
        
        # Add mask info to query if present
        query = message
        if mask_bbox:
            query += f"\n[User drew box: {mask_bbox}]"
        
        # Update history
        history = history + [(message, "🔄 Processing...")]
        yield history, None
        
        try:
            # Run pipeline
            result = orchestrator.process(image, query, force_segmentation=enable_segmentation)
            
            # Build response
            response_parts = []
            
            # Triage info
            if result.triage_modality:
                response_parts.append(f"📋 **Modality**: {result.triage_modality} ({result.triage_confidence:.0%})")
            
            # Detection info
            if result.verified_boxes:
                response_parts.append(f"🎯 **Detected**: {len(result.verified_boxes)} region(s)")
            
            # Strategic plan
            if result.strategic_plan:
                plan = result.strategic_plan
                response_parts.append(f"📍 **Strategy**: {plan.strategy.value} → {plan.target_object}")
            
            # LLaVA analysis
            if result.llava_analysis:
                response_parts.append(f"\n🧠 **Analysis**:\n{result.llava_analysis}")
            
            # Final report
            if result.final_report:
                response_parts.append(f"\n📝 **Report**:\n{result.final_report}")
            
            response = "\n\n".join(response_parts) if response_parts else "Processing complete."
            
            # Output image
            output_image = result.annotated_image if result.annotated_image else image
            
            # Update state
            state.add_user_message(message, image)
            state.add_assistant_message(response, output_image)
            state.detection_results = {
                'boxes': result.verified_boxes,
                'masks': result.masks,
            }
            
            history[-1] = (message, response)
            yield history, output_image
            
        except Exception as e:
            logger.error(f"Processing error: {e}")
            history[-1] = (message, f"❌ Error: {str(e)}")
            yield history, None
    
    def clear_conversation():
        """Clear conversation and reset state."""
        state.clear()
        return [], None, None
    
    def load_example(example_url: str, example_query: str):
        """Load example image and query."""
        import requests
        from io import BytesIO
        
        try:
            response = requests.get(example_url, timeout=10)
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return image, example_query
        except:
            return None, example_query
    
    # Build Gradio interface
    with gr.Blocks(css=custom_css, title="TriMedAgent", theme=gr.themes.Soft()) as demo:
        
        gr.Markdown(title_md)
        
        with gr.Row():
            # Left column - Image input
            with gr.Column(scale=4):
                image_input = gr.Image(
                    label="📷 Upload Medical Image",
                    type="pil",
                    height=400,
                    elem_classes="image-preview"
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    
                with gr.Accordion("⚙️ Settings", open=False):
                    temperature = gr.Slider(0, 1, value=0.2, step=0.1, label="Temperature")
                    max_tokens = gr.Slider(128, 1024, value=512, step=64, label="Max Tokens")
                    enable_detection = gr.Checkbox(value=True, label="Enable Detection")
                    enable_segmentation = gr.Checkbox(value=True, label="Enable Segmentation")
                
                with gr.Accordion("📚 Examples", open=False):
                    gr.Examples(
                        examples=[
                            ["images/example_chest.png", "Find any nodules in the lungs"],
                            ["images/example_mri.png", "Find any tumors in the brain"],
                        ],
                        inputs=[image_input, gr.Textbox(visible=False)],
                        label="Click to load example"
                    )
            
            # Right column - Chat
            with gr.Column(scale=6):
                chatbot = gr.Chatbot(
                    label="💬 Chat",
                    height=350,
                    elem_classes="chat-container",
                    avatar_images=(None, "🤖")
                )
                
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="Ask about the medical image...",
                        label="",
                        scale=8,
                        container=False
                    )
                    submit_btn = gr.Button("🚀 Send", variant="primary", scale=1)
                
                output_image = gr.Image(
                    label="📊 Analysis Result",
                    type="pil",
                    height=300,
                    elem_classes="result-image"
                )
        
        # Footer
        gr.Markdown("""
        ---
        **⚠️ Disclaimer**: This tool is for research/educational purposes only. 
        Not intended for clinical diagnosis. Always consult healthcare professionals.
        
        Built with ❤️ using [TriMedAgent](https://github.com/ngnam1104/TriMedAgent)
        """)
        
        # Event handlers
        submit_btn.click(
            process_message,
            inputs=[text_input, image_input, chatbot, temperature, max_tokens, enable_detection, enable_segmentation],
            outputs=[chatbot, output_image]
        )
        
        text_input.submit(
            process_message,
            inputs=[text_input, image_input, chatbot, temperature, max_tokens, enable_detection, enable_segmentation],
            outputs=[chatbot, output_image]
        )
        
        clear_btn.click(
            clear_conversation,
            outputs=[chatbot, output_image, image_input]
        )
    
    return demo


def launch_demo(
    orchestrator=None,
    share: bool = False,
    port: int = 7860,
    debug: bool = False
):
    """
    Launch Gradio demo.
    
    Args:
        orchestrator: HybridReActOrchestrator instance
        share: Create public link
        port: Port number
        debug: Enable debug mode
    """
    demo = create_gradio_demo(orchestrator)
    
    # Handle Colab/Kaggle
    try:
        import nest_asyncio
        nest_asyncio.apply()
    except ImportError:
        pass
    
    demo.queue().launch(
        share=share,
        server_port=port,
        debug=debug,
        show_error=True
    )


# =============================================================================
# Standalone launcher
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Port number")
    parser.add_argument("--debug", action="store_true", help="Debug mode")
    args = parser.parse_args()
    
    print("🚀 Launching TriMedAgent Demo...")
    print("⚠️ Note: Load orchestrator first for full functionality")
    
    launch_demo(share=args.share, port=args.port, debug=args.debug)
