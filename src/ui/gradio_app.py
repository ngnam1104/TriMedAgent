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
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get chat history for Gradio Chatbot (OpenAI-style messages format)."""
        history = []
        for role, content in self.messages:
            if role == "user":
                history.append({"role": "user", "content": content})
            else:
                if isinstance(content, tuple):
                    text, img = content
                    history.append({"role": "assistant", "content": text})
                else:
                    history.append({"role": "assistant", "content": content})
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
    
    # Premium CSS for beautiful medical UI
    custom_css = """
    /* === Global Styles === */
    :root {
        --primary-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        --secondary-gradient: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        --medical-blue: #0077b6;
        --medical-teal: #00b4d8;
        --success-green: #2ecc71;
        --warning-orange: #f39c12;
        --error-red: #e74c3c;
        --bg-dark: #1a1a2e;
        --bg-card: #16213e;
        --text-light: #eef2f7;
        --border-radius: 16px;
    }
    
    .gradio-container {
        background: linear-gradient(180deg, #0f0c29 0%, #302b63 50%, #24243e 100%) !important;
        font-family: 'Inter', 'Segoe UI', sans-serif !important;
    }
    
    /* === Header === */
    .header-banner {
        background: var(--primary-gradient);
        padding: 24px 32px;
        border-radius: var(--border-radius);
        margin-bottom: 24px;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.3);
    }
    
    .header-banner h1 {
        color: white !important;
        font-size: 2.5rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .header-banner p {
        color: rgba(255,255,255,0.9) !important;
        font-size: 1.1rem !important;
        margin-top: 8px !important;
    }
    
    /* === Feature Pills === */
    .feature-pills {
        display: flex;
        flex-wrap: wrap;
        gap: 12px;
        margin-top: 16px;
    }
    
    .feature-pill {
        background: rgba(255,255,255,0.15);
        backdrop-filter: blur(10px);
        padding: 8px 16px;
        border-radius: 20px;
        color: white;
        font-size: 0.9rem;
        border: 1px solid rgba(255,255,255,0.2);
    }
    
    /* === Cards === */
    .glass-card {
        background: rgba(255,255,255,0.05) !important;
        backdrop-filter: blur(20px) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: var(--border-radius) !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.2) !important;
    }
    
    /* === Image Upload === */
    .image-upload-area {
        border: 2px dashed var(--medical-teal) !important;
        border-radius: var(--border-radius) !important;
        background: rgba(0, 180, 216, 0.05) !important;
        transition: all 0.3s ease !important;
    }
    
    .image-upload-area:hover {
        border-color: var(--medical-blue) !important;
        background: rgba(0, 119, 182, 0.1) !important;
        transform: translateY(-2px) !important;
    }
    
    /* === Chat Container === */
    .chat-container {
        background: rgba(255,255,255,0.03) !important;
        border-radius: var(--border-radius) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
    }
    
    .chat-container .message {
        padding: 12px 16px !important;
        margin: 8px !important;
        border-radius: 12px !important;
    }
    
    .chat-container .user {
        background: var(--primary-gradient) !important;
        color: white !important;
        margin-left: 20% !important;
    }
    
    .chat-container .bot {
        background: rgba(255,255,255,0.08) !important;
        color: var(--text-light) !important;
        margin-right: 20% !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
    }
    
    /* === Buttons === */
    .btn-primary {
        background: var(--primary-gradient) !important;
        border: none !important;
        color: white !important;
        padding: 12px 24px !important;
        border-radius: 12px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        cursor: pointer !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .btn-primary:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    .btn-secondary {
        background: rgba(255,255,255,0.1) !important;
        border: 1px solid rgba(255,255,255,0.2) !important;
        color: var(--text-light) !important;
        padding: 10px 20px !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    
    .btn-secondary:hover {
        background: rgba(255,255,255,0.15) !important;
        border-color: rgba(255,255,255,0.3) !important;
    }
    
    /* === Result Image === */
    .result-display {
        border-radius: var(--border-radius) !important;
        overflow: hidden !important;
        box-shadow: 0 8px 32px rgba(0,0,0,0.3) !important;
        border: 2px solid var(--success-green) !important;
    }
    
    /* === Settings Accordion === */
    .accordion {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        border-radius: 12px !important;
        margin-top: 16px !important;
    }
    
    .accordion-header {
        color: var(--text-light) !important;
        font-weight: 500 !important;
    }
    
    /* === Sliders & Inputs === */
    input[type="range"] {
        accent-color: var(--medical-teal) !important;
    }
    
    input[type="text"], textarea {
        background: rgba(255,255,255,0.05) !important;
        border: 1px solid rgba(255,255,255,0.15) !important;
        color: var(--text-light) !important;
        border-radius: 10px !important;
        padding: 12px 16px !important;
    }
    
    input[type="text"]:focus, textarea:focus {
        border-color: var(--medical-teal) !important;
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(0, 180, 216, 0.2) !important;
    }
    
    /* === Example Buttons === */
    .example-btn {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.2), rgba(118, 75, 162, 0.2)) !important;
        border: 1px solid rgba(102, 126, 234, 0.3) !important;
        color: var(--text-light) !important;
        padding: 8px 16px !important;
        border-radius: 8px !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
    }
    
    .example-btn:hover {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.3), rgba(118, 75, 162, 0.3)) !important;
        transform: translateY(-1px) !important;
    }
    
    /* === Status Indicators === */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .status-success {
        background: rgba(46, 204, 113, 0.2);
        color: var(--success-green);
        border: 1px solid rgba(46, 204, 113, 0.3);
    }
    
    .status-processing {
        background: rgba(243, 156, 18, 0.2);
        color: var(--warning-orange);
        border: 1px solid rgba(243, 156, 18, 0.3);
    }
    
    /* === Footer === */
    .footer {
        text-align: center;
        padding: 20px;
        margin-top: 24px;
        color: rgba(255,255,255,0.5);
        font-size: 0.9rem;
        border-top: 1px solid rgba(255,255,255,0.1);
    }
    
    .footer a {
        color: var(--medical-teal);
        text-decoration: none;
    }
    
    /* === Responsive === */
    @media (max-width: 768px) {
        .header-banner h1 {
            font-size: 1.8rem !important;
        }
        
        .feature-pills {
            justify-content: center;
        }
    }
    
    /* === Animations === */
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    .processing-indicator {
        animation: pulse 1.5s infinite;
    }
    """
    
    # Premium Title with HTML
    title_md = """
    <div class="header-banner">
        <h1>🏥 TriMedAgent</h1>
        <p>Advanced Medical Image Analysis with Hybrid ReAct AI</p>
        <div class="feature-pills">
            <span class="feature-pill">🔬 BiomedCLIP Triage</span>
            <span class="feature-pill">🎯 Grounding DINO Detection</span>
            <span class="feature-pill">🎭 MedSAM Segmentation</span>
            <span class="feature-pill">🧠 LLaVA-Med Reasoning</span>
        </div>
    </div>
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
    
    # Helper to load local example safely (works in Kaggle/Colab)
    def load_local_example(example_path: str, example_query: str):
        try:
            img = Image.open(example_path).convert("RGB")
            return img, example_query
        except Exception:
            return None, example_query

    # Build Gradio interface with premium theme
    with gr.Blocks(
        css=custom_css, 
        title="TriMedAgent - Medical AI", 
        theme=gr.themes.Base(
            primary_hue=gr.themes.colors.purple,
            secondary_hue=gr.themes.colors.cyan,
            neutral_hue=gr.themes.colors.slate,
            font=gr.themes.GoogleFont("Inter"),
        ).set(
            body_background_fill="*neutral_950",
            body_background_fill_dark="*neutral_950",
            block_background_fill="*neutral_900",
            block_background_fill_dark="*neutral_900",
            border_color_primary="*neutral_700",
            block_border_width="1px",
            block_shadow="0 4px 20px rgba(0,0,0,0.3)",
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_400",
        )
    ) as demo:
        
        gr.HTML(title_md)
        
        with gr.Row(equal_height=True):
            # Left column - Image input
            with gr.Column(scale=4):
                gr.Markdown("### 📷 Input Image")
                image_input = gr.Image(
                    label="Upload Medical Image (X-ray, CT, MRI)",
                    type="pil",
                    height=400,
                    elem_classes=["image-upload-area", "glass-card"]
                )
                
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear All", variant="secondary", elem_classes=["btn-secondary"])
                    
                with gr.Accordion("⚙️ Advanced Settings", open=False, elem_classes=["accordion"]):
                    temperature = gr.Slider(
                        0, 1, value=0.2, step=0.1, 
                        label="🌡️ Temperature",
                        info="Lower = more focused, Higher = more creative"
                    )
                    max_tokens = gr.Slider(
                        128, 1024, value=512, step=64, 
                        label="📝 Max Response Length"
                    )
                    with gr.Row():
                        enable_detection = gr.Checkbox(
                            value=True, 
                            label="🎯 Enable Detection",
                            info="Grounding DINO"
                        )
                        enable_segmentation = gr.Checkbox(
                            value=True, 
                            label="🎭 Enable Segmentation",
                            info="MedSAM"
                        )
                
                with gr.Accordion("📚 Quick Examples", open=True, elem_classes=["accordion"]):
                    gr.Markdown("*Click to load sample image and query*")
                    with gr.Row():
                        example_chest = gr.Button(
                            "🫁 Chest X-ray: Find Nodules", 
                            elem_classes=["example-btn"]
                        )
                        example_mri = gr.Button(
                            "🧠 Brain MRI: Detect Tumor", 
                            elem_classes=["example-btn"]
                        )
                    with gr.Row():
                        example_cardio = gr.Button(
                            "❤️ Check Cardiomegaly", 
                            elem_classes=["example-btn"]
                        )
                        example_pneumonia = gr.Button(
                            "🦠 Detect Pneumonia", 
                            elem_classes=["example-btn"]
                        )
            
            # Right column - Chat
            with gr.Column(scale=6):
                gr.Markdown("### 💬 AI Analysis Chat")
                chatbot = gr.Chatbot(
                    label="Conversation",
                    height=300,
                    elem_classes=["chat-container", "glass-card"],
                    avatar_images=(None, "🤖"),
                    type="messages",
                    show_copy_button=True
                )
                
                with gr.Row():
                    text_input = gr.Textbox(
                        placeholder="💬 Ask about the medical image... (e.g., 'Find nodules in the lungs')",
                        label="",
                        scale=8,
                        container=False,
                        elem_classes=["glass-card"]
                    )
                    submit_btn = gr.Button(
                        "🚀 Analyze", 
                        variant="primary", 
                        scale=2,
                        elem_classes=["btn-primary"]
                    )
                
                gr.Markdown("### 📊 Analysis Result")
                output_image = gr.Image(
                    label="Annotated Result",
                    type="pil",
                    height=350,
                    elem_classes=["result-display", "glass-card"],
                    show_download_button=True
                )
        
        # Footer
        gr.HTML("""
        <div class="footer">
            <p>⚠️ <strong>Disclaimer:</strong> This tool is for research/educational purposes only. 
            Not intended for clinical diagnosis. Always consult healthcare professionals.</p>
            <p>Built with ❤️ using 
            <a href="https://github.com/ngnam1104/TriMedAgent" target="_blank">TriMedAgent</a> | 
            Powered by LLaVA-Med, Grounding DINO, MedSAM, BiomedCLIP</p>
        </div>
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
        
        # Wire example buttons now that text_input is defined
        example_chest.click(
            lambda: load_local_example("images/example_chest.png", "Find any nodules or suspicious lesions in the lungs"),
            outputs=[image_input, text_input]
        )
        example_mri.click(
            lambda: load_local_example("images/example_mri.png", "Detect any tumors or abnormal masses in the brain"),
            outputs=[image_input, text_input]
        )
        example_cardio.click(
            lambda: load_local_example("images/example_chest.png", "Check for cardiomegaly and evaluate heart size"),
            outputs=[image_input, text_input]
        )
        example_pneumonia.click(
            lambda: load_local_example("images/example_chest.png", "Look for signs of pneumonia or lung infiltrates"),
            outputs=[image_input, text_input]
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
    
    # Try specified port, fallback to auto-select if busy
    try:
        demo.queue().launch(
            share=share,
            server_port=port,
            debug=debug,
            show_error=True
        )
    except OSError:
        print(f"⚠️ Port {port} busy, using auto-select...")
        demo.queue().launch(
            share=share,
            server_port=None,  # Auto-select available port
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
