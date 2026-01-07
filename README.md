# 🏥 TriMedAgent: Hybrid ReAct Medical AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

> **TriMedAgent V2** - SOTA Medical AI Agent với kiến trúc **Hybrid ReAct**: Plan → Zoom → Verify
> 
> Giải quyết vấn đề "box quá to" khi detect small pathologies như lung nodules

---

## 🌟 What's New in V2

| Feature | V1 (Linear) | V2 (Hybrid ReAct) |
|---------|-------------|-------------------|
| **Architecture** | Linear pipeline | ReAct Loop + Coarse-to-Fine |
| **Small Object Detection** | ❌ Often fails | ✅ Zoom-in strategy |
| **Planning** | Fixed intent parsing | LLaVA Strategic Planner |
| **Verification** | Simple gatekeeper | Size + Semantic validation |
| **Box Size Control** | No control | Automatic rejection if > threshold |

---

## 🧠 Hybrid ReAct Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HYBRID ReAct AGENT                       │
│                                                              │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐ │
│  │   PLANNER    │────▶│   EXECUTOR   │────▶│   VERIFIER   │ │
│  │   (LLaVA)    │     │ (DINO+SAM)   │     │   (LLaVA)    │ │
│  └──────────────┘     └──────────────┘     └──────────────┘ │
│         │                    │                    │         │
│         │              ┌─────▼─────┐              │         │
│         │              │  ZOOMER   │◀─────────────┘         │
│         │              │(Crop/Map) │  (retry if failed)     │
│         │              └───────────┘                        │
│         │                    │                              │
│         └────────────────────┴───────────────────────────── │
│                         ▼                                    │
│                  ┌─────────────┐                            │
│                  │ SYNTHESIZER │                            │
│                  │  (Report)   │                            │
│                  └─────────────┘                            │
└─────────────────────────────────────────────────────────────┘
```

### The Problem (V1)
```
Query: "Find nodules in right lung"
→ DINO detects ENTIRE lung (box ratio > 40%)
→ MedSAM segments entire lung
→ WRONG!
```

### The Solution (V2)
```
Query: "Find nodules in right lung"
→ LLaVA Planner: {target: "nodule", size: "tiny", region: "right_lung", strategy: "zoom_in"}
→ Zoomer: Crop to right lung region
→ DINO on cropped: Finds small nodule (box ratio < 5%)
→ Verifier: ✓ Size OK, ✓ Semantic check passed
→ Map coordinates back to original
→ MedSAM segments nodule precisely
→ CORRECT!
```

---

## 📁 Project Structure

```
TriMedAgent/
├── src/                         # 🔧 Core modules
│   ├── __init__.py             # Package exports
│   ├── utils.py                # Image utilities
│   ├── biomedclip_tool.py      # Visual triage
│   ├── grounding_dino_tool.py  # Object detection
│   ├── medsam_tool.py          # Segmentation
│   ├── llava_tool.py           # Visual reasoning (4-bit)
│   ├── rag_engine.py           # Medical RAG with Groq
│   └── orchestrator_v2.py      # 🆕 Hybrid ReAct Agent
├── images/                      # Sample images
├── demo_trimedagent_colab.ipynb # 📓 Demo notebook
├── pyproject.toml              # Dependencies
└── README.md                   # This file
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

1. Upload `demo_trimedagent_colab.ipynb` to Colab
2. Select Runtime → Change runtime type → **T4 GPU**
3. Run all cells!

### Option 2: Local Installation

```bash
# Clone repository
git clone https://github.com/your-repo/TriMedAgent.git
cd TriMedAgent

# Install dependencies
pip install -e .

# Or manual install
pip install torch transformers accelerate bitsandbytes
pip install gradio pillow numpy scipy
pip install segment-anything groundingdino-py
pip install groq sentence-transformers
```

---

## 💻 Usage

### Basic Usage with Hybrid ReAct

```python
from src import HybridReActOrchestrator
from PIL import Image

# Load image
image = Image.open("chest_xray.jpg")

# Initialize orchestrator
orchestrator = HybridReActOrchestrator(
    max_iterations=3,           # ReAct loop iterations
    enable_verification=True    # LLaVA semantic verification
)
orchestrator.load_tools(llava_quantize=True)  # 4-bit for T4 GPU

# Process with automatic zoom for small objects
result = orchestrator.process(image, "Find any nodules in the right lung")

# Access results
print(f"Strategy used: {result.strategic_plan.strategy.value}")
print(f"Target size: {result.strategic_plan.target_size.value}")
print(f"Detected: {len(result.verified_boxes)} regions")
print(f"ReAct iterations: {result.agent_iterations}")
print(result.llava_analysis)
```

### Understanding the Strategic Plan

```python
# The planner automatically generates a strategy
plan = result.strategic_plan

print(f"Target: {plan.target_object}")        # "nodule"
print(f"Size: {plan.target_size.value}")      # "tiny" 
print(f"Location: {plan.anatomical_location}") # "right_upper_lobe"
print(f"Strategy: {plan.strategy.value}")      # "zoom_in"
print(f"Detection prompt: {plan.detection_prompt}")  # "small nodule. white spot."
```

### Manual Strategy Override

```python
from src import Strategy, TargetSize, StrategicPlan

# Create custom plan
custom_plan = StrategicPlan(
    target_object="tumor",
    target_size=TargetSize.SMALL,
    anatomical_location="liver",
    strategy=Strategy.ZOOM_IN,
    detection_prompt="tumor. mass. hepatic lesion.",
    confidence_threshold=0.2
)

# Use with orchestrator
result = orchestrator.process(image, "Find liver tumor", plan=custom_plan)
```
print(result.llava_response)
print(f"Detected: {len(result.verified_boxes)} regions")
```

### With Gradio UI

```python
import gradio as gr
from src import HybridReActOrchestrator

# Initialize
orchestrator = HybridReActOrchestrator()
orchestrator.load_tools()

def analyze(image, question):
    result = orchestrator.process(image, question)
    info = f"""
**Strategy:** {result.strategic_plan.strategy.value}
**Target Size:** {result.strategic_plan.target_size.value}
**Iterations:** {result.agent_iterations}
**Detections:** {len(result.verified_boxes)}

{result.llava_analysis}
"""
    return result.annotated_image, info

# Create UI
demo = gr.Interface(
    fn=analyze,
    inputs=[gr.Image(type="pil"), gr.Textbox(label="Question")],
    outputs=[gr.Image(label="Result"), gr.Markdown(label="Analysis")]
)
demo.launch()
```

---

## 🔧 Detection Strategies

### Strategy Types

| Strategy | When Used | How It Works |
|----------|-----------|--------------|
| `GLOBAL_SCAN` | Large objects, unknown location | Full image detection |
| `ZOOM_IN` | Small objects (nodules) with known region | Crop → Detect → Map back |
| `MULTI_SCALE` | Uncertain size | Full + Grid detection |
| `SLIDING_WINDOW` | Very small objects | Overlapping grid scan |

### Target Size Thresholds

| Size | Max Box Ratio | Examples |
|------|---------------|----------|
| `TINY` | 5% | Nodules, calcifications |
| `SMALL` | 15% | Lesions, small tumors |
| `MEDIUM` | 35% | Organs, larger tumors |
| `LARGE` | 60% | Full organs, diffuse patterns |

### Anatomical Regions (Pre-defined)

```python
# Chest X-ray
"right_lung", "left_lung", "right_upper_lobe", "right_lower_lobe"
"left_upper_lobe", "left_lower_lobe", "heart", "mediastinum"

# Abdominal CT
"liver", "spleen", "right_kidney", "left_kidney", "pancreas"

# Brain MRI
"frontal_lobe", "temporal_lobe_left", "temporal_lobe_right"
"occipital_lobe", "cerebellum"

# Generic
"upper_left", "upper_right", "lower_left", "lower_right", "center"
```

---

## 📊 ReAct Loop Flow

```
Iteration 1:
├─ THINK: "Starting detection for 'nodule' (size: tiny, strategy: zoom_in) in 'right_lung'"
├─ ACT: Crop to right_lung → DINO detect → Map boxes back
├─ OBSERVE: Box ratio 3% ✓, Semantic verify ✓
└─ RESULT: SUCCESS (1 valid detection)

Iteration 2 (if needed):
├─ THINK: "Previous action was 'reject_size'. Trying with stricter threshold."
├─ ACT: Zoom tighter or use fallback region
├─ OBSERVE: Re-validate
└─ RESULT: SUCCESS or retry
```

---

## 🔬 Tool Details

### BiomedCLIPTool
```python
from src import BiomedCLIPTool

clip = BiomedCLIPTool()
result = clip.triage(image)
# Returns: modality, modality_confidence, abnormality, abnormality_confidence
```

### LLaVATool (4-bit Quantization)
```python
from src import LLaVATool

llava = LLaVATool(quantize_4bit=True)  # For T4 GPU
response = llava.query(image, "What abnormalities are present?")
```

### GroundingDINOTool
```python
from src import GroundingDINOTool

dino = GroundingDINOTool()
boxes = dino.detect(image, "tumor. mass. lesion.")
# Returns: [[x1, y1, x2, y2], ...]
```

### MedSAMTool
```python
from src import MedSAMTool

medsam = MedSAMTool()
masks = medsam.segment(image, boxes=[[100, 100, 200, 200]])
# Returns: [numpy.ndarray, ...]
```

### MedicalRAG
```python
from src import MedicalRAG

# Requires GROQ_API_KEY environment variable
rag = MedicalRAG()
response = rag.query("What is the treatment for pneumonia?")
print(response.answer)
```

---

## ⚙️ Configuration

### Environment Variables

```bash
# For RAG functionality
export GROQ_API_KEY="your-groq-api-key"

# For Colab secrets
# Add GROQ_API_KEY in Colab's Secrets panel
```

### GPU Memory

| GPU | LLaVA Mode | Memory Usage |
|-----|------------|--------------|
| T4 (16GB) | 4-bit | ~8GB |
| A100 (40GB) | Full | ~14GB |
| L4 (24GB) | 4-bit/Full | ~8-14GB |

---

## 📝 License

Apache 2.0 License

---

## 🙏 Acknowledgments

- [MedRAX](https://arxiv.org/abs/2406.xxxxx) - ReAct Loop inspiration
- [MMedAgent](https://arxiv.org/abs/2406.xxxxx) - Coarse-to-Fine strategy
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med) - Medical visual reasoning
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_PMC-ViT-B-16-224) - Medical image classification
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO) - Object detection
- [Segment Anything](https://github.com/facebookresearch/segment-anything) - Segmentation
- [Groq](https://groq.com/) - Fast LLM inference
