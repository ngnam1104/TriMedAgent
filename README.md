# 🏥 TriMed-Agent V2: State-based Medical AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/)

> **TriMed-Agent V2** - Hệ thống AI Y tế với kiến trúc **State-based ReAct**
> 
> 🧠 **Logic Router** → 📚 **RAG/Planner** → 🔧 **ToolKit** → ✅ **Verification**

---

## 🌟 What's New in V2

| Feature | V1 (Linear) | V2 (State-based) |
|---------|-------------|------------------|
| **Architecture** | Linear pipeline | ReAct Loop + State Machine |
| **Intent Routing** | ❌ Fixed | ✅ Logic Router (Theory/Diagnosis) |
| **Knowledge** | Parametric only | RAG + Parametric |
| **Planning** | Rule-based | LLaVA-Med + LoRA (JSON output) |
| **Training** | Pretrained | **SFT → RL (GRPO)** |
| **Small Objects** | Often fails | Smart Zoom + Fallback |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        TriMed-Agent V2 System                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│   User Input (Image + Query)                                            │
│         │                                                                │
│         ▼                                                                │
│   ┌─────────────────────────────────────┐                               │
│   │         LOGIC ROUTER                 │                               │
│   │   Intent: Theory (0.3) / Diagnosis   │                               │
│   └──────────────┬──────────────────────┘                               │
│                  │                                                       │
│        ┌─────────┴─────────┐                                            │
│        ▼                   ▼                                            │
│   ┌─────────┐      ┌─────────────────────────────────────────────┐     │
│   │   RAG   │      │            PLANNER (LLaVA-Med + LoRA)        │     │
│   │ Module  │      │  ┌─────────────────────────────────────────┐ │     │
│   │         │      │  │              ReAct Loop                 │ │     │
│   │PubMedBERT│     │  │  Thought → Plan → Action → Observation  │ │     │
│   │ + FAISS │      │  └─────────────────────────────────────────┘ │     │
│   └────┬────┘      └───────────────────┬─────────────────────────┘     │
│        │                               │                                │
│        │                               ▼                                │
│        │           ┌─────────────────────────────────────────────┐     │
│        │           │                TOOLKIT                       │     │
│        │           │  BiomedCLIP │ DINO │ MedSAM │ Zoom │ Gate   │     │
│        │           └─────────────────────────────────────────────┘     │
│        │                               │                                │
│        └───────────────┬───────────────┘                               │
│                        ▼                                                │
│                 Final Response                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

📖 **Chi tiết kiến trúc**: [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)

---

## 🎯 Key Features

### 1. Logic Router (Intent Classification)
```python
# Tự động phân loại câu hỏi
"Viêm phổi là gì?" → Theory → RAG Module
"Ảnh này có tổn thương không?" → Diagnosis → Planner
```

### 2. RAG Module (Medical Knowledge)
```python
# Trả lời câu hỏi lý thuyết với kiến thức y khoa
from src import MedicalRAG

rag = MedicalRAG()
answer = rag.query("Triệu chứng của COVID-19 trên X-quang?")
```

### 3. Planner (Structured JSON Output)
```python
# LLaVA-Med sinh ra kế hoạch có cấu trúc
{
  "thought": "Cần xác định loại ảnh trước",
  "action": "BiomedCLIP", 
  "action_input": {"task": "classify"}
}
```

### 4. Two-Stage Training
```
Stage 1: SFT (LoRA)     →  Học cách sinh JSON
Stage 2: RL (GRPO)      →  Tối ưu chiến lược
```

---

## 📁 Project Structure

```
TriMedAgent/
├── src/
│   ├── core/
│   │   ├── orchestrator.py      # Main ReAct loop
│   │   ├── logic_router.py      # Intent classification
│   │   └── state.py             # Agent state
│   ├── modules/
│   │   ├── brain/
│   │   │   └── planner.py       # LLaVA Planner
│   │   ├── knowledge/
│   │   │   └── rag_engine.py    # RAG with PubMedBERT
│   │   ├── vision/
│   │   │   ├── biomed_clip.py
│   │   │   ├── dino_detector.py
│   │   │   └── zoom_processor.py
│   │   └── segmentation/
│   │       └── medsam_wrapper.py
│   └── training/
│       ├── sft_trainer.py       # SFT with LoRA
│       └── grpo_trainer.py      # GRPO RL
├── scripts/
│   ├── train_sft.py             # 🆕 SFT training script
│   ├── train_rl_grpo.py         # 🆕 RL training script
│   └── merge_adapters.py
├── configs/
│   ├── config.yaml
│   ├── sft_config.yaml          # 🆕 SFT config
│   └── rl_config.yaml           # 🆕 RL config
├── docs/
│   └── ARCHITECTURE_V2.md       # 🆕 Detailed architecture
└── demo_colab_1xt4.ipynb
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```bash
# Upload demo_colab_1xt4.ipynb to Colab
# Select Runtime → T4 GPU → Run all cells
```

### Option 2: Local Installation

```bash
git clone https://github.com/ngnam1104/TriMedAgent.git
cd TriMedAgent
pip install -e .
```

---

## 💻 Usage

### Basic Inference

```python
from src import TriMedOrchestrator
from PIL import Image

# Initialize
config = {"device": {"llava_device": "cuda:0"}}
agent = TriMedOrchestrator(config)

# Process
image = Image.open("chest_xray.jpg")
result = agent.process(image, "Có tổn thương phổi không?")

print(result.final_report)
```

### With RAG (Theoretical Questions)

```python
from src import MedicalRAG

rag = MedicalRAG(api_key="your-groq-key")
response = rag.query("Viêm phổi do vi khuẩn điều trị như thế nào?")
print(response.answer)
```

---

## 🎓 Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)

```bash
# Train LoRA adapter to output structured JSON
python scripts/train_sft.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --dataset "data/sft_dataset" \
    --output_dir "outputs/sft-v1" \
    --lora_r 64 \
    --lora_alpha 128 \
    --epochs 3
```

### Stage 2: RL (GRPO)

```bash
# Continue training with RL to optimize policy
python scripts/train_rl_grpo.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --sft_adapter "outputs/sft-v1" \
    --output_dir "outputs/rl-v1" \
    --reward_weights "0.3,0.4,0.2,0.1"
```

### Upload to HuggingFace

```bash
# Push adapter to HF Hub
huggingface-cli upload ngnam1104/TriMedAgent-V2 outputs/rl-v1
```

📖 **Chi tiết training**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

---

## 📊 Reward Function (RL)

$$R_{total} = 0.3 \cdot R_{IoU} + 0.4 \cdot R_{Acc} + 0.2 \cdot R_{Format} + 0.1 \cdot R_{Step}$$

| Component | Description |
|-----------|-------------|
| $R_{IoU}$ | IoU với ground truth boxes |
| $R_{Acc}$ | Độ chính xác câu trả lời |
| $R_{Format}$ | JSON hợp lệ (+1) / Sai (-1) |
| $R_{Step}$ | Penalty mỗi bước (-0.1) |

---

## 🔧 Tools

| Tool | Model | Purpose |
|------|-------|---------|
| `BiomedCLIP` | microsoft/BiomedCLIP | Image triage |
| `GroundingDINO` | IDEA-Research | Detection |
| `MedSAM` | SAM-ViT-B | Segmentation |
| `ZoomProcessor` | Custom | Smart crop |
| `Gatekeeper` | LLaVA-Med | Verification |

---

## ⚙️ Requirements

- Python 3.10+
- CUDA 11.8+
- GPU: T4 (16GB) hoặc cao hơn

### Key Dependencies

```
torch>=2.0
transformers>=4.36
peft>=0.7  # LoRA
trl>=0.7   # GRPO training
accelerate
bitsandbytes
```

---

## 📝 License

Apache 2.0 License

---

## 🙏 Acknowledgments

- [ReAct](https://arxiv.org/abs/2210.03629) - Reasoning + Acting
- [GRPO](https://arxiv.org/abs/2402.03300) - DeepSeek-R1
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- [LoRA](https://arxiv.org/abs/2106.09685)
