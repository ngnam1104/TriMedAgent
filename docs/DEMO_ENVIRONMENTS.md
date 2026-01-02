# 🖥️ Hướng Dẫn Demo TriMedAgent trên Các Môi Trường

## 🆕 Kiến Trúc Orchestrator (Khuyến nghị)

TriMedAgent sử dụng **Orchestrator Pattern** - một Thin Client điều phối các Worker qua HTTP.

```
┌─────────────────────────────────────────────────────────────────┐
│     demo_trimedagent_orchestrator.ipynb (Thin Client)           │
│     → from src import TriMedOrchestrator                        │
│     → result = orchestrator.run_full_chain(image, query)        │
└─────────────────────────────────────────────────────────────────┘
                              │ HTTP
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
   │ BiomedCLIP  │     │  LLaVA-Med  │     │  DINO/SAM   │
   │   :21006    │     │   :21002    │     │ :21003/21004│
   └─────────────┘     └─────────────┘     └─────────────┘
```

---

## 📊 Tổng Quan Yêu Cầu Tài Nguyên

### Model Requirements

| Model | VRAM | Disk | Purpose |
|-------|------|------|---------|
| **LLaVA-Med 7B** | ~14GB | ~14GB | Main VQA model |
| **BiomedCLIP** | ~0.5GB | ~0.4GB | Triage + Gatekeeper |
| **Grounding DINO** | ~1.5GB | ~1.2GB | Detection |
| **MedSAM** | ~2.5GB | ~2.4GB | Segmentation |
| **Total** | **~18-20GB** | **~18GB** | Full pipeline |

### Optimization Options

| Mode | VRAM Required | Models Active |
|------|---------------|---------------|
| **Full Pipeline** | 18-20GB | All models |
| **Lite (4-bit)** | 8-10GB | LLaVA quantized + others |
| **Triage Only** | 2GB | BiomedCLIP only |
| **Demo (simulated)** | 1GB | BiomedCLIP + fake responses |

---

## 📂 Demo Notebooks

| Notebook | Environment | Description |
|----------|-------------|-------------|
| `demo_trimedagent_colab.ipynb` | **Colab/Kaggle** | ⭐ **CHÍNH** - Full demo với Tools + Orchestrator + Chatbot |
| `demo_trimedagent_orchestrator.ipynb` | **Local + Workers** | HTTP Orchestrator Pattern |
| `demo_colab.ipynb` | **Colab** | Legacy demo |

### ✨ `demo_trimedagent_colab.ipynb` - KHUYẾN NGHỊ

Notebook này chạy **trực tiếp** trên Colab/Kaggle (không cần HTTP workers):

```
┌─────────────────────────────────────────────────────────────────┐
│                 demo_trimedagent_colab.ipynb                     │
├─────────────────────────────────────────────────────────────────┤
│  Section 1: Setup Environment                                    │
│  Section 2: 🔧 Individual Tools Demo                            │
│             → BiomedCLIP (Triage + Gatekeeper)                  │
│             → Grounding DINO (Detection)                        │
│             → MedSAM (Segmentation)                             │
│  Section 3: 🎯 Orchestrator Demo                                │
│             → TriMedOrchestratorLocal                           │
│             → Full pipeline execution                           │
│  Section 4: 💬 Interactive Chatbot                              │
│             → Gradio interface                                   │
│             → Upload image & chat                               │
└─────────────────────────────────────────────────────────────────┘
```

**Cách sử dụng Orchestrator (Local Edition):**

```python
# 1. Import
from src import TriMedOrchestrator

# 2. Initialize (không cần GPU cho client)
orchestrator = TriMedOrchestrator()

# 3. Health Check
print(orchestrator.health_check())

# 4. Run Pipeline
result = orchestrator.run_full_chain(
    image="path/to/image.jpg",
    user_query="Find and segment any tumors"
)

# 5. Access Results
print(result.triage.modality)        # "Chest X-ray"
print(result.verified_boxes)          # [[x1,y1,x2,y2], ...]
print(result.masks)                   # Segmentation masks
```

---

## 🔵 Option 1: Google Colab

### 1.1. Sử dụng `demo_trimedagent_colab.ipynb` ⭐ KHUYẾN NGHỊ

**Step-by-step:**

1. **Upload notebook** lên Google Colab:
   - Vào [colab.research.google.com](https://colab.research.google.com)
   - File → Upload notebook → Chọn `demo_trimedagent_colab.ipynb`

2. **Chọn GPU Runtime:**
   - Runtime → Change runtime type → GPU
   - Colab Free: T4 (tự động)
   - Colab Pro: A100 (chọn trong dropdown)

3. **Chạy từng section:**
   ```
   Section 1: Setup      → Cài đặt dependencies
   Section 2: Tools      → Test từng tool riêng biệt
   Section 3: Orchestrator → Chạy full pipeline
   Section 4: Chatbot    → Gradio interface (có link public)
   ```

**Chế độ theo GPU:**

| GPU | VRAM | Mode | Features |
|-----|------|------|----------|
| T4 | 16GB | LITE | BiomedCLIP + DINO |
| A100 | 40GB | FULL | All tools + MedSAM |

### 1.2. Quick Start Code

```python
# Cell 1: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

# Cell 2: Install
!pip install -q open_clip_torch pillow requests matplotlib gradio

# Cell 2: Load BiomedCLIP
import open_clip
model, preprocess = open_clip.create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
```

---

## 🟠 Option 2: Kaggle

### 2.1. Sử dụng `demo_trimedagent_colab.ipynb` ⭐ KHUYẾN NGHỊ

**Notebook này hoạt động trên cả Colab và Kaggle!**

**Step-by-step:**

1. **Tạo Notebook mới:**
   - Vào [kaggle.com/code](https://www.kaggle.com/code)
   - New Notebook → File → Import Notebook
   - Upload `demo_trimedagent_colab.ipynb`

2. **Chọn GPU:**
   - Settings (sidebar phải) → Accelerator → GPU P100 hoặc T4 x2

3. **Chạy notebook:**
   - Run All hoặc chạy từng cell
   - Section 4 (Chatbot) sẽ tạo link Gradio public

**Kaggle Specs:**
- GPU: P100/T4 (16GB VRAM)
- RAM: 13-30GB
- Session: 12 giờ (30 giờ nếu verified)

### 2.2. Quick Start

1. Vào [Kaggle Notebooks](https://www.kaggle.com/notebooks)
2. Upload `demo_kaggle.ipynb`
3. Settings → Accelerator → GPU P100
4. Run All Cells

```python
# Cell 1: Install
!pip install -q open_clip_torch gradio

# Cell 2: Check GPU
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 🟢 Option 3: GPU Rental (RunPod/Vast.ai)

### 3.1. Sử dụng `demo.ipynb` (Khuyến nghị)

**Platforms:**
- **RunPod:** RTX 3090/4090/A100
- **Vast.ai:** A100, RTX 4090
- **Lambda Labs:** A100
- **Paperspace:** RTX 4000+

### 3.2. RunPod Quick Start

1. Vào [runpod.io](https://runpod.io)
2. Deploy → GPU Pod → RTX 4090 (24GB)
3. Template: `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel`
4. Start Jupyter → Upload `demo.ipynb`

**Giá ước tính:**
- RTX 3090: ~$0.34/hr
- RTX 4090: ~$0.44/hr
- A100 40GB: ~$0.79/hr

### 3.3. Vast.ai Quick Start

1. Vào [vast.ai](https://vast.ai)
2. Search → Filter: RTX 4090, VRAM > 20GB
3. Select instance → Launch Jupyter
4. Upload `demo.ipynb` và chạy

---

## 🖥️ Option 4: Local GPU

### 4.1. Yêu cầu

- **Minimum:** RTX 3080 10GB (Lite mode)
- **Recommended:** RTX 3090/4090 24GB (Full pipeline)
- Python 3.10+, CUDA 11.8+

### 4.2. Setup

```bash
# Clone repo
git clone <your-repo>
cd TriMedAgent

# Create environment
conda create -n trimedagent python=3.10 -y
conda activate trimedagent

# Install dependencies
pip install -e .
pip install open_clip_torch gradio

# Run demo
jupyter notebook demo_trimedagent.ipynb
```

---

## 📋 Pipeline Overview

```
User uploads image
        ↓
┌─────────────────────────────────────┐
│  STAGE 1: PERCEPTION (Triage)       │
│  BiomedCLIP classifies modality     │
│  → "Chest X-ray" (95% confidence)   │
│  → Recommend: [grounding_dino, medsam]
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  STAGE 2: REASONING (Context)       │
│  Inject triage context into prompt  │
│  → Enhanced understanding           │
└─────────────────────────────────────┘
        ↓
┌─────────────────────────────────────┐
│  STAGE 3: GATEKEEPING (Verify)      │
│  BiomedCLIP verifies detections     │
│  → 4 boxes → 2 verified boxes       │
└─────────────────────────────────────┘
        ↓
Return response to user
```

---

## 🔧 Configuration

Tất cả configuration nằm trong `serve/labels.json`:

```json
{
  "triage_labels": [
    "Chest X-ray", "Brain MRI", "Abdominal CT", "Histopathology",
    "Ultrasound", "Dermoscopy", "Gross pathology", "Bone X-ray",
    "Lung CT", "Retinal fundus", "Mammography"
  ],
  "gatekeeper_prompts": {
    "positive": "Pathological finding, lesion, tumor, abnormality",
    "negative": "Normal tissue, healthy anatomy, background noise"
  },
  "thresholds": {
    "triage_confidence": 0.5,
    "gatekeeper_confidence": 0.6
  }
}
```

---

## ❓ Troubleshooting

### CUDA Out of Memory

```python
# Option 1: Use 4-bit quantization
from transformers import BitsAndBytesConfig
quantization_config = BitsAndBytesConfig(load_in_4bit=True)

# Option 2: Use CPU for some models
device = "cpu"

# Option 3: Reduce batch size
batch_size = 1
```

### Model Download Failed

```python
# Use offline mode
import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Or download manually
!wget https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/resolve/main/open_clip_pytorch_model.bin
```

### Gradio Not Launching

```python
# Try different port
demo.launch(server_port=7861)

# Enable share
demo.launch(share=True)

# Debug mode
demo.launch(debug=True)
```

---

## 📚 Files Reference

| File | Purpose |
|------|---------|
| `demo_trimedagent_orchestrator.ipynb` | ⭐ Main demo (Orchestrator) |
| `src/trimed_orchestrator.py` | ⭐ Thin Client logic |
| `src/__init__.py` | Package exports |
| `serve/labels.json` | Pipeline configuration |
| `serve/biomedclip_worker.py` | Triage + Gatekeeper worker |
| `serve/grounding_dino_worker.py` | Detection worker |
| `serve/MedSAM_worker.py` | Segmentation worker |
| `llava/serve/model_worker.py` | LLaVA-Med worker |
| `QUICKSTART.md` | Quick start guide |

---

## 🚀 Khởi Chạy Workers (Trên máy có GPU)

```bash
# Terminal 1: Controller
python -m serve.controller --port 20001

# Terminal 2: LLaVA-Med
python -m llava.serve.model_worker --port 21002 \
    --controller-address http://localhost:20001 \
    --model-path liuhaotian/llava-v1.5-7b

# Terminal 3: BiomedCLIP (Triage + Gatekeeper)
python -m serve.biomedclip_worker --port 21006 \
    --controller-address http://localhost:20001

# Terminal 4: Grounding DINO
python -m serve.grounding_dino_worker --port 21003 \
    --controller-address http://localhost:20001

# Terminal 5: MedSAM
python -m serve.MedSAM_worker --port 21004 \
    --controller-address http://localhost:20001
```

---

## 🔗 Liên Kết Giữa Các File

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          FILE DEPENDENCY GRAPH                          │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  demo_trimedagent_orchestrator.ipynb                                    │
│         │                                                               │
│         └──▶ src/__init__.py                                            │
│                    │                                                    │
│                    └──▶ src/trimed_orchestrator.py                      │
│                              │                                          │
│                              ├──▶ llava/conversation.py (conv_templates)│
│                              │                                          │
│                              └──▶ serve/labels.json (config)            │
│                                                                         │
│  HTTP WORKERS (Running separately):                                     │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  serve/biomedclip_worker.py   ←── Port 21006                    │   │
│  │  serve/grounding_dino_worker.py ←── Port 21003                  │   │
│  │  serve/MedSAM_worker.py       ←── Port 21004                    │   │
│  │  llava/serve/model_worker.py  ←── Port 21002                    │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
