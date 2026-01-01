# 🖥️ Hướng Dẫn Demo TriMedAgent trên Các Môi Trường

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

## 🔵 Option 1: Google Colab

### 1.1. Colab Free (T4 16GB) - ⚠️ Limited

**Giới hạn:**
- GPU: NVIDIA T4 16GB VRAM
- RAM: 12.7GB
- Disk: ~78GB
- Session: 12 giờ max

**Kết luận:** ❌ **KHÔNG ĐỦ** cho full pipeline (cần 18-20GB VRAM)

**Giải pháp:** Dùng **Lite Mode (4-bit quantization)**

```python
# Colab Free - Lite Mode Setup
# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Install dependencies
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
!pip install transformers accelerate bitsandbytes
!pip install open_clip_torch gradio pillow
!pip install einops timm safetensors

# Cell 3: Clone repo
!git clone https://github.com/YOUR_USERNAME/MMedAgent-v2.git
%cd MMedAgent-v2

# Cell 4: Download models (smaller versions)
!pip install gdown
!gdown --folder https://drive.google.com/drive/folders/YOUR_FOLDER_ID -O ./models/

# Cell 5: Load with 4-bit quantization
from transformers import BitsAndBytesConfig
import torch

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# This reduces LLaVA from 14GB to ~4GB VRAM
```

### 1.2. Colab Pro/Pro+ (A100 40GB) - ✅ Recommended

**Thông số:**
- GPU: NVIDIA A100 40GB VRAM
- RAM: 83GB
- Disk: ~166GB
- Session: 24 giờ max

**Kết luận:** ✅ **ĐỦ** cho full pipeline

**Giá:** ~$9.99/tháng (Pro) hoặc ~$49.99/tháng (Pro+)

```python
# Colab Pro - Full Pipeline Setup

# Cell 1: Verify A100
!nvidia-smi
# Should show: NVIDIA A100-SXM4-40GB

# Cell 2: Install all dependencies
%%bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install transformers==4.36.0 accelerate==0.25.0
pip install open_clip_torch==2.23.0
pip install gradio==4.8.0
pip install einops timm safetensors sentencepiece
pip install groundingdino-py
pip install segment-anything

# Cell 3: Clone and setup
!git clone https://github.com/YOUR_USERNAME/MMedAgent-v2.git
%cd MMedAgent-v2

# Cell 4: Download models
# Option A: From HuggingFace
!huggingface-cli download microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 --local-dir ./models/biomedclip
!huggingface-cli download microsoft/llava-med-v1.5-mistral-7b --local-dir ./models/llava-med

# Option B: From Google Drive (faster if you uploaded)
!gdown --folder YOUR_GDRIVE_FOLDER_ID -O ./models/

# Cell 5: Run demo
%run demo_trimedagent.ipynb
```

### 1.3. Colab Notebook Template

```python
# ============================================================
# 🏥 TRIMEDAGENT COLAB DEMO
# ============================================================

# @title 1. Setup Environment
# @markdown Chọn mode phù hợp với GPU của bạn

MODE = "lite"  # @param ["full", "lite", "triage_only"]
GPU_CHECK = True  # @param {type:"boolean"}

import subprocess
if GPU_CHECK:
    result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], 
                          capture_output=True, text=True)
    print(result.stdout)
    
    if "A100" in result.stdout:
        print("✅ A100 detected - Full mode available")
        MODE = "full"
    elif "T4" in result.stdout:
        print("⚠️ T4 detected - Using Lite mode")
        MODE = "lite"
    else:
        print("❓ Unknown GPU - Using Lite mode")
        MODE = "lite"

# @title 2. Install Dependencies
!pip install -q torch torchvision torchaudio
!pip install -q transformers accelerate bitsandbytes
!pip install -q open_clip_torch gradio pillow einops timm

# @title 3. Load Models
import torch
import open_clip

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Load BiomedCLIP (always needed)
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
model = model.to(device)
print("✅ BiomedCLIP loaded")

# @title 4. Run Chatbot Demo
# [Insert TriMedAgent class and Gradio UI code here]
```

---

## 🟠 Option 2: Kaggle Notebooks

### 2.1. Kaggle Free (P100/T4 x2)

**Thông số:**
- GPU: NVIDIA P100 16GB hoặc T4 x2 (2x16GB)
- RAM: 13GB
- Disk: ~20GB (persistent) + ~5GB temp
- Session: 12 giờ/tuần GPU

**Kết luận:** ⚠️ **HẠN CHẾ** - Cần optimization

**Ưu điểm:**
- Miễn phí
- Persistent storage
- Có thể save models

**Nhược điểm:**
- Giới hạn 30h GPU/tuần
- VRAM hạn chế

### 2.2. Kaggle Setup

```python
# ============================================================
# 🏥 TRIMEDAGENT KAGGLE DEMO
# ============================================================

# Cell 1: Check GPU
!nvidia-smi

# Cell 2: Install dependencies
!pip install -q open_clip_torch transformers accelerate bitsandbytes
!pip install -q gradio einops timm safetensors

# Cell 3: Upload models to Kaggle Datasets
# Trước đó: Tạo Kaggle Dataset chứa models
# 1. Vào kaggle.com/datasets
# 2. New Dataset > Upload files
# 3. Upload: biomedclip/, llava-med/ (hoặc quantized version)

# Cell 4: Link to dataset
from kaggle_secrets import UserSecretsClient
import os

# Mount your model dataset
# Trong Kaggle Notebook: Add Data > Your Datasets > trimedagent-models
MODEL_PATH = "/kaggle/input/trimedagent-models"

# Cell 5: Load BiomedCLIP
import torch
import open_clip

device = "cuda"
model, preprocess, tokenizer = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
model = model.to(device).eval()

# Cell 6: Load LLaVA-Med with 4-bit (for T4/P100)
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4"
)

# Load quantized model
llava_model = AutoModelForCausalLM.from_pretrained(
    f"{MODEL_PATH}/llava-med-7b",
    quantization_config=bnb_config,
    device_map="auto"
)
```

### 2.3. Kaggle Dataset Structure

Tạo Kaggle Dataset với cấu trúc:

```
trimedagent-models/
├── biomedclip/
│   ├── config.json
│   ├── model.safetensors
│   └── tokenizer/
├── llava-med-7b-4bit/  # Quantized version
│   ├── config.json
│   ├── model-4bit.safetensors
│   └── tokenizer/
├── grounding-dino/
│   └── groundingdino_swinb_cogcoor.pth
└── medsam/
    └── medsam_vit_b.pth
```

---

## 🟢 Option 3: Thuê GPU Cloud

### 3.1. So Sánh Các Provider

| Provider | GPU | VRAM | Giá/giờ | Min Cost |
|----------|-----|------|---------|----------|
| **RunPod** | RTX 4090 | 24GB | $0.44 | ~$0.44 |
| **RunPod** | A100 40GB | 40GB | $1.04 | ~$1.04 |
| **Vast.ai** | RTX 3090 | 24GB | $0.20-0.40 | ~$0.20 |
| **Vast.ai** | A100 40GB | 40GB | $0.80-1.20 | ~$0.80 |
| **Lambda Labs** | A100 40GB | 40GB | $1.10 | $1.10 |
| **Paperspace** | RTX 4000 | 8GB | $0.45 | $0.45 |
| **Paperspace** | A100 40GB | 40GB | $3.09 | $3.09 |

### 3.2. Recommended: RunPod (RTX 4090 24GB)

**Tại sao chọn RTX 4090:**
- ✅ 24GB VRAM - đủ cho full pipeline
- ✅ Giá rẻ: $0.44/giờ
- ✅ Setup nhanh với Docker
- ✅ Persistent storage

**Tính toán chi phí:**
- Demo 2 giờ: ~$0.88
- Demo 1 ngày (8h): ~$3.52
- Demo 1 tuần: ~$25

### 3.3. RunPod Setup Guide

**Step 1: Tạo Account**
1. Vào https://runpod.io
2. Sign up và add credit ($10 minimum)

**Step 2: Deploy Pod**
```bash
# Trong RunPod UI:
# 1. Click "Deploy" > "GPU Pods"
# 2. Chọn RTX 4090 24GB (~$0.44/hr)
# 3. Template: PyTorch 2.1.0
# 4. Container Disk: 50GB
# 5. Volume Disk: 100GB (persistent)
# 6. Click "Deploy"
```

**Step 3: Connect và Setup**
```bash
# SSH vào pod hoặc dùng Web Terminal

# Clone repo
git clone https://github.com/YOUR_USERNAME/MMedAgent-v2.git
cd MMedAgent-v2

# Install dependencies
pip install torch torchvision torchaudio
pip install transformers accelerate
pip install open_clip_torch gradio
pip install einops timm safetensors

# Download models
python -c "
import open_clip
model, _, _ = open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
print('BiomedCLIP downloaded')
"

# Download LLaVA-Med
huggingface-cli download microsoft/llava-med-v1.5-mistral-7b --local-dir ./models/llava-med

# Run demo
python -m jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root
```

**Step 4: Access Notebook**
- RunPod cung cấp public URL cho port 8888
- Hoặc dùng Gradio share link

### 3.4. Vast.ai Setup (Budget Option)

**Tại sao Vast.ai:**
- ✅ Rẻ nhất: $0.20-0.40/giờ cho RTX 3090
- ✅ Community GPUs
- ⚠️ Ít stable hơn RunPod

```bash
# Step 1: Tạo account vast.ai

# Step 2: Search GPU
# Filter: RTX 3090, 24GB VRAM, $0.30/hr

# Step 3: Rent instance với Docker image
# Image: pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Step 4: SSH và setup
ssh -p PORT root@IP_ADDRESS

cd /workspace
git clone https://github.com/YOUR_USERNAME/MMedAgent-v2.git
cd MMedAgent-v2

# Install và run như RunPod
```

---

## 📋 Quick Comparison

| Feature | Colab Free | Colab Pro | Kaggle | RunPod | Vast.ai |
|---------|------------|-----------|--------|--------|---------|
| **VRAM** | 16GB | 40GB | 16GB | 24GB | 24GB |
| **Full Pipeline** | ❌ | ✅ | ⚠️ | ✅ | ✅ |
| **Cost** | Free | $10/mo | Free | $0.44/hr | $0.20/hr |
| **Session Limit** | 12h | 24h | 12h/wk | None | None |
| **Persistent** | ❌ | ❌ | ✅ | ✅ | ✅ |
| **Ease of Use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |

---

## 🎯 Recommendations

### Cho Demo Nhanh (< 2 giờ):
1. **Colab Pro** - Dễ nhất, A100 40GB
2. **RunPod RTX 4090** - Rẻ, đủ VRAM

### Cho Demo Dài (> 1 ngày):
1. **RunPod với Volume** - Persistent storage
2. **Vast.ai** - Budget option

### Cho Presentation/Demo:
1. **Colab Pro** - Share link dễ dàng
2. **Gradio share=True** - Public URL

### Cho Development:
1. **RunPod** - Full control, persistent
2. **Local GPU** (nếu có RTX 3090/4090)

---

## 📦 Pre-built Docker Image (Optional)

Để setup nhanh hơn, tôi có thể tạo Docker image:

```dockerfile
# Dockerfile.trimedagent
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

WORKDIR /app

# Install dependencies
RUN pip install transformers accelerate open_clip_torch gradio \
    einops timm safetensors bitsandbytes

# Clone repo
RUN git clone https://github.com/YOUR_USERNAME/MMedAgent-v2.git

# Pre-download BiomedCLIP
RUN python -c "import open_clip; open_clip.create_model_and_transforms('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')"

WORKDIR /app/MMedAgent-v2

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
```

Build và push:
```bash
docker build -t trimedagent:latest -f Dockerfile.trimedagent .
docker push YOUR_DOCKERHUB/trimedagent:latest
```

Sử dụng trên RunPod/Vast.ai:
```
Image: YOUR_DOCKERHUB/trimedagent:latest
```

---

## ⚠️ Lưu Ý Quan Trọng

1. **VRAM Monitoring:**
   ```python
   import torch
   print(f"VRAM Used: {torch.cuda.memory_allocated()/1e9:.2f}GB")
   print(f"VRAM Total: {torch.cuda.get_device_properties(0).total_memory/1e9:.2f}GB")
   ```

2. **Clear GPU Memory:**
   ```python
   import gc
   torch.cuda.empty_cache()
   gc.collect()
   ```

3. **Model Loading Order:**
   - Load BiomedCLIP first (smallest)
   - Load LLaVA-Med last (largest)
   - Use `device_map="auto"` for automatic placement

4. **Gradio Share:**
   ```python
   demo.launch(share=True)  # Creates public URL valid for 72h
   ```
