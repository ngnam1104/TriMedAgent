# 🏥 TriMedAgent: Intelligent Triage Pipeline for Medical Image Analysis

[![Based on MMedAgent](https://img.shields.io/badge/Based%20on-MMedAgent-blue.svg)](https://github.com/Wangyixinxin/MMedAgent)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](http://www.apache.org/licenses/LICENSE-2.0)

> **TriMedAgent** là kiến trúc tác nhân y tế cải tiến dựa trên MMedAgent, tập trung vào quy trình **"Intelligent Triage Pipeline"** (Phân loại Thông minh) để nâng cao độ chính xác trong chẩn đoán hình ảnh y tế.

---

## 📋 Mục Lục

- [1. Giới Thiệu](#1-giới-thiệu)
- [2. Vấn Đề & Động Lực](#2-vấn-đề--động-lực)
- [3. Giải Pháp Đề Xuất](#3-giải-pháp-đề-xuất)
- [4. Kiến Trúc Hệ Thống](#4-kiến-trúc-hệ-thống)
- [5. Chi Tiết Kỹ Thuật](#5-chi-tiết-kỹ-thuật)
- [6. Cài Đặt](#6-cài-đặt)
- [7. Hướng Dẫn Sử Dụng](#7-hướng-dẫn-sử-dụng)
- [8. Chatbot - Tính Năng Chính](#8-chatbot---tính-năng-chính) ⭐ **NEW**
- [9. Cấu Hình](#9-cấu-hình)
- [10. Demo & Thử Nghiệm](#10-demo--thử-nghiệm)
- [11. Đóng Góp](#11-đóng-góp)
- [12. Tài Liệu Tham Khảo](#12-tài-liệu-tham-khảo)

---

## 1. Giới Thiệu

### 1.1. MMedAgent là gì?

**MMedAgent** (Multi-modal Medical Agent) là hệ thống AI y tế đa phương thức đầu tiên tích hợp nhiều công cụ chuyên biệt để xử lý các tác vụ y tế khác nhau:

| Tác vụ | Công cụ | Mô tả |
|--------|---------|-------|
| VQA | LLaVA-Med | Hỏi đáp về hình ảnh y tế |
| Classification | BiomedCLIP | Phân loại hình ảnh y tế |
| Grounding | Grounding DINO | Phát hiện vùng bất thường |
| Segmentation | MedSAM | Phân vùng tổn thương |
| G-Seg | DINO + MedSAM | Phân vùng theo text prompt |
| MRG | ChatCAD | Sinh báo cáo y tế |
| RAG | ChatCAD+ | Truy vấn kiến thức y tế |

### 1.2. TriMedAgent là gì?

**TriMedAgent** là phiên bản cải tiến của MMedAgent, bổ sung:

- 🔬 **Visual Triage**: Phân loại tự động loại hình ảnh y tế
- 🧠 **Context-Aware Reasoning**: Tiêm ngữ cảnh vào LLM
- 🛡️ **Gatekeeper Verification**: Lọc bỏ false positives
- ⚡ **Conditional Execution**: Tối ưu hóa việc gọi công cụ

---

## 2. Vấn Đề & Động Lực

### 2.1. Hạn Chế của MMedAgent Gốc

```
┌─────────────────────────────────────────────────────────────────────┐
│                    VẤN ĐỀ VỚI MMEDAGENT GỐC                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  1. THIẾU NGỮ CẢNH (Context-Blind)                                 │
│     ┌─────────────┐                                                │
│     │   Image     │──▶ LLaVA-Med ──▶ "Detect tumor"                │
│     │  (Unknown)  │                                                 │
│     └─────────────┘                                                │
│     ❌ LLM không biết đây là X-ray, MRI hay CT                     │
│     ❌ Có thể gọi tool không phù hợp                               │
│                                                                     │
│  2. FALSE POSITIVES (Dương Tính Giả)                               │
│     ┌─────────────┐      ┌─────────────┐                           │
│     │  Grounding  │──▶   │   5 boxes   │                           │
│     │    DINO     │      │  detected   │                           │
│     └─────────────┘      └─────────────┘                           │
│     ❌ Nhiều box là nhiễu (background, artifacts)                  │
│     ❌ Không có cơ chế kiểm chứng                                  │
│                                                                     │
│  3. GỌI TOOL THỪA THÃI                                             │
│     Histopathology image ──▶ Grounding DINO ──▶ ❌ Không hiệu quả  │
│     (Ảnh mô học không cần detection, chỉ cần classification)       │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2. Tại Sao Cần Triage?

Trong y tế thực tế, **Triage** (Phân loại) là bước đầu tiên quan trọng:

```
Bệnh nhân ──▶ [Triage Nurse] ──▶ Khoa phù hợp ──▶ Bác sĩ chuyên khoa
```

Tương tự, với hình ảnh y tế:

```
Medical Image ──▶ [Visual Triage] ──▶ Recommended Tools ──▶ Specialist AI
```

---

## 3. Giải Pháp Đề Xuất

### 3.1. Intelligent Triage Pipeline

TriMedAgent áp dụng chiến lược **"Chia để trị"** qua 3 giai đoạn:

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    TRIMEDAGENT: INTELLIGENT TRIAGE PIPELINE              │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  INPUT: Medical Image + User Question                                    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  📍 STEP 1: PERCEPTION (Nhận thức)                              │    │
│  │  ┌─────────────┐                                                │    │
│  │  │  BiomedCLIP │ ──▶ Modality: "Chest X-ray" (95%)             │    │
│  │  │   Triage    │ ──▶ Recommended: [grounding_dino, medsam]     │    │
│  │  └─────────────┘ ──▶ Skip Detection: False                     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  🧠 STEP 2: REASONING (Suy luận)                                │    │
│  │                                                                  │    │
│  │  Original Prompt: "Detect abnormalities in this image"          │    │
│  │                              │                                   │    │
│  │                              ▼                                   │    │
│  │  ┌───────────────────────────────────────────────────────────┐  │    │
│  │  │ [System Context: Image is Chest X-ray with 95% conf.      │  │    │
│  │  │  Act as a specialist. Recommended tools: grounding_dino]  │  │    │
│  │  │                                                            │  │    │
│  │  │ Detect abnormalities in this image                        │  │    │
│  │  └───────────────────────────────────────────────────────────┘  │    │
│  │                              │                                   │    │
│  │                              ▼                                   │    │
│  │  LLaVA-Med ──▶ Tool Call: grounding_dino(prompts=["opacity"]) │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  🔧 TOOL EXECUTION                                              │    │
│  │                                                                  │    │
│  │  Grounding DINO ──▶ 4 boxes detected                            │    │
│  │  [box1: tumor?, box2: noise, box3: lesion?, box4: artifact]    │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  ┌─────────────────────────────────────────────────────────────────┐    │
│  │  🛡️ STEP 3: GATEKEEPING (Kiểm chứng)                           │    │
│  │                                                                  │    │
│  │  For each detected box:                                         │    │
│  │  ┌─────────┐     ┌─────────────┐     ┌─────────────────────┐   │    │
│  │  │  Crop   │ ──▶ │ BiomedCLIP  │ ──▶ │ "Pathological" 85%  │   │    │
│  │  │  Box 1  │     │  Verifier   │     │      ✅ KEEP        │   │    │
│  │  └─────────┘     └─────────────┘     └─────────────────────┘   │    │
│  │  ┌─────────┐     ┌─────────────┐     ┌─────────────────────┐   │    │
│  │  │  Crop   │ ──▶ │ BiomedCLIP  │ ──▶ │ "Normal" 72%        │   │    │
│  │  │  Box 2  │     │  Verifier   │     │      ❌ DROP        │   │    │
│  │  └─────────┘     └─────────────┘     └─────────────────────┘   │    │
│  │                                                                  │    │
│  │  Result: 4 boxes ──▶ 2 verified boxes (50% false positive)     │    │
│  └─────────────────────────────────────────────────────────────────┘    │
│         │                                                                │
│         ▼                                                                │
│  OUTPUT: Verified detections + Natural language response                │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 3.2. Ba Đóng Góp Chính

| # | Đóng góp | Mô tả |
|---|----------|-------|
| 1 | **Structured Agent** | Chuyển Agent từ trạng thái tự do sang có kiểm soát |
| 2 | **Triage & Gatekeeper** | BiomedCLIP đóng vai "Bác sĩ phân loại" và "Người kiểm tra" |
| 3 | **Conditional Execution** | Giảm thiểu gọi tool thừa thãi |

---

## 4. Kiến Trúc Hệ Thống

### 4.1. High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           TRIMEDAGENT SYSTEM                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│     ┌──────────────┐                                                        │
│     │    User      │                                                        │
│     │   Browser    │                                                        │
│     └──────┬───────┘                                                        │
│            │ HTTP                                                           │
│            ▼                                                                │
│     ┌──────────────┐         ┌──────────────┐                              │
│     │   Gradio     │ ──────▶ │  Controller  │                              │
│     │   Web UI     │         │   :20001     │                              │
│     │   :7860      │         └──────┬───────┘                              │
│     └──────────────┘                │                                       │
│                                     │ Load Balancing                        │
│            ┌────────────────────────┼────────────────────────┐             │
│            │                        │                        │             │
│            ▼                        ▼                        ▼             │
│     ┌──────────────┐         ┌──────────────┐         ┌──────────────┐    │
│     │   LLaVA-Med  │         │  BiomedCLIP  │         │   Tool       │    │
│     │    Worker    │         │   Worker     │         │  Workers     │    │
│     │   :40000     │         │   :21006     │         │ :40001-N     │    │
│     └──────────────┘         └──────────────┘         └──────────────┘    │
│            │                        │                        │             │
│            │                        │                        │             │
│     ┌──────▼────────────────────────▼────────────────────────▼──────┐     │
│     │                        GPU MEMORY                              │     │
│     │  ┌─────────┐  ┌─────────────┐  ┌────────┐  ┌────────────┐    │     │
│     │  │ LLaMA   │  │ BiomedCLIP  │  │ DINO   │  │   MedSAM   │    │     │
│     │  │  7B     │  │  ViT-B/16   │  │        │  │            │    │     │
│     │  └─────────┘  └─────────────┘  └────────┘  └────────────┘    │     │
│     └───────────────────────────────────────────────────────────────┘     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.2. Data Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW                                       │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  1. USER INPUT                                                              │
│     Image (Base64) + Question (Text)                                        │
│                    │                                                        │
│                    ▼                                                        │
│  2. TRIAGE (BiomedCLIP)                                                     │
│     ┌──────────────────────────────────────────┐                           │
│     │ Input:  Image Base64                      │                           │
│     │ Labels: ["Chest X-ray", "Brain MRI", ...] │                           │
│     │ Output: {                                 │                           │
│     │   "modality": "Chest X-ray",              │                           │
│     │   "confidence": 0.95,                     │                           │
│     │   "recommended_tools": ["dino", "medsam"] │                           │
│     │ }                                         │                           │
│     └──────────────────────────────────────────┘                           │
│                    │                                                        │
│                    ▼                                                        │
│  3. CONTEXT INJECTION                                                       │
│     Original: "Detect lesions"                                              │
│     Enhanced: "[Context: Chest X-ray, 95%] Detect lesions"                 │
│                    │                                                        │
│                    ▼                                                        │
│  4. LLaVA-Med INFERENCE                                                     │
│     ┌──────────────────────────────────────────┐                           │
│     │ Output: {                                 │                           │
│     │   "thoughts": "Need detection model",     │                           │
│     │   "actions": [{                           │                           │
│     │     "API_name": "grounding_dino",         │                           │
│     │     "API_params": {"prompts": ["opacity"]}│                           │
│     │   }]                                      │                           │
│     │ }                                         │                           │
│     └──────────────────────────────────────────┘                           │
│                    │                                                        │
│                    ▼                                                        │
│  5. TOOL EXECUTION (Grounding DINO)                                        │
│     Output: {"boxes": [[0.2,0.3,0.5,0.6], ...], "phrases": ["opacity"]}    │
│                    │                                                        │
│                    ▼                                                        │
│  6. GATEKEEPER VERIFICATION                                                │
│     ┌──────────────────────────────────────────┐                           │
│     │ For each box:                             │                           │
│     │   crop = image.crop(box)                  │                           │
│     │   result = BiomedCLIP.classify(crop,      │                           │
│     │     ["Pathological finding", "Normal"])   │                           │
│     │   if result.confidence > 0.6:             │                           │
│     │     KEEP box                              │                           │
│     │   else:                                   │                           │
│     │     DROP box                              │                           │
│     └──────────────────────────────────────────┘                           │
│                    │                                                        │
│                    ▼                                                        │
│  7. FINAL OUTPUT                                                           │
│     Verified boxes + Annotated image + Natural language response           │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4.3. File Structure

```
MMedAgent-v2/
├── 📄 README_TRIMEDAGENT.md      # Tài liệu này
├── 📓 demo_trimedagent.ipynb     # Demo notebook
│
├── serve/                         # Tool Workers
│   ├── 📄 labels.json            # ⭐ Cấu hình TriMedAgent
│   ├── 📄 biomedclip_worker.py   # ⭐ Triage & Gatekeeper
│   ├── 📄 controller.py          # Điều phối workers
│   ├── 📄 grounding_dino_worker.py
│   ├── 📄 MedSAM_worker.py
│   └── ...
│
├── llava/
│   ├── serve/
│   │   ├── 📄 gradio_web_server_mmedagent.py  # ⭐ Main pipeline
│   │   ├── 📄 model_worker.py                  # LLaVA-Med worker
│   │   └── ...
│   ├── model/                     # LLaVA model code
│   ├── train/                     # Training scripts
│   └── eval/                      # Evaluation scripts
│
├── docs/                          # Documentation
│   ├── PIPELINE_GUIDE.md
│   ├── AGENT_ORCHESTRATION_GUIDE.md
│   └── ...
│
└── data_json/                     # Training/Eval data
    └── ...
```

---

## 5. Chi Tiết Kỹ Thuật

### 5.1. Step 1: Visual Triage

**Mục đích**: Xác định loại hình ảnh y tế để chọn tool phù hợp.

**Thuật toán**:
```python
def run_visual_triage(image, labels):
    """
    Input:  image (PIL.Image), labels (List[str])
    Output: {"modality": str, "confidence": float, "recommended_tools": List}
    """
    # 1. Encode image với BiomedCLIP
    image_features = biomedclip.encode_image(image)
    
    # 2. Encode text labels
    text_features = biomedclip.encode_text([f"this is a {l}" for l in labels])
    
    # 3. Tính cosine similarity
    similarities = image_features @ text_features.T
    probs = softmax(similarities * 100)  # Temperature scaling
    
    # 4. Lấy top prediction
    top_idx = argmax(probs)
    return {
        "modality": labels[top_idx],
        "confidence": probs[top_idx],
        "recommended_tools": TOOL_MAPPING[labels[top_idx]]
    }
```

**Supported Modalities**:
- Chest X-ray
- Brain MRI
- Abdominal CT
- Lung CT
- Histopathology
- Dermoscopy
- Ultrasound
- Bone X-ray
- Retinal fundus
- Mammography
- Gross pathology

### 5.2. Step 2: Context Injection

**Mục đích**: Cung cấp ngữ cảnh cho LLaVA-Med để ra quyết định chính xác hơn.

**Template**:
```python
# High confidence (>= 50%)
context = """[System Context: The input image is identified as {modality} 
with {confidence:.0%} confidence. Act as a specialist for this modality. 
Recommended tools: {tools}.]"""

# Low confidence (< 50%)
context = """[System Context: Image modality unclear 
(best guess: {modality}, conf: {confidence:.0%}). 
Proceed with general medical analysis.]"""

# Final prompt
enhanced_prompt = context + "\n\n" + original_user_question
```

### 5.3. Step 3: Gatekeeper Verification

**Mục đích**: Lọc bỏ false positives từ detection results.

**Thuật toán**:
```python
def verify_boxes_with_gatekeeper(image, boxes, target_entity):
    """
    Input:  image, boxes (List[x1,y1,x2,y2]), target_entity (str)
    Output: {"verified_boxes": List, "rejected_indices": List}
    """
    verified = []
    rejected = []
    
    # Labels cho binary classification
    pos_label = f"Pathological finding of {target_entity}"
    neg_label = "Normal tissue, background noise"
    
    for i, box in enumerate(boxes):
        # 1. Crop vùng ảnh
        crop = image.crop(box_to_pixels(box))
        
        # 2. Classify với BiomedCLIP
        result = biomedclip.classify(crop, [pos_label, neg_label])
        
        # 3. Kiểm tra ngưỡng
        if result["prediction"] == pos_label and result["confidence"] >= 0.6:
            verified.append(box)
        else:
            rejected.append(i)
    
    return {"verified_boxes": verified, "rejected_indices": rejected}
```

**Lưu ý quan trọng**: Gatekeeper đồng bộ lọc cả `boxes`, `logits`, `phrases`, và `masks_rle` để đảm bảo tính nhất quán.

### 5.4. Conditional Execution

**Mục đích**: Bỏ qua tools không phù hợp với loại ảnh.

**Logic**:
```python
# Ảnh Histopathology/Dermoscopy → Không cần detection
if modality in ["Histopathology", "Dermoscopy"]:
    skip_detection = True
    recommended_tools = ["biomedclip"]  # Chỉ cần classification

# Ảnh X-ray/MRI/CT → Cần detection + segmentation
if modality in ["Chest X-ray", "Brain MRI", "CT"]:
    skip_detection = False
    recommended_tools = ["grounding_dino", "medsam"]
```

---

## 6. Cài Đặt

### 6.1. Yêu Cầu Hệ Thống

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| GPU | RTX 3060 (12GB) | RTX 3090 (24GB) |
| RAM | 16GB | 32GB |
| Storage | 50GB | 100GB |
| CUDA | 11.7+ | 12.0+ |
| Python | 3.10 | 3.10 |

### 6.2. Cài Đặt Môi Trường

```bash
# 1. Clone repository
git clone https://github.com/YOUR_REPO/MMedAgent-v2.git
cd MMedAgent-v2

# 2. Tạo conda environment
conda create -n trimedagent python=3.10 -y
conda activate trimedagent

# 3. Cài đặt dependencies
pip install --upgrade pip
pip install -e .

# 4. Cài đặt thêm cho training (optional)
pip install -e ".[train]"
pip install flash-attn --no-build-isolation

# 5. Cài đặt BiomedCLIP
pip install open_clip_torch
```

### 6.3. Download Models

```bash
# 1. Download MMedAgent checkpoint (LoRA weights)
git lfs install
git clone https://huggingface.co/andy0207/mmedagent

# 2. Download LLaMA-7B base model
# Theo hướng dẫn tại: https://huggingface.co/docs/transformers/main/model_doc/llama

# 3. Apply delta weights
python3 -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target ./base_model \
    --delta /path/to/llava_med_delta_weights

# 4. BiomedCLIP (tự động download khi chạy)
# Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

---

## 7. Hướng Dẫn Sử Dụng

### 7.1. Khởi Động Server

**Terminal 1: Controller**
```bash
python -m serve.controller --host 0.0.0.0 --port 20001
```

**Terminal 2: LLaVA-Med Worker**
```bash
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --port 40000 \
    --controller-address http://localhost:20001 \
    --model-path ./base_model \
    --model-name "llava-med"
```

**Terminal 3: BiomedCLIP Worker (Triage + Gatekeeper)**
```bash
python -m serve.biomedclip_worker \
    --host 0.0.0.0 \
    --port 21006 \
    --controller-address http://localhost:20001
```

**Terminal 4: Grounding DINO Worker**
```bash
python -m serve.grounding_dino_worker \
    --host 0.0.0.0 \
    --port 40001 \
    --controller-address http://localhost:20001
```

**Terminal 5: MedSAM Worker**
```bash
python -m serve.MedSAM_worker \
    --host 0.0.0.0 \
    --port 40002 \
    --controller-address http://localhost:20001
```

**Terminal 6: Web UI**
```bash
python -m llava.serve.gradio_web_server_mmedagent \
    --controller-url http://localhost:20001 \
    --port 7860
```

### 7.2. Truy Cập Web UI

Mở browser và truy cập: **http://localhost:7860**

### 7.3. Sử Dụng Demo Notebook

```bash
# Mở Jupyter
jupyter notebook demo_trimedagent.ipynb

# Hoặc trong VS Code
# Mở file demo_trimedagent.ipynb và chạy từng cell
```

---

## 8. Chatbot - Tính Năng Chính

> 🎯 **Mục đích chính của TriMedAgent là CHATBOT y tế thông minh** - cho phép người dùng gửi ảnh y tế và hỏi đáp về ảnh đó trong phiên chat.

### 8.1. Kiến Trúc Chatbot

```
┌────────────────────────────────────────────────────────────────────┐
│                    TRIMEDAGENT CHATBOT FLOW                        │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  ┌──────────────┐                                                  │
│  │ User uploads │                                                  │
│  │    image     │                                                  │
│  └──────┬───────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ╔══════════════════════════════════════════════════════════════╗  │
│  ║  STAGE 1: AUTO TRIAGE (BiomedCLIP)                          ║  │
│  ║  • Phân loại loại ảnh: X-ray, MRI, CT, ...                  ║  │
│  ║  • Recommend tools phù hợp                                   ║  │
│  ║  • Context injection                                         ║  │
│  ╚══════════════════════════════════════════════════════════════╝  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────┐                                                  │
│  │ Session Ready │  ◄── User có thể chat nhiều lượt             │
│  └──────┬───────┘                                                  │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                    CONVERSATION LOOP                          │  │
│  │  ┌───────────────────────────────────────────────────────┐   │  │
│  │  │ User: "Can you detect any abnormalities?"             │   │  │
│  │  └───────────────────────────────────────────────────────┘   │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ╔═══════════════════════════════════════════════════════╗   │  │
│  │  ║  STAGE 2: INTENT DETECTION                            ║   │  │
│  │  ║  • Detection intent → Grounding DINO                  ║   │  │
│  │  ║  • Segmentation intent → MedSAM                       ║   │  │
│  │  ║  • Classification intent → BiomedCLIP                 ║   │  │
│  │  ║  • Description intent → LLaVA-Med                     ║   │  │
│  │  ╚═══════════════════════════════════════════════════════╝   │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ╔═══════════════════════════════════════════════════════╗   │  │
│  │  ║  STAGE 3: GATEKEEPER (if detection)                   ║   │  │
│  │  ║  • Verify mỗi detection bằng BiomedCLIP               ║   │  │
│  │  ║  • Loại bỏ false positives                            ║   │  │
│  │  ╚═══════════════════════════════════════════════════════╝   │  │
│  │         │                                                     │  │
│  │         ▼                                                     │  │
│  │  ┌───────────────────────────────────────────────────────┐   │  │
│  │  │ Agent: "Found 3 regions, 2 verified as abnormalities" │   │  │
│  │  └───────────────────────────────────────────────────────┘   │  │
│  │                                                               │  │
│  │  ──────────── Loop continues ─────────────                   │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

### 8.2. Tính Năng Chatbot

| Tính Năng | Mô Tả |
|-----------|-------|
| **Session Management** | Mỗi phiên chat độc lập, có context riêng |
| **Auto Triage** | Tự động phân loại ảnh khi upload |
| **Multi-turn Chat** | Hỗ trợ nhiều lượt hỏi đáp trong 1 phiên |
| **Intent Detection** | Nhận diện yêu cầu từ câu hỏi (detect, segment, classify) |
| **Tool Routing** | Tự động chọn công cụ phù hợp với intent |
| **History Tracking** | Lưu lịch sử chat trong phiên |
| **Image Switching** | Có thể đổi ảnh giữa phiên, tự động triage lại |

### 8.3. Sử Dụng Chatbot

#### Option 1: Gradio UI (Recommended)

```python
# Trong notebook demo_trimedagent.ipynb

# Khởi tạo agent
agent = TriMedAgent(model, preprocess, tokenizer, config, device)

# Tạo và chạy Gradio UI
chatbot_ui = GradioChatbot(agent)
demo = chatbot_ui.create_ui()
demo.launch(share=True)  # Tạo public link
```

Truy cập: `http://localhost:7860` hoặc link public

#### Option 2: Python API

```python
# Bắt đầu phiên chat mới
agent.start_new_session(image)  # Tự động chạy Triage

# Chat nhiều lượt
response1 = agent.chat("What type of image is this?")
response2 = agent.chat("Can you detect any abnormalities?")
response3 = agent.chat("Please segment the detected regions")

# Đổi ảnh (tự động triage lại)
agent.set_image(new_image)

# Xóa phiên
agent.clear_session()
```

#### Option 3: Console Demo

```python
# Demo console có sẵn trong notebook
# Chạy cell "CHATBOT DEMO - Console Version"
```

### 8.4. Các Intent Được Hỗ Trợ

| Intent | Trigger Words | Tool Used |
|--------|---------------|-----------|
| **Detection** | detect, find, locate, where, tìm, phát hiện | Grounding DINO + Gatekeeper |
| **Segmentation** | segment, outline, boundary, phân vùng | MedSAM |
| **Classification** | what is, diagnose, classify, chẩn đoán | BiomedCLIP |
| **Description** | describe, explain, tell me, mô tả | LLaVA-Med |
| **General** | (default) | Context-aware response |

### 8.5. Ví Dụ Conversation

```
🆕 NEW CHAT SESSION STARTED
📷 Image detected. Running automatic triage...

🔬 STEP 1: VISUAL TRIAGE (Perception)
   • Detected Modality: Chest X-ray
   • Confidence: 89.2%
   • Recommended Tools: [grounding_dino, medsam]

👤 User: What do you see in this image?

🤖 Agent: This is a Chest X-ray image with 89% confidence. 
   I can help you with:
   - Detection of abnormalities
   - Segmentation of regions
   - Description of findings

👤 User: Can you detect any abnormalities?

🛡️ STEP 3: GATEKEEPER VERIFICATION
   Box 0: ✅ KEEP (conf: 78%)
   Box 1: ❌ DROP (conf: 23%)
   Box 2: ✅ KEEP (conf: 85%)

🤖 Agent: 🔍 Detection Results for Chest X-ray
   Raw Detections: 3 regions found
   After Gatekeeper: 2 verified regions
   
   Verified Findings:
   • Region 1: [0.25, 0.30, 0.45, 0.70] - Left lung area
   • Region 2: [0.35, 0.25, 0.65, 0.45] - Upper region

👤 User: Please segment these regions

🤖 Agent: 🎯 Segmentation Results
   Tool Used: MedSAM
   Segmented Regions: 2
   [Mask visualization would appear here]
```

---

## 9. Cấu Hình

### 8.1. File Cấu Hình: `serve/labels.json`

```json
{
  "triage_labels": [
    "Chest X-ray",
    "Brain MRI",
    "Abdominal CT",
    "Histopathology",
    "Ultrasound",
    "Dermoscopy",
    "Gross pathology",
    "Bone X-ray",
    "Lung CT",
    "Retinal fundus",
    "Mammography"
  ],
  
  "gatekeeper_prompts": {
    "positive": "Pathological finding, lesion, tumor, abnormality",
    "negative": "Normal tissue, healthy anatomy, background noise, blurry area"
  },
  
  "thresholds": {
    "triage_confidence": 0.5,
    "gatekeeper_confidence": 0.6,
    "low_confidence_fallback": 0.3
  },
  
  "conditional_execution": {
    "enabled": true,
    "skip_detection_modalities": ["Histopathology", "Dermoscopy"],
    "require_segmentation_modalities": ["Chest X-ray", "Brain MRI", "Abdominal CT"],
    "tool_mapping": {
      "Chest X-ray": ["grounding_dino", "medsam"],
      "Brain MRI": ["grounding_dino", "medsam"],
      "Histopathology": ["biomedclip"],
      "default": ["grounding_dino"]
    }
  },
  
  "prompt_templates": {
    "triage_context": "[System Context: The input image is identified as {modality} with {confidence:.0%} confidence. Act as a specialist. Recommended tools: {tools}.]",
    "low_confidence_context": "[System Context: Image modality unclear (best guess: {modality}, conf: {confidence:.0%}). Proceed with general analysis.]"
  }
}
```

### 8.2. Các Tham Số Quan Trọng

| Tham số | Mặc định | Mô tả |
|---------|----------|-------|
| `triage_confidence` | 0.5 | Ngưỡng để tin tưởng kết quả triage |
| `gatekeeper_confidence` | 0.6 | Ngưỡng để giữ lại detection box |
| `skip_detection_modalities` | [...] | Các modality không cần detection |
| `tool_mapping` | {...} | Map modality → recommended tools |

---

## 9. Demo & Thử Nghiệm

### 9.1. Quick Test với Python

```python
import torch
import open_clip
from PIL import Image

# Load model
model, preprocess = open_clip.create_model_from_pretrained(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
model.eval()

# Test triage
image = Image.open("test_xray.jpg")
labels = ["Chest X-ray", "Brain MRI", "CT scan", "Histopathology"]

image_tensor = preprocess(image).unsqueeze(0)
texts = tokenizer([f"this is a {l}" for l in labels])

with torch.no_grad():
    image_features = model.encode_image(image_tensor)
    text_features = model.encode_text(texts)
    
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
    
print("Triage Results:")
for label, prob in zip(labels, probs[0]):
    print(f"  {label}: {prob:.2%}")
```

### 9.2. Expected Output

```
🔬 STEP 1: VISUAL TRIAGE (Perception)
============================================================

📊 Triage Result:
   • Detected Modality: Chest X-ray
   • Confidence: 94.52%
   • Recommended Tools: ['grounding_dino', 'medsam']
   • Skip Detection: False

💉 Context Prompt:
   [System Context: Image identified as Chest X-ray (95% confidence)...]

📈 Top-5 Predictions:
   1. Chest X-ray: 94.52% ██████████████████
   2. Bone X-ray: 3.21% █
   3. Lung CT: 1.45% 
   4. Mammography: 0.52%
   5. Brain MRI: 0.15%
```

---

## 10. Đóng Góp

### 10.1. Các Cải Tiến So Với MMedAgent Gốc

| Feature | MMedAgent | TriMedAgent |
|---------|-----------|-------------|
| Visual Triage | ❌ | ✅ BiomedCLIP-based |
| Context Injection | ❌ | ✅ Dynamic prompts |
| Gatekeeper | ❌ | ✅ False positive filtering |
| Conditional Execution | ❌ | ✅ Skip unnecessary tools |
| Configurable Thresholds | ❌ | ✅ via labels.json |
| Synchronized Filtering | ❌ | ✅ boxes, logits, phrases, masks |

### 10.2. Các File Đã Thay Đổi

| File | Thay đổi |
|------|----------|
| `serve/labels.json` | Thêm thresholds, conditional_execution, templates |
| `serve/biomedclip_worker.py` | Hỗ trợ dynamic labels |
| `llava/serve/gradio_web_server_mmedagent.py` | Thêm pipeline logic |
| `demo_trimedagent.ipynb` | **MỚI** - Demo notebook |

### 10.3. Đóng Góp Code

```bash
# 1. Fork repository
# 2. Tạo branch mới
git checkout -b feature/your-feature

# 3. Commit changes
git commit -m "Add: your feature description"

# 4. Push và tạo Pull Request
git push origin feature/your-feature
```

---

## 11. Tài Liệu Tham Khảo

### 11.1. Papers

1. **MMedAgent**: Li et al., "MMedAgent: Learning to Use Medical Tools with Multi-modal Agent", EMNLP 2024
2. **LLaVA-Med**: Li et al., "LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine in One Day"
3. **BiomedCLIP**: Zhang et al., "BiomedCLIP: A Multimodal Biomedical Foundation Model"
4. **Grounding DINO**: Liu et al., "Grounding DINO: Marrying DINO with Grounded Pre-Training"
5. **MedSAM**: Ma et al., "Segment Anything in Medical Images"

### 11.2. Repositories

- [MMedAgent](https://github.com/Wangyixinxin/MMedAgent)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [Grounding DINO](https://github.com/IDEA-Research/GroundingDINO)
- [MedSAM](https://github.com/bowang-lab/MedSAM)

### 11.3. Documentation

- [Pipeline Guide](docs/PIPELINE_GUIDE.md)
- [Agent Orchestration Guide](docs/AGENT_ORCHESTRATION_GUIDE.md)
- [LLaVA Core Guide](docs/LLAVA_CORE_GUIDE.md)

---

## 📝 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **MMedAgent Team** for the original codebase
- **Microsoft Research** for BiomedCLIP and LLaVA-Med
- **IDEA Research** for Grounding DINO
- **Segment Anything Team** for MedSAM

---

<div align="center">

**TriMedAgent** - Intelligent Triage Pipeline for Medical Image Analysis

Made with ❤️ for Healthcare AI

</div>
