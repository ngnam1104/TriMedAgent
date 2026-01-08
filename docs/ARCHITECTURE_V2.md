# 🏗️ TriMed-Agent V2 Architecture

> **State-based Artificial Cognitive System** for Medical Image Analysis

---

## 1. Tổng quan Kiến trúc (Architecture Overview)

TriMed-Agent V2 được thiết kế như một **hệ thống nhận thức nhân tạo** (Artificial Cognitive System), nơi các mô-đun chuyên biệt phối hợp dưới sự chỉ huy của một **bộ não trung tâm** (Planner).

### Khác biệt V1 vs V2

| Aspect | V1 (Linear) | V2 (State-based) |
|--------|-------------|------------------|
| **Luồng xử lý** | Một chiều (Pipeline) | Vòng lặp ReAct |
| **Định tuyến** | Cố định | Logic Router động |
| **Kiến thức** | Parametric (LLM) | Parametric + RAG |
| **Training** | Pretrained only | SFT + RL (GRPO) |
| **Output** | Free-form text | Structured JSON |

---

## 2. Sơ đồ Kiến trúc Hệ thống

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TriMed-Agent V2 System                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│    ┌──────────────┐                                                         │
│    │  User Input  │  Image (I) + Query (Q)                                  │
│    └──────┬───────┘                                                         │
│           │                                                                  │
│           ▼                                                                  │
│    ┌──────────────────────────────────────────┐                             │
│    │           LOGIC ROUTER                    │                             │
│    │  ┌─────────────────────────────────────┐ │                             │
│    │  │ Intent Classifier (DistilBERT)      │ │                             │
│    │  │ P(diagnosis|Q) > θ ?                │ │                             │
│    │  └─────────────────────────────────────┘ │                             │
│    └──────────────┬───────────────────────────┘                             │
│                   │                                                          │
│         ┌─────────┴─────────┐                                               │
│         │                   │                                               │
│         ▼                   ▼                                               │
│  ┌─────────────┐    ┌─────────────────────────────────────────────────┐    │
│  │ RAG MODULE  │    │              PLANNER (LLaVA-Med + LoRA)          │    │
│  │             │    │  ┌─────────────────────────────────────────────┐ │    │
│  │ PubMedBERT  │    │  │              ReAct Loop                     │ │    │
│  │     +       │    │  │  ┌─────────┐  ┌─────────┐  ┌─────────────┐ │ │    │
│  │   FAISS     │    │  │  │ Thought │→ │  Plan   │→ │   Action    │ │ │    │
│  │             │    │  │  └─────────┘  └─────────┘  └──────┬──────┘ │ │    │
│  │ Knowledge   │    │  │       ▲                          │        │ │    │
│  │   Base      │    │  │       │      ┌───────────────────┘        │ │    │
│  └──────┬──────┘    │  │       │      ▼                            │ │    │
│         │           │  │  ┌─────────────────┐                      │ │    │
│         │           │  │  │   Observation   │ ← Tool Results       │ │    │
│         │           │  │  └─────────────────┘                      │ │    │
│         │           │  └─────────────────────────────────────────────┘ │    │
│         │           └──────────────────────┬──────────────────────────┘    │
│         │                                  │                               │
│         │                                  ▼                               │
│         │           ┌─────────────────────────────────────────────────┐    │
│         │           │                  TOOLKIT                         │    │
│         │           │  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │    │
│         │           │  │BiomedCLIP │ │   DINO    │ │    MedSAM     │  │    │
│         │           │  │ (Triage)  │ │(Detection)│ │(Segmentation) │  │    │
│         │           │  └───────────┘ └───────────┘ └───────────────┘  │    │
│         │           │  ┌───────────┐ ┌───────────┐ ┌───────────────┐  │    │
│         │           │  │  ChatCAD  │ │   Zoom    │ │  Gatekeeper   │  │    │
│         │           │  │ (Report)  │ │ (SmartCrop)│ │ (Verify)     │  │    │
│         │           │  └───────────┘ └───────────┘ └───────────────┘  │    │
│         │           └─────────────────────────────────────────────────┘    │
│         │                                  │                               │
│         └──────────────────┬───────────────┘                               │
│                            ▼                                               │
│                  ┌─────────────────┐                                       │
│                  │  Final Response │                                       │
│                  └─────────────────┘                                       │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 3. Chi tiết các Module

### 3.1 Logic Router (Intent Classification)

**Mục đích**: Phân loại câu hỏi thành `Theory` (lý thuyết) hoặc `Diagnosis` (chẩn đoán).

```python
class LogicRouter:
    """
    Lightweight intent classifier using DistilBERT.
    
    Decision Logic:
    - P(diagnosis|Q) > θ (0.7) → Planner
    - Otherwise → RAG Module
    
    Override Rules:
    - "this image", "hình ảnh này" → Force Diagnosis
    - Spatial words (left, right, upper) → Force Diagnosis
    """
```

**Input**: Query text `Q`  
**Output**: `{"intent": "diagnosis" | "theory", "confidence": float}`

### 3.2 RAG Module (Knowledge Retrieval)

**Mục đích**: Trả lời câu hỏi lý thuyết bằng tri thức y khoa bên ngoài.

```python
class MedicalRAG:
    """
    Retrieval-Augmented Generation for medical knowledge.
    
    Components:
    - Embedding: PubMedBERT (biomedical domain)
    - Vector DB: FAISS / ChromaDB
    - Knowledge Base: PubMed, Merck Manual
    
    Pipeline:
    1. Encode query Q → vector v_q
    2. Retrieve top-k chunks from KB
    3. Inject context into LLM prompt
    4. Generate answer
    """
```

### 3.3 Planner (LLaVA-Med + LoRA)

**Mục đích**: Bộ não của hệ thống, sinh ra chuỗi suy luận và lệnh JSON.

```python
class Planner:
    """
    ReAct-style reasoning with structured JSON output.
    
    Output Format:
    {
        "thought": "Reasoning about current state",
        "action": "ToolName",
        "action_input": {"param": "value"}
    }
    
    Termination:
    {
        "thought": "I have enough information",
        "action": "Final Answer",
        "action_input": {"answer": "..."}
    }
    """
```

### 3.4 ToolKit (Specialist Models)

| Tool | Model | Purpose | Input | Output |
|------|-------|---------|-------|--------|
| `BiomedCLIP` | microsoft/BiomedCLIP | Image Triage | Image | Modality, Abnormality |
| `GroundingDINO` | IDEA-Research | Object Detection | Image + Prompt | Bounding Boxes |
| `MedSAM` | SAM-ViT-B | Segmentation | Image + Boxes | Masks |
| `ZoomProcessor` | Custom | Smart Crop | Image | Cropped Regions |
| `Gatekeeper` | LLaVA-Med | Verification | Image + Boxes | Valid/Invalid |
| `ChatCAD` | Custom | Report Gen | Findings | Medical Report |

---

## 4. Luồng Xử lý Thông tin (Information Flow)

### 4.1 Theory Query Flow

```
User: "Viêm phổi là gì?"
         │
         ▼
    Logic Router
    P(diagnosis) = 0.15 < 0.7
         │
         ▼
    RAG Module
    ├── Encode query → v_q
    ├── Search FAISS → top-5 chunks
    └── Generate answer with context
         │
         ▼
    Response: "Viêm phổi là tình trạng nhiễm trùng..."
```

### 4.2 Diagnosis Query Flow (ReAct Loop)

```
User: "Ảnh X-quang này có tổn thương phổi không?"
         │
         ▼
    Logic Router
    P(diagnosis) = 0.92 > 0.7
         │
         ▼
    Planner (ReAct Loop)
    
    ┌── Step 1 ──────────────────────────────────────┐
    │ Thought: "Cần xác định loại ảnh trước"         │
    │ Action: BiomedCLIP                             │
    │ Action_Input: {"task": "classify"}             │
    └────────────────────────────────────────────────┘
         │
         ▼
    Observation: {"modality": "X-ray", "confidence": 0.95}
    
    ┌── Step 2 ──────────────────────────────────────┐
    │ Thought: "Đây là X-quang, cần tìm tổn thương"  │
    │ Action: GroundingDINO                          │
    │ Action_Input: {"prompt": "lung lesion"}        │
    └────────────────────────────────────────────────┘
         │
         ▼
    Observation: {"boxes": [[100, 200, 150, 250]], "scores": [0.87]}
    
    ┌── Step 3 ──────────────────────────────────────┐
    │ Thought: "Tìm thấy 1 vùng, cần xác minh"       │
    │ Action: Gatekeeper                             │
    │ Action_Input: {"boxes": [...], "target": "lesion"} │
    └────────────────────────────────────────────────┘
         │
         ▼
    Observation: {"verified": true, "reason": "Opacity confirmed"}
    
    ┌── Step 4 ──────────────────────────────────────┐
    │ Thought: "Đã xác nhận tổn thương, trả lời"     │
    │ Action: Final Answer                           │
    │ Action_Input: {"answer": "Có 1 vùng mờ..."}   │
    └────────────────────────────────────────────────┘
```

---

## 5. Chiến lược Huấn luyện (Training Strategy)

### 5.1 Stage 1: Supervised Fine-Tuning (SFT)

**Mục tiêu**: Dạy LLaVA-Med sinh JSON có cấu trúc.

```
┌─────────────────────────────────────────────────────────┐
│                    SFT Pipeline                          │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐  │
│  │  VQA-RAD     │    │   Teacher    │    │  ReAct    │  │
│  │  SLAKE       │ →  │   (GPT-4)    │ →  │  Dataset  │  │
│  │  PathVQA     │    │  Transform   │    │  (JSON)   │  │
│  └──────────────┘    └──────────────┘    └─────┬─────┘  │
│                                                │         │
│                                                ▼         │
│  ┌──────────────────────────────────────────────────┐   │
│  │              LoRA Fine-tuning                     │   │
│  │  • Rank (r): 64                                   │   │
│  │  • Alpha (α): 128                                 │   │
│  │  • Target: q_proj, v_proj, k_proj, o_proj        │   │
│  │  • Loss: CrossEntropy (masked)                   │   │
│  └──────────────────────────────────────────────────┘   │
│                           │                              │
│                           ▼                              │
│                  ┌─────────────────┐                    │
│                  │  LoRA Adapter   │                    │
│                  │   (SFT v1.0)    │                    │
│                  └─────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

**Data Format**:
```json
{
  "image": "path/to/xray.jpg",
  "conversations": [
    {"role": "user", "content": "<image>\nTim có to không?"},
    {"role": "assistant", "content": "{\"thought\": \"Cần đo tỷ lệ tim/lồng ngực\", \"action\": \"GroundingDINO\", \"action_input\": {\"prompt\": \"heart\"}}"}
  ]
}
```

### 5.2 Stage 2: Reinforcement Learning (GRPO)

**Mục tiêu**: Tối ưu hóa chiến lược sử dụng công cụ.

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Pipeline                         │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐                                       │
│  │ SFT Adapter  │                                       │
│  │  (v1.0)      │                                       │
│  └──────┬───────┘                                       │
│         │                                               │
│         ▼                                               │
│  ┌─────────────────────────────────────────────────┐   │
│  │              GRPO Training Loop                  │   │
│  │                                                  │   │
│  │  for each prompt:                               │   │
│  │    1. Sample G outputs from policy              │   │
│  │    2. Execute each output in RLEnv              │   │
│  │    3. Compute rewards R_1, R_2, ..., R_G        │   │
│  │    4. Baseline b = mean(R)                      │   │
│  │    5. Advantage A_i = R_i - b                   │   │
│  │    6. Update policy with clipped objective      │   │
│  │                                                  │   │
│  └─────────────────────────────────────────────────┘   │
│                           │                              │
│                           ▼                              │
│                  ┌─────────────────┐                    │
│                  │  LoRA Adapter   │                    │
│                  │   (RL v1.0)     │                    │
│                  └─────────────────┘                    │
└─────────────────────────────────────────────────────────┘
```

**Reward Function**:

$$R_{total} = \alpha \cdot R_{IoU} + \beta \cdot R_{Acc} + \gamma \cdot R_{Format} + \delta \cdot R_{Step}$$

| Component | Weight | Description |
|-----------|--------|-------------|
| $R_{IoU}$ | 0.3 | IoU với ground truth boxes |
| $R_{Acc}$ | 0.4 | Độ chính xác câu trả lời |
| $R_{Format}$ | 0.2 | JSON hợp lệ (+1) / Sai (-1) |
| $R_{Step}$ | 0.1 | Penalty mỗi bước (-0.1) |

### 5.3 Curriculum Learning (3 Levels)

```
Level 1 (Imitation)     Level 2 (Verification)    Level 3 (Reasoning)
─────────────────────   ─────────────────────────  ─────────────────────
• Tool always correct   • Tool có noise           • Tool có thể fail
• Learn to trust tools  • Learn Gatekeeper        • Learn fallback
• Simple cases          • Medium complexity       • Complex cases
```

---

## 6. Deployment Architecture

### HuggingFace Hub Structure

```
ngnam1104/TriMedAgent-V2/
├── base_model/           # LLaVA-Med-7B (pointer)
├── adapters/
│   ├── sft-v1.0/         # SFT LoRA adapter
│   ├── rl-v1.0/          # RL LoRA adapter  
│   └── merged-v1.0/      # Merged full model
├── configs/
│   ├── lora_config.json
│   └── training_args.json
└── README.md
```

### Inference Pipeline

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base + adapter
base = AutoModelForCausalLM.from_pretrained("llava-med-7b")
model = PeftModel.from_pretrained(base, "ngnam1104/TriMedAgent-V2/adapters/rl-v1.0")

# Or load merged
model = AutoModelForCausalLM.from_pretrained("ngnam1104/TriMedAgent-V2/adapters/merged-v1.0")
```

---

## 7. File Structure

```
TriMedAgent/
├── src/
│   ├── core/
│   │   ├── orchestrator.py      # Main ReAct loop
│   │   ├── logic_router.py      # Intent classification
│   │   └── state.py             # Agent state management
│   ├── modules/
│   │   ├── brain/
│   │   │   ├── planner.py       # LLaVA-Med Planner
│   │   │   └── llava_brain.py   # Base LLaVA wrapper
│   │   ├── knowledge/
│   │   │   ├── rag_engine.py    # RAG with PubMedBERT
│   │   │   └── knowledge_base.py
│   │   ├── vision/
│   │   │   ├── biomed_clip.py
│   │   │   ├── dino_detector.py
│   │   │   └── zoom_processor.py
│   │   └── segmentation/
│   │       └── medsam_wrapper.py
│   └── training/
│       ├── sft_trainer.py       # SFT with LoRA
│       ├── grpo_trainer.py      # GRPO RL
│       └── reward_functions.py
├── scripts/
│   ├── train_sft.py
│   ├── train_rl_grpo.py
│   └── merge_adapters.py
├── configs/
│   ├── config.yaml
│   ├── sft_config.yaml
│   └── rl_config.yaml
└── data/
    ├── sft_dataset/
    └── rl_dataset/
```

---

## 8. References

1. ReAct: Synergizing Reasoning and Acting in Language Models (Yao et al., 2022)
2. LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
3. GRPO: Group Relative Policy Optimization (DeepSeek, 2024)
4. LLaVA-Med: Large Language and Vision Assistant for Biomedicine
5. PubMedBERT: Domain-Specific Language Models for Biomedical NLP
