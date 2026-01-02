# 📂 Giải Thích Cấu Trúc File TriMedAgent

## 🌳 Tổng Quan Cây Thư Mục

```
TriMedAgent/
│
├── 📄 README.md                           # Tài liệu chính (tiếng Việt)
├── 📄 QUICKSTART.md                       # Hướng dẫn nhanh
├── 📄 LICENSE                             # Apache 2.0 License
├── 📄 pyproject.toml                      # Python package configuration
├── 📄 cog.yaml                            # Cog (Replicate) configuration
│
├── 📓 demo_trimedagent_orchestrator.ipynb # ⭐ DEMO CHÍNH
├── 📓 instruction_generation.ipynb        # Notebook sinh dữ liệu huấn luyện
│
├── 📁 src/                                # ⭐ CORE BUSINESS LOGIC
├── 📁 serve/                              # ⭐ WORKER SERVERS
├── 📁 llava/                              # LLaVA-Med model code
├── 📁 docs/                               # Documentation
├── 📁 data_json/                          # Training/Eval data
├── 📁 data_processing/                    # Data processing scripts
├── 📁 scripts/                            # Utility scripts
└── 📁 images/                             # Sample images
```

---

## 🔥 CORE FILES (Quan Trọng Nhất)

### 1. `src/trimed_orchestrator.py` ⭐⭐⭐

**Vai trò:** Thin Client Orchestrator - "Bộ não" điều phối pipeline

**Chức năng:**
- Không load model, không cần GPU
- Gọi HTTP đến các Worker để thực hiện inference
- Quản lý luồng dữ liệu giữa 5 stages

**Classes & Functions:**

```python
# Data Classes (kết quả từ mỗi stage)
@dataclass
class TriageResult:        # Stage 1: Kết quả phân loại modality
class DetectionResult:     # Stage 3: Kết quả detection
class GatekeeperResult:    # Stage 4: Kết quả verify box
class SegmentationResult:  # Stage 5: Kết quả segmentation
class PipelineResult:      # Tổng hợp tất cả stages

# Main Class
class TriMedOrchestrator:
    def __init__(worker_urls, config, timeout, conv_template)
    
    # Private API Wrappers
    def _call_triage(image_b64) -> TriageResult
    def _call_llava(prompt, image_b64) -> str
    def _call_dino(query, image_b64) -> DetectionResult
    def _call_gatekeeper(image_b64, box, target_label) -> GatekeeperResult
    def _call_medsam(image_b64, boxes) -> SegmentationResult
    
    # Public Pipeline
    def run_full_chain(image, user_query) -> PipelineResult  # ⭐ Main entry
    
    # Convenience Methods
    def triage(image) -> TriageResult
    def ask_llava(image, question) -> str
    def detect(image, query) -> DetectionResult
    def segment(image, boxes) -> SegmentationResult
    def health_check() -> Dict[str, bool]
```

**Liên kết:**
```
trimed_orchestrator.py
     │
     ├──imports──▶ llava/conversation.py  (conv_templates để build prompt)
     │
     ├──reads──▶ serve/labels.json  (config: thresholds, labels)
     │
     └──HTTP calls──▶ Workers (21002, 21003, 21004, 21006)
```

---

### 2. `src/__init__.py`

**Vai trò:** Package exports - cho phép `from src import ...`

**Exports:**
```python
from src import (
    TriMedOrchestrator,    # Main class
    TriageResult,          # Data classes
    DetectionResult,
    GatekeeperResult,
    SegmentationResult,
    PipelineResult,
    image_to_base64,       # Utilities
    base64_to_image,
    WORKER_MAP,            # Configuration
    DEFAULT_CONFIG,
)
```

---

### 3. `serve/labels.json` ⭐⭐

**Vai trò:** Central Configuration - cấu hình toàn bộ pipeline

**Nội dung:**
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
    "triage_confidence": 0.5,      // Ngưỡng Triage
    "gatekeeper_confidence": 0.6,  // Ngưỡng Gatekeeper
    "low_confidence_fallback": 0.3
  },
  "conditional_execution": {
    "enabled": true,
    "skip_detection_modalities": ["Histopathology", "Dermoscopy"],
    "tool_mapping": {
      "Chest X-ray": ["grounding_dino", "medsam"],
      "Brain MRI": ["grounding_dino", "medsam"],
      "default": ["grounding_dino"]
    }
  }
}
```

**Ai đọc file này:**
- `src/trimed_orchestrator.py` (tự động load khi init)
- `serve/biomedclip_worker.py`
- `llava/serve/gradio_web_server_mmedagent.py`

---

## 🖥️ WORKER SERVERS (serve/)

### 4. `serve/biomedclip_worker.py` ⭐⭐

**Vai trò:** BiomedCLIP Worker - xử lý Triage và Gatekeeper

**Port:** `21006`

**Endpoint:** `POST /worker_generate`

**Payload:**
```json
{
  "image": "<base64_string>",
  "labels": ["Chest X-ray", "Brain MRI", ...]
}
```

**Response:**
```json
{
  "prediction": "Chest X-ray",
  "confidence": 0.95,
  "all_predictions": {
    "Chest X-ray": 0.95,
    "Brain MRI": 0.03,
    ...
  }
}
```

**Model:** `microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224`

---

### 5. `serve/grounding_dino_worker.py`

**Vai trò:** Grounding DINO Worker - object detection

**Port:** `21003`

**Endpoint:** `POST /worker_generate`

**Payload:**
```json
{
  "image": "<base64_string>",
  "prompt": "tumor"
}
```

**Response:**
```json
{
  "boxes": [[0.2, 0.3, 0.5, 0.6], ...],
  "phrases": ["tumor", ...],
  "logits": [0.85, ...]
}
```

---

### 6. `serve/MedSAM_worker.py`

**Vai trò:** MedSAM Worker - medical image segmentation

**Port:** `21004`

**Endpoint:** `POST /worker_generate`

**Payload:**
```json
{
  "image": "<base64_string>",
  "boxes": [[x1, y1, x2, y2], ...]
}
```

**Response:**
```json
{
  "masks": ["<rle_encoded_mask>", ...],
  "masks_rle": [...]
}
```

---

### 7. `serve/controller.py`

**Vai trò:** Worker Coordinator - quản lý và load balance workers

**Port:** `20001`

**Chức năng:**
- Registry: Lưu trữ danh sách workers
- Heart beat: Kiểm tra worker còn sống không
- Load balancing: Chọn worker phù hợp

---

## 🤖 LLAVA MODULE (llava/)

### 8. `llava/serve/model_worker.py`

**Vai trò:** LLaVA-Med Worker - Visual Question Answering

**Port:** `21002`

**Endpoint:** `POST /worker_generate_stream`

**Payload:**
```json
{
  "prompt": "USER: <image>\nWhat abnormalities? ASSISTANT:",
  "images": ["<base64_string>"],
  "temperature": 0.2,
  "max_new_tokens": 512
}
```

**Response (streaming):**
```
data: {"text": "I can see..."}
data: {"text": "I can see a lesion in..."}
...
```

---

### 9. `llava/conversation.py` ⭐

**Vai trò:** Prompt Templates - định dạng prompt cho LLaVA

**Exports:**
```python
from llava.conversation import conv_templates

# Available templates:
# - "llava_v1" (default)
# - "vicuna_v1"
# - "llama_2"
# - "plain"
# ...

# Usage trong trimed_orchestrator.py:
conv = conv_templates["llava_v1"].copy()
conv.append_message(conv.roles[0], "<image>\n" + user_query)
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()
# Output: "USER: <image>\nWhat do you see? ASSISTANT:"
```

---

### 10. `llava/serve/gradio_web_server_mmedagent.py`

**Vai trò:** Legacy Web UI - Gradio interface (monolithic)

**Port:** `7860`

**Lưu ý:** File này là phiên bản cũ (monolithic). Khuyến nghị dùng Orchestrator pattern mới thông qua `demo_trimedagent_orchestrator.ipynb`.

---

## 📓 NOTEBOOKS

### 11. `demo_trimedagent_orchestrator.ipynb` ⭐⭐⭐

**Vai trò:** Main Demo Notebook - chạy pipeline hoàn chỉnh

**Cấu trúc:**
1. Setup & Imports
2. Initialize Orchestrator
3. Health Check
4. Load Test Image
5. Run Individual Stages (optional)
6. Run Full Pipeline
7. Visualization (Raw boxes vs Verified boxes vs Masks)
8. Export Results to JSON

**Cách sử dụng:**
```python
from src import TriMedOrchestrator

orchestrator = TriMedOrchestrator()
result = orchestrator.run_full_chain(image, "Find the tumor")

# Visualize
visualize_pipeline_result(image, result)
```

---

### 12. `instruction_generation.ipynb`

**Vai trò:** Data Generation - sinh dữ liệu huấn luyện

**Chức năng:**
- Sinh instruction-following data
- Format dữ liệu cho training LLaVA

---

## 📁 OTHER DIRECTORIES

### `data_json/`
- `train_example.jsonl` - Training data
- `eval_example.jsonl` - Evaluation data  
- `eval_agent_tool_use_all.jsonl` - Tool use evaluation

### `data_processing/`
- `download_data.py` - Download datasets
- `dataset_loading.py` - Load datasets
- `data_process_func.py` - Processing utilities

### `scripts/`
- `finetune.sh` - Fine-tuning scripts
- `merge_lora_weights.py` - Merge LoRA weights
- Training và evaluation scripts

### `docs/`
- `DEMO_ENVIRONMENTS.md` - Platform-specific guides
- `PIPELINE_GUIDE.md` - Detailed pipeline docs
- `AGENT_ORCHESTRATION_GUIDE.md` - Agent architecture

---

## 🔗 DEPENDENCY GRAPH (Visual)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           FILE DEPENDENCY GRAPH                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  demo_trimedagent_orchestrator.ipynb                                │   │
│  │                    │                                                │   │
│  │                    ▼                                                │   │
│  │  ┌─────────────────────────────────────────────────────────────┐   │   │
│  │  │  src/__init__.py                                            │   │   │
│  │  │       │                                                     │   │   │
│  │  │       ▼                                                     │   │   │
│  │  │  src/trimed_orchestrator.py                                 │   │   │
│  │  │       │                                                     │   │   │
│  │  │       ├──────────▶ llava/conversation.py (conv_templates)   │   │   │
│  │  │       │                                                     │   │   │
│  │  │       └──────────▶ serve/labels.json (config)               │   │   │
│  │  └─────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│                              │ HTTP                                         │
│                              │ (requests library)                           │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    WORKER SERVERS (Running on GPU)                  │   │
│  │                                                                     │   │
│  │  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐  │   │
│  │  │ serve/           │  │ serve/           │  │ serve/           │  │   │
│  │  │ biomedclip_      │  │ grounding_dino_  │  │ MedSAM_          │  │   │
│  │  │ worker.py        │  │ worker.py        │  │ worker.py        │  │   │
│  │  │ :21006           │  │ :21003           │  │ :21004           │  │   │
│  │  │ ────────────     │  │ ────────────     │  │ ────────────     │  │   │
│  │  │ BiomedCLIP       │  │ Grounding DINO   │  │ MedSAM           │  │   │
│  │  │ (Triage+Gate)    │  │ (Detection)      │  │ (Segmentation)   │  │   │
│  │  └──────────────────┘  └──────────────────┘  └──────────────────┘  │   │
│  │                                                                     │   │
│  │  ┌──────────────────┐  ┌──────────────────┐                        │   │
│  │  │ llava/serve/     │  │ serve/           │                        │   │
│  │  │ model_worker.py  │  │ controller.py    │                        │   │
│  │  │ :21002           │  │ :20001           │                        │   │
│  │  │ ────────────     │  │ ────────────     │                        │   │
│  │  │ LLaVA-Med        │  │ Worker Registry  │                        │   │
│  │  │ (VQA)            │  │ (Coordinator)    │                        │   │
│  │  └──────────────────┘  └──────────────────┘                        │   │
│  │                                                                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 DATA FLOW (Chi tiết)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              PIPELINE DATA FLOW                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  INPUT: Image (PIL/Path/Base64) + Query ("Find the tumor")                 │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 1: PERCEIVE (Triage)                                         │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  Orchestrator._call_triage(image_b64)                               │   │
│  │       │                                                             │   │
│  │       ├──▶ HTTP POST http://localhost:21006/worker_generate         │   │
│  │       │    Payload: {image: b64, labels: [...]}                     │   │
│  │       │                                                             │   │
│  │       ◀── Response: {prediction: "Chest X-ray", confidence: 0.95}   │   │
│  │                                                                     │   │
│  │  Output: TriageResult(modality="Chest X-ray", confidence=0.95)      │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  CONTEXT INJECTION                                                  │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  context = "[System: Image is Chest X-ray with 95% confidence...]"  │   │
│  │  prompt = context + "\n\n" + user_query                             │   │
│  │                                                                     │   │
│  │  Uses: llava/conversation.py (conv_templates["llava_v1"])           │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 2: REASON (LLaVA)                                            │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  Orchestrator._call_llava(prompt, image_b64)                        │   │
│  │       │                                                             │   │
│  │       ├──▶ HTTP POST http://localhost:21002/worker_generate_stream  │   │
│  │       │    Payload: {prompt, images: [b64], temperature: 0.2}       │   │
│  │       │                                                             │   │
│  │       ◀── Response (streaming): "I can see a suspicious mass..."    │   │
│  │                                                                     │   │
│  │  Output: llava_response (string)                                    │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  DECISION: Should proceed to detection?                             │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  Check if query contains: "find", "detect", "segment", "locate"     │   │
│  │                                                                     │   │
│  │  YES → Continue to Stage 3                                          │   │
│  │  NO  → Return {triage, llava_response} and exit                     │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 3: ACT - Detection (DINO)                                    │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  target = _extract_target_entity(query)  # "tumor"                  │   │
│  │  Orchestrator._call_dino(target, image_b64)                         │   │
│  │       │                                                             │   │
│  │       ├──▶ HTTP POST http://localhost:21003/worker_generate         │   │
│  │       │    Payload: {image: b64, prompt: "tumor"}                   │   │
│  │       │                                                             │   │
│  │       ◀── Response: {boxes: [[0.2,0.3,0.5,0.6], ...], logits: [...]}│   │
│  │                                                                     │   │
│  │  Output: DetectionResult(boxes=[...], scores=[...])                 │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 4: VERIFY (Gatekeeper)                                       │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  for each box in raw_boxes:                                         │   │
│  │      cropped = crop_image_by_box(image, box)                        │   │
│  │      result = _call_gatekeeper(cropped, target)                     │   │
│  │          │                                                          │   │
│  │          ├──▶ HTTP POST http://localhost:21006/worker_generate      │   │
│  │          │    Payload: {image: cropped_b64,                         │   │
│  │          │              labels: ["Pathology of tumor", "Normal"]}   │   │
│  │          │                                                          │   │
│  │          ◀── Response: {prediction: "Pathology", confidence: 0.85}  │   │
│  │                                                                     │   │
│  │      if confidence > 0.6: KEEP box                                  │   │
│  │      else: REJECT box                                               │   │
│  │                                                                     │   │
│  │  Output: verified_boxes, rejected_boxes                             │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  STAGE 5: ACT - Segmentation (MedSAM)                               │   │
│  │  ──────────────────────────────────────────────────────────────     │   │
│  │  Orchestrator._call_medsam(image_b64, verified_boxes)               │   │
│  │       │                                                             │   │
│  │       ├──▶ HTTP POST http://localhost:21004/worker_generate         │   │
│  │       │    Payload: {image: b64, boxes: [[x1,y1,x2,y2], ...]}       │   │
│  │       │                                                             │   │
│  │       ◀── Response: {masks: [...]}                                  │   │
│  │                                                                     │   │
│  │  Output: SegmentationResult(masks=[...])                            │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│         │                                                                   │
│         ▼                                                                   │
│  OUTPUT: PipelineResult                                                    │
│  {                                                                          │
│    triage: TriageResult,                                                   │
│    llava_response: str,                                                    │
│    dino_raw_boxes: [...],                                                  │
│    verified_boxes: [...],                                                  │
│    rejected_boxes: [...],                                                  │
│    masks: [...],                                                           │
│    execution_time: float                                                   │
│  }                                                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 🎯 TÓM TẮT

| File | Tầm quan trọng | Vai trò |
|------|----------------|---------|
| `src/trimed_orchestrator.py` | ⭐⭐⭐ | Core logic - Thin Client |
| `serve/labels.json` | ⭐⭐ | Central configuration |
| `serve/biomedclip_worker.py` | ⭐⭐ | Triage + Gatekeeper |
| `llava/conversation.py` | ⭐⭐ | Prompt templates |
| `demo_trimedagent_orchestrator.ipynb` | ⭐⭐⭐ | Main demo |
| `serve/grounding_dino_worker.py` | ⭐ | Detection |
| `serve/MedSAM_worker.py` | ⭐ | Segmentation |
| `llava/serve/model_worker.py` | ⭐ | LLaVA VQA |

---

## ✅ CHECKLIST: Chạy Pipeline

1. [ ] Start Controller: `python -m serve.controller --port 20001`
2. [ ] Start BiomedCLIP: `python -m serve.biomedclip_worker --port 21006`
3. [ ] Start LLaVA: `python -m llava.serve.model_worker --port 21002`
4. [ ] Start DINO: `python -m serve.grounding_dino_worker --port 21003`
5. [ ] Start MedSAM: `python -m serve.MedSAM_worker --port 21004`
6. [ ] Run Notebook: `jupyter notebook demo_trimedagent_orchestrator.ipynb`
7. [ ] Health Check: `orchestrator.health_check()` → all True
8. [ ] Run Pipeline: `orchestrator.run_full_chain(image, query)`
