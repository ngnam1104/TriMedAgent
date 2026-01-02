# 🏥 TriMedAgent - Hướng Dẫn Nhanh

## 🎯 Giới Thiệu 30 Giây

**TriMedAgent** = MMedAgent + 3 cải tiến:

```
1. TRIAGE      → BiomedCLIP phân loại ảnh (X-ray? MRI? CT?)
2. INJECTION   → Thêm context vào prompt cho LLaVA  
3. GATEKEEPER  → Lọc bỏ false positives từ detection
```

## 📋 Pipeline (Orchestrator Pattern)

```
┌─────────────────────────────────────────────────────────────────────────┐
│  TriMedOrchestrator.run_full_chain(image, query)                        │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Stage 1: PERCEIVE ──▶ BiomedCLIP Triage ──▶ "Chest X-ray" (95%)       │
│                │                                                        │
│                ▼                                                        │
│  Stage 2: REASON ──▶ LLaVA-Med + Context ──▶ "Use detection tool"      │
│                │                                                        │
│                ▼                                                        │
│  Stage 3: ACT ──▶ Grounding DINO ──▶ 4 boxes detected                  │
│                │                                                        │
│                ▼                                                        │
│  Stage 4: VERIFY ──▶ Gatekeeper ──▶ 2 boxes verified ✓                 │
│                │                                                        │
│                ▼                                                        │
│  Stage 5: ACT ──▶ MedSAM ──▶ 2 masks generated                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Orchestrator Pattern (Khuyến nghị) ⭐

**Bước 1: Khởi chạy Workers (trên máy có GPU)**

```bash
# Terminal 1: Controller
python -m serve.controller --port 20001

# Terminal 2: LLaVA-Med
python -m llava.serve.model_worker --port 21002 --controller-address http://localhost:20001

# Terminal 3: BiomedCLIP
python -m serve.biomedclip_worker --port 21006 --controller-address http://localhost:20001

# Terminal 4: Grounding DINO
python -m serve.grounding_dino_worker --port 21003 --controller-address http://localhost:20001

# Terminal 5: MedSAM
python -m serve.MedSAM_worker --port 21004 --controller-address http://localhost:20001
```

**Bước 2: Chạy Notebook**

```bash
jupyter notebook demo_trimedagent_orchestrator.ipynb
```

**Hoặc Python code:**

```python
from src import TriMedOrchestrator

# Initialize (Thin Client - không cần GPU)
orchestrator = TriMedOrchestrator()

# Health Check
status = orchestrator.health_check()
print(status)  # {'llava': True, 'biomedclip': True, ...}

# Run Full Pipeline
result = orchestrator.run_full_chain(
    image="path/to/xray.jpg",
    user_query="Find and segment any abnormalities"
)

# Access Results
print(f"Modality: {result.triage.modality}")
print(f"Verified boxes: {result.verified_boxes}")
print(f"Masks: {len(result.masks)}")
```

### Option 2: Demo Notebook

| Notebook | Description |
|----------|-------------|
| `demo_trimedagent_orchestrator.ipynb` | ⭐ Full pipeline với Orchestrator |

---

## 📂 File Quan Trọng

| File | Chức năng |
|------|-----------|
| `src/trimed_orchestrator.py` | ⭐ **Thin Client** - điều phối pipeline qua HTTP |
| `src/__init__.py` | Package exports |
| `serve/labels.json` | Cấu hình: thresholds, labels, tools |
| `serve/biomedclip_worker.py` | Worker: Triage + Gatekeeper (Port 21006) |
| `serve/grounding_dino_worker.py` | Worker: Detection (Port 21003) |
| `serve/MedSAM_worker.py` | Worker: Segmentation (Port 21004) |
| `llava/serve/model_worker.py` | Worker: LLaVA-Med (Port 21002) |
| `llava/conversation.py` | Prompt templates (`conv_templates`) |

---

## ⚙️ Cấu Hình Nhanh

Mở `serve/labels.json`:

```json
{
  "triage_labels": [
    "Chest X-ray", "Brain MRI", "Abdominal CT", "Histopathology",
    "Ultrasound", "Dermoscopy", "Gross pathology", "Bone X-ray",
    "Lung CT", "Retinal fundus", "Mammography"
  ],
  "thresholds": {
    "triage_confidence": 0.5,
    "gatekeeper_confidence": 0.6
  },
  "action_keywords": [
    "find", "detect", "locate", "segment", "where is"
  ]
}
```

---

## 🖥️ VRAM Requirements

| Mode | VRAM | Use Case |
|------|------|----------|
| **Demo** | ~1GB | BiomedCLIP only |
| **Lite** | ~8-10GB | 4-bit quantized LLaVA + BiomedCLIP |
| **Full** | ~18-20GB | All models: LLaVA + BiomedCLIP + DINO + MedSAM |

---

## 📊 So Sánh MMedAgent vs TriMedAgent

| Feature | MMedAgent | TriMedAgent |
|---------|-----------|-------------|
| Biết loại ảnh? | ❌ Không | ✅ Triage (BiomedCLIP) |
| Lọc false positive? | ❌ Không | ✅ Gatekeeper |
| Skip tool thừa? | ❌ Không | ✅ Conditional execution |
| Context-aware? | ❌ Không | ✅ Context injection |
| Architecture | Monolithic | ✅ **Orchestrator Pattern** |

---

## 🔗 File Dependency Graph

```
demo_trimedagent_orchestrator.ipynb
         │
         └──▶ src/__init__.py
                   │
                   └──▶ src/trimed_orchestrator.py
                              │
                              ├──▶ llava/conversation.py (prompt templates)
                              │
                              └──▶ serve/labels.json (config)
                                            │
                                            ▼
                              ┌─────────────────────────────┐
                              │  HTTP Workers (Port Map)    │
                              ├─────────────────────────────┤
                              │  21002: LLaVA-Med           │
                              │  21003: Grounding DINO      │
                              │  21004: MedSAM              │
                              │  21006: BiomedCLIP          │
                              │  20001: Controller          │
                              └─────────────────────────────┘
```

---

## ❓ Troubleshooting

### Worker không khởi động
```bash
# Check port đã được sử dụng chưa
netstat -ano | findstr :21006
```

### CUDA Out of Memory
```python
# Use 4-bit quantization
from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(load_in_4bit=True)
```

### Health Check Failed
```python
orchestrator = TriMedOrchestrator()
status = orchestrator.health_check()
# {'llava': False, 'biomedclip': True, ...}
# → Worker 'llava' chưa chạy
```

---

## 📚 Tài Liệu Thêm

- [docs/DEMO_ENVIRONMENTS.md](docs/DEMO_ENVIRONMENTS.md) - Hướng dẫn từng platform
- [docs/PIPELINE_GUIDE.md](docs/PIPELINE_GUIDE.md) - Chi tiết pipeline
- [README.md](README.md) - Tổng quan project
