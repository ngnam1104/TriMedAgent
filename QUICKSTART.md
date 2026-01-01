# 🏥 TriMedAgent - Hướng Dẫn Nhanh

## Giới Thiệu 30 Giây

**TriMedAgent** = MMedAgent + 3 cải tiến:

```
1. TRIAGE      → BiomedCLIP phân loại ảnh (X-ray? MRI? CT?)
2. INJECTION   → Thêm context vào prompt cho LLaVA  
3. GATEKEEPER  → Lọc bỏ false positives từ detection
```

## Pipeline

```
Ảnh Y Tế ──▶ [Triage] ──▶ [LLaVA + Context] ──▶ [Tool] ──▶ [Gatekeeper] ──▶ Kết Quả
              │                │                  │              │
          "Chest X-ray"    "Dùng DINO"        4 boxes      2 boxes ✓
           (95% conf)                        (raw)       (verified)
```

## Cài Đặt Nhanh

```bash
# 1. Clone & setup
git clone <repo>
cd MMedAgent-v2
conda create -n trimedagent python=3.10 -y
conda activate trimedagent
pip install -e .
pip install open_clip_torch

# 2. Chạy demo notebook
jupyter notebook demo_trimedagent.ipynb
```

## Chạy Full System

```bash
# Terminal 1: Controller
python -m serve.controller --port 20001

# Terminal 2: LLaVA-Med
python -m llava.serve.model_worker --port 40000 --controller-address http://localhost:20001

# Terminal 3: BiomedCLIP (Triage + Gatekeeper)
python -m serve.biomedclip_worker --port 21006 --controller-address http://localhost:20001

# Terminal 4: Web UI
python -m llava.serve.gradio_web_server_mmedagent --controller-url http://localhost:20001 --port 7860

# Mở browser: http://localhost:7860
```

## File Quan Trọng

| File | Chức năng |
|------|-----------|
| `serve/labels.json` | Cấu hình thresholds, tools |
| `serve/biomedclip_worker.py` | Triage + Gatekeeper |
| `llava/serve/gradio_web_server_mmedagent.py` | Pipeline chính |
| `demo_trimedagent.ipynb` | Demo notebook |

## Cấu Hình Nhanh

Mở `serve/labels.json`:

```json
{
  "thresholds": {
    "triage_confidence": 0.5,      // Ngưỡng tin tưởng triage
    "gatekeeper_confidence": 0.6   // Ngưỡng giữ box
  }
}
```

## So Sánh

| | MMedAgent | TriMedAgent |
|---|---|---|
| Biết loại ảnh? | ❌ | ✅ Triage |
| Lọc false positive? | ❌ | ✅ Gatekeeper |
| Skip tool thừa? | ❌ | ✅ Conditional |

## Tài Liệu Chi Tiết

👉 Xem [README_TRIMEDAGENT.md](README_TRIMEDAGENT.md) để biết thêm chi tiết.
