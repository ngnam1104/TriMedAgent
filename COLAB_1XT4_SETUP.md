# TriMedAgent on Google Colab (1xT4) Setup Guide

## ✅ Migration Complete: `demo_colab_1xt4.ipynb`

This notebook has been fully adapted from the Kaggle 2xT4 version to run on **Google Colab with 1 T4 GPU** (~15GB VRAM).

---

## 🔧 Key Configuration Changes

### 1. **GPU Configuration (Cell 1.1)**
```python
LITE_MODE = True  # 4-bit quantization REQUIRED for single GPU
LOAD_LLAVA = True      # LLaVA-Med
LOAD_DINO = True       # Grounding DINO
LOAD_SAM = True        # MedSAM
LOAD_BIOMED = True     # BiomedCLIP
```

### 2. **Device Mapping (Cells 2.1-2.4)**
Changed from dual-GPU distribution to single GPU:
```python
# Colab 1xT4 setup
device_map = {
    "llava": "cuda:0",
    "biomedclip": "cuda:0",
    "grounding_dino": "cuda:0",
    "medsam": "cuda:0"
}
```

### 3. **Orchestrator Initialization (Cell 4.1)**
```python
orchestrator = HybridReActOrchestrator(
    llava_device="cuda:0",      # All on single GPU
    tool_device="cuda:0",
    max_iterations=3,
    enable_verification=True,
    enable_rag=False
)
```

### 4. **Paths Updated**
- Repository clone: `/content/TriMedAgent` (Colab standard)
- Sample image: `/content/TriMedAgent/images/example_chest.jpg`
- Working directory: `/content/` instead of `/kaggle/working/`

---

## 📊 Memory Optimization Settings

| Component | Configuration | Reason |
|-----------|---------------|--------|
| **LLaVA-Med** | 4-bit quantization | ~6GB instead of 15GB |
| **BiomedCLIP** | GPU lazy load | 784MB on demand |
| **Grounding DINO** | GPU lazy load | 694MB on demand |
| **MedSAM** | GPU lazy load + offload | 375MB on demand |
| **Garbage Collection** | Enabled between tools | Free VRAM between operations |

---

## 📋 Notebook Structure (30 cells)

### Section 1: Setup (Cells 1.1-1.3)
- Check GPU & Configuration
- Clone Repo & Install Dependencies
- Download Pre-trained Weights

### Section 2: Load Tools (Cells 2.1-2.4)
- BiomedCLIP (Triage)
- Grounding DINO (Detection)
- MedSAM (Segmentation)
- LLaVA-Med (Visual Reasoning)

### Section 3: Test Tools (Cells 3.1-3.4)
- Load Sample Image
- Test BiomedCLIP
- Test Grounding DINO
- Test MedSAM

### Section 4: Orchestrator (Cells 4.1-4.3)
- Initialize Orchestrator
- Load All Tools
- Test Integration

### Section 5: Pipeline Demo (Cells 5.1-5.2)
- Full End-to-End Pipeline
- Test with Custom Query

### Section 6: Gradio Interface (Cells 6.1-6.2)
- Launch Web Interface
- Stop Server (Optional)

---

## ⚠️ Important Notes

### **First Run Expected Times:**
- **Cell 1.2 (Dependencies)**: 3-5 minutes
- **Cell 1.3 (Weights)**: 5-10 minutes (~2GB download)
- **Cell 2.4 (LLaVA load)**: 2-3 minutes (includes conversion to 4-bit)
- **Cell 3.x (First tool test)**: 1-2 minutes (initialization)
- **Cell 6.1 (Gradio launch)**: 30-60 seconds

### **Memory Management:**
- **Do not** load all models at once on Colab 1xT4
- If OOM errors occur:
  - Disable unused tools (LOAD_DINO=False, etc.)
  - Restart Colab runtime
  - Skip Gradio demo if tools alone use >12GB

### **Public URL for Sharing:**
```python
demo = launch_demo(orchestrator, share=True)
# Generates: https://xxxxx.gradio.live
```

---

## 🚀 Quick Start

1. **Open notebook**: `demo_colab_1xt4.ipynb`
2. **Run Cell 1.1-1.3**: Setup environment
3. **Run Cell 2.x**: Load individual tools (takes 10-15 min total)
4. **Run Cell 3.x**: Test tools with sample image
5. **Run Cell 6.1**: Launch Gradio interface

---

## 🔄 Differences from Kaggle Version

| Aspect | Kaggle (2xT4) | Colab (1xT4) |
|--------|---------------|-------------|
| **GPU Count** | 2 | 1 |
| **Device Mapping** | Split across GPUs | All on `cuda:0` |
| **LLaVA Quantization** | 4-bit | 4-bit |
| **Memory per GPU** | 29GB each | ~15GB |
| **Working Dir** | `/kaggle/working/` | `/content/` |
| **Gradio Share** | Via Kaggle proxy | Via Gradio public URL |
| **LITE_MODE** | Optional | Required |

---

## 📞 Troubleshooting

### **Error: "No such file or directory: /content/TriMedAgent"**
→ Make sure Cell 1.2 (clone repo) completed successfully

### **Error: "RuntimeError: CUDA out of memory"**
→ Disable optional tools or restart Colab and run cells sequentially

### **Error: "Module not found: open_clip"**
→ Re-run Cell 1.2 to install dependencies

### **Gradio URL doesn't load**
→ Make sure `share=True` is set in Cell 6.1, wait 30 seconds for ngrok tunnel

---

## 📁 Project Structure

```
/content/TriMedAgent/
├── src/
│   ├── utils/
│   │   └── kaggle_config.py       # GPU config & device mapping
│   ├── tools/                     # Tool implementations
│   │   ├── llava.py              # LLaVA-Med with 4-bit quantization
│   │   ├── biomedclip.py         # BiomedCLIP triage
│   │   ├── grounding_dino.py     # Object detection
│   │   └── medsam.py             # Medical segmentation
│   ├── core/                      # Orchestrator logic
│   └── ui/
│       └── gradio_app.py         # Gradio interface
├── demo_colab_1xt4.ipynb         # This notebook
├── images/
│   └── example_chest.jpg         # Sample medical image
└── weights/                       # Downloaded models (auto-created)
    ├── medsam_vit_b.pth
    ├── groundingdino_swint_ogc.pth
    └── GroundingDINO_SwinT_OGC.py
```

---

**Last Updated**: November 2024  
**Status**: ✅ Ready for Colab Deployment
