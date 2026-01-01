# 📚 Hướng Dẫn Chạy MMedAgent Từ Đầu

> **MMedAgent** - Multi-modal Medical Agent: AI Agent y tế đa phương thức đầu tiên tích hợp nhiều công cụ để xử lý các tác vụ y tế trên nhiều modality khác nhau.

---

## 📋 Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Yêu Cầu Hệ Thống](#2-yêu-cầu-hệ-thống)
3. [Cài Đặt Môi Trường](#3-cài-đặt-môi-trường)
4. [Tải Model & Checkpoints](#4-tải-model--checkpoints)
5. [Chạy Inference (Đánh Giá)](#5-chạy-inference-đánh-giá)
6. [Huấn Luyện Model](#6-huấn-luyện-model)
7. [Chạy Web UI & Server](#7-chạy-web-ui--server)
8. [🎯 Chạy Trên Kaggle](#8-chạy-trên-kaggle)
9. [Cấu Trúc Project](#9-cấu-trúc-project)
10. [Troubleshooting](#10-troubleshooting)

---

## 1. Tổng Quan

### 1.1 MMedAgent là gì?

MMedAgent là một AI Agent y tế đa phương thức (multi-modal) được xây dựng trên nền tảng LLaVA-Med và LLaVA-Plus. Nó có khả năng:

| Tác vụ | Công cụ | Mô tả |
|--------|---------|-------|
| **VQA** | LLaVA-Med | Hỏi đáp hình ảnh y tế |
| **Classification** | BiomedCLIP | Phân loại hình ảnh y tế |
| **Grounding** | Grounding DINO | Định vị vùng quan tâm |
| **Segmentation** | MedSAM | Phân đoạn với bounding-box |
| **G-Seg** | Grounding DINO + MedSAM | Phân đoạn với text prompts |
| **MRG** | ChatCAD | Sinh báo cáo y tế |
| **RAG** | ChatCAD+ | Retrieval Augmented Generation |

### 1.2 Kiến Trúc Tổng Quan

```
┌─────────────────────────────────────────────────────────────┐
│                      Gradio Web UI                          │
│                   (localhost:7860)                          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                    Controller                               │
│               (localhost:20001)                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
┌───────▼───────┐ ┌───▼───┐ ┌───────▼───────┐
│ Model Worker  │ │ Tool  │ │ Tool Workers  │
│ (LLaVA-Med)   │ │Workers│ │ (MedSAM, etc) │
│ :40000        │ │:21001+│ │               │
└───────────────┘ └───────┘ └───────────────┘
```

---

## 2. Yêu Cầu Hệ Thống & Chi Phí

### 2.1 Phần Cứng

| Thành phần | Tối thiểu | Khuyến nghị |
|------------|-----------|-------------|
| **GPU** | NVIDIA 8GB VRAM | NVIDIA 24GB+ VRAM (A100, RTX 3090/4090) |
| **RAM** | 16GB | 32GB+ |
| **Disk** | 50GB | 100GB+ (cho models và data) |
| **CUDA** | 11.7+ | 12.0+ |

### 2.2 💰 Chi Phí Tài Nguyên Chi Tiết

#### 🖥️ GPU Requirements theo Task

| Task | GPU VRAM | Thời gian ước tính | Ghi chú |
|------|----------|-------------------|---------|
| **Inference (1 image)** | 8-16GB | ~5-10s/image | RTX 3080 trở lên |
| **Inference (batch)** | 16-24GB | ~2-5s/image | A100/RTX 4090 |
| **Training LoRA** | 24-40GB | ~10-20h (30 epochs) | A100 40GB khuyến nghị |
| **Training Full** | 80GB+ | ~2-5 ngày | Multi-GPU A100 80GB |
| **Web UI (tất cả tools)** | 24-48GB | Continuous | Chạy nhiều workers |

#### 💵 Chi Phí GPU Cloud (Ước tính)

| Provider | GPU | Giá/giờ | Chi phí Training 20h |
|----------|-----|---------|---------------------|
| **Google Colab Pro+** | A100 40GB | ~$0.50/h | ~$10 |
| **AWS EC2 p4d** | A100 40GB | ~$3.50/h | ~$70 |
| **Lambda Labs** | A100 40GB | ~$1.10/h | ~$22 |
| **RunPod** | A100 40GB | ~$1.00/h | ~$20 |
| **Vast.ai** | RTX 4090 | ~$0.40/h | ~$8 |

#### 🔑 API Keys & Chi Phí

| API | Cần cho | Chi phí | Ghi chú |
|-----|---------|---------|---------|
| **OpenAI API** | GPT-4 evaluation, ChatCAD-R (RAG) | ~$0.03/1K tokens (GPT-4) | **Bắt buộc** cho eval_gpt4.py và RAG |
| **HuggingFace** | Tải models (LLaMA) | Miễn phí (cần đăng ký) | Cần access token |
| **Weights & Biases** | Training logging | Miễn phí (basic) | Tuỳ chọn |

#### 📊 Ước Tính Chi Phí OpenAI cho Evaluation

```
GPT-4 Evaluation (100 samples):
- Input: ~500 tokens/sample × 100 = 50,000 tokens
- Output: ~200 tokens/sample × 100 = 20,000 tokens
- Chi phí: ~$1.50 - $3.00 (GPT-4)
- Chi phí: ~$0.10 - $0.30 (GPT-4o-mini)

ChatCAD-R RAG (mỗi query):
- Input: ~1000-2000 tokens
- Output: ~500 tokens
- Chi phí: ~$0.05 - $0.10/query (GPT-4)
```

#### 💾 Storage Requirements

| Item | Dung lượng | Ghi chú |
|------|-----------|---------|
| **LLaMA-7B base** | ~13GB | Bắt buộc |
| **LLaVA-Med delta** | ~11GB | Bắt buộc |
| **MMedAgent LoRA** | ~500MB | Từ HuggingFace |
| **Merged model** | ~14GB | Sau khi merge |
| **Tool checkpoints** | ~2GB | MedSAM, DINO, ChatCAD |
| **Training data** | ~5-20GB | Tuỳ dataset |
| **Tổng cộng** | **~50-80GB** | Khuyến nghị 100GB+ |

#### ⚡ Chạy Miễn Phí / Chi Phí Thấp

**Option 1: Google Colab (Miễn phí/Pro)**
```
- Free tier: T4 16GB (inference only)
- Pro: A100 40GB (training possible)
- Hạn chế: Session timeout, storage limited
```

**Option 2: Kaggle Notebooks**
```
- GPU: P100 16GB hoặc T4 x2
- 30h GPU/tuần miễn phí
- Phù hợp: Inference, small training
```

**Option 3: Local GPU (One-time cost)**
```
- RTX 3090 24GB: ~$700-900 (used)
- RTX 4090 24GB: ~$1,600-2,000
- Phù hợp: Long-term development
```

### 2.3 Phần Mềm

- **OS**: Linux (Ubuntu 20.04+), Windows 10/11 với WSL2
- **Python**: 3.10 (khuyến nghị)
- **Conda**: Miniconda hoặc Anaconda
- **Git LFS**: Để tải model weights

---

## 3. Cài Đặt Môi Trường

### 3.1 Clone Repository

```bash
# Clone repo
git clone https://github.com/Wangyixinxin/MMedAgent.git
cd MMedAgent
```

### 3.2 Tạo Môi Trường Conda

```bash
# Tạo môi trường Python 3.10
conda create -n mmedagent python=3.10 -y
conda activate mmedagent

# Cập nhật pip
pip install --upgrade pip
```

### 3.3 Cài Đặt Dependencies Cơ Bản

```bash
# Cài đặt package chính (editable mode)
pip install -e .
```

### 3.4 Cài Đặt Dependencies Cho Training (Tuỳ Chọn)

```bash
# Cài đặt thêm cho training
pip install -e ".[train]"

# Cài đặt flash-attention (tăng tốc)
pip install flash-attn --no-build-isolation
```

### 3.5 Kiểm Tra Cài Đặt

```bash
# Kiểm tra PyTorch và CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"

# Kiểm tra các package chính
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import gradio; print(f'Gradio: {gradio.__version__}')"
```

**Output mong đợi:**
```
PyTorch: 2.0.1
CUDA available: True
CUDA version: 11.7
Transformers: 4.31.0
Gradio: 3.35.2
```

---

## 4. Tải Model & Checkpoints

### 4.1 Tải MMedAgent Checkpoint (LoRA)

```bash
# Cài đặt Git LFS
git lfs install

# Tải model và data từ HuggingFace
git clone https://huggingface.co/andy0207/mmedagent
```

**Cấu trúc sau khi tải:**
```
mmedagent/
├── final_model_lora/          # LoRA weights
├── instruction_data/          # Instruction tuning data
│   └── instruction_all.json   # ~97MB
└── ...
```

### 4.2 Tải Base Model (LLaVA-Med)

#### Bước 1: Tải Delta Weights

```bash
# Tải delta weights (~11GB)
wget https://hanoverprod.z21.web.core.windows.net/med_llava/models/llava_med_in_text_60k_ckpt2_delta.zip

# Giải nén
unzip llava_med_in_text_60k_ckpt2_delta.zip -d ./llava_med_delta
```

#### Bước 2: Tải LLaMA-7B Weights

Truy cập [HuggingFace LLaMA](https://huggingface.co/docs/transformers/main/model_doc/llama) và làm theo hướng dẫn để tải `llama-7b` (cần đăng ký).

#### Bước 3: Apply Delta Weights

```bash
# Tạo base_model từ LLaMA + delta weights
python -m llava.model.apply_delta \
    --base /path/to/llama-7b \
    --target ./base_model \
    --delta ./llava_med_delta
```

### 4.3 Merge LoRA Weights (Tạo Model Hoàn Chỉnh)

```bash
# Merge LoRA weights vào base model
CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path ./mmedagent/final_model_lora \
    --model-base ./base_model \
    --save-model-path ./llava_med_agent
```

**Hoặc dùng script:**
```bash
# Chỉnh sửa merge.sh nếu cần, rồi chạy
bash merge.sh
```

### 4.4 Tải Tool Checkpoints (Cho Web UI)

#### GroundingDINO Checkpoint
```bash
# Tải từ Google Drive và lưu vào src/
# Link: https://drive.google.com/drive/folders/1eK27gz0tkbcp-9hx2fI9J_Wj2zHv14pL
# File: groundingdinomed-checkpoint0005_slim.pth → src/
```

#### MedSAM Checkpoint
```bash
# Tải từ Google Drive và lưu vào src/
# Link: https://drive.google.com/drive/folders/1ETWmi4AiniJeWOt6HAsYgTjYv_fkgzoN
# File: medsam_vit_b.pth → src/
```

#### ChatCAD Dependencies
```bash
# Tải từ Google Drive
# Link: https://drive.google.com/drive/folders/14OWwsFjphsjqT-nH9GHgf5Sy7f1aL9Lz

# Lưu files:
# - r2gcmn_mimic-cxr.pth → src/ChatCAD_R/weights/
# - JFchexpert.pth → src/ChatCAD_R/weights/
# - annotation.json → src/ChatCAD_R/r2g/
```

### 4.5 Cài Đặt Tool Dependencies

```bash
# GroundingDINO
cd src
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO
pip install -e .
cd ../..

# MedSAM
cd src
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM
pip install -e .
cd ../..

# Thay thế build_sam.py (dùng vit_b)
cp src/build_sam.py src/MedSAM/segment_anything/build_sam.py

# ChatCAD_R dependencies
pip install -r src/ChatCAD_R/requirements.txt

# Fix version conflicts
pip install httpx==0.24.0 supervision==0.10.0
```

---

## 5. Chạy Inference (Đánh Giá)

### 5.1 Inference Cơ Bản

```bash
# Chạy inference trên eval dataset
CUDA_VISIBLE_DEVICES=0 python llava/eval/model_vqa.py \
    --model-path ./llava_med_agent \
    --question-file ./eval_data_json/eval_example.jsonl \
    --image-folder ./eval_images \
    --answers-file ./eval_data_json/output_agent_eval_example.jsonl \
    --temperature 0.2
```

**Hoặc dùng script:**
```bash
bash eval.sh
```

### 5.2 Inference với GPT-4o (Cần API Key)

```bash
# Set API key
export OPENAI_API_KEY="your-api-key"

# Chạy GPT-4o inference
python llava/eval/eval_gpt4o.py \
    --api-key "$OPENAI_API_KEY" \
    --question ./eval_data_json/eval_example.jsonl \
    --output ./eval_data_json/output_gpt4o_eval_example.jsonl \
    --max-tokens 1024
```

### 5.3 Đánh Giá Bằng GPT-4

```bash
# So sánh output với GPT-4
python ./llava/eval/eval_gpt4.py \
    --question_input_path ./eval_data_json/eval_example.jsonl \
    --input_path ./eval_data_json/output_agent_eval_example.jsonl \
    --output_path ./eval_data_json/compare_review.jsonl
```

### 5.4 Format Input/Output

**Input (`eval_example.jsonl`):**
```json
{
  "question_id": 1,
  "image": "image_001.jpg",
  "text": "What abnormalities can you see in this chest X-ray?"
}
```

**Output (`output_agent_eval_example.jsonl`):**
```json
{
  "question_id": 1,
  "prompt": "What abnormalities can you see in this chest X-ray?",
  "text": "The chest X-ray shows...",
  "model_id": "llava_med_agent"
}
```

---

## 6. Huấn Luyện Model

### 6.1 Chuẩn Bị Data

**Format training data (`train_data_json/example.jsonl`):**
```json
{
  "id": "unique_id",
  "image": "image_name.jpg",
  "conversations": [
    {"from": "human", "value": "<image>\nYour question here"},
    {"from": "gpt", "value": "Model response here"}
  ]
}
```

### 6.2 Training với LoRA (Khuyến Nghị)

```bash
# Training với DeepSpeed + LoRA
deepspeed llava/train/train_mem.py \
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path ./base_model \
    --version v1 \
    --data_path ./train_data_json/example.jsonl \
    --image_folder ./train_images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length False \
    --bf16 True \
    --output_dir ./checkpoints/output_lora_weights \
    --num_train_epochs 30 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 2 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
```

**Hoặc dùng script:**
```bash
bash tuning.sh
```

### 6.3 Các Tham Số Quan Trọng

| Tham số | Giá trị | Mô tả |
|---------|---------|-------|
| `--lora_r` | 128 | Rank của LoRA adapter |
| `--lora_alpha` | 256 | Scaling factor |
| `--per_device_train_batch_size` | 12 | Batch size (giảm nếu OOM) |
| `--gradient_accumulation_steps` | 2 | Accumulation steps |
| `--num_train_epochs` | 30 | Số epochs |
| `--learning_rate` | 2e-4 | Learning rate |

### 6.4 Sau Khi Training

```bash
# Merge LoRA weights vào model hoàn chỉnh
CUDA_VISIBLE_DEVICES=0 python scripts/merge_lora_weights.py \
    --model-path ./checkpoints/output_lora_weights \
    --model-base ./base_model \
    --save-model-path ./llava_med_agent_finetuned
```

---

## 7. Chạy Web UI & Server

### 7.1 Tổng Quan Architecture

Để chạy Web UI, cần khởi động **4 loại services** theo thứ tự:

1. **Controller** - Điều phối requests
2. **Model Worker** - Chạy LLaVA-Med model
3. **Tool Workers** - Các công cụ y tế (MedSAM, GroundingDINO, etc.)
4. **Gradio Web Server** - Giao diện người dùng

### 7.2 Khởi Động Services

**Terminal 1 - Controller:**
```bash
python -m llava.serve.controller --host 0.0.0.0 --port 20001
```

**Terminal 2 - Model Worker:**
```bash
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:20001 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path ./llava_med_agent
```

**Terminal 3 - Tool Workers (chạy từng cái):**
```bash
# Grounding DINO
python serve/grounding_dino_worker.py

# MedSAM
python serve/MedSAM_worker.py

# Grounded MedSAM (kết hợp)
python serve/grounded_medsam_worker.py

# BiomedCLIP
python serve/biomedclip_worker.py

# ChatCAD (Medical Report Generation)
python serve/chatcad_G_worker.py

# ChatCAD (RAG)
python serve/chatcad_R_worker.py
```

**Terminal 4 - Gradio Web Server:**
```bash
python llava/serve/gradio_web_server_mmedagent.py \
    --controller http://localhost:20001 \
    --model-list-mode reload
```

### 7.3 Truy Cập Web UI

Mở trình duyệt và truy cập: **http://localhost:7860**

### 7.4 Script Khởi Động Tự Động (Linux/WSL)

Tạo file `start_server.sh`:
```bash
#!/bin/bash

# Start controller
python -m llava.serve.controller --host 0.0.0.0 --port 20001 &
sleep 5

# Start model worker
python -m llava.serve.model_worker \
    --host 0.0.0.0 \
    --controller http://localhost:20001 \
    --port 40000 \
    --worker http://localhost:40000 \
    --model-path ./llava_med_agent &
sleep 10

# Start tool workers
python serve/grounding_dino_worker.py &
python serve/MedSAM_worker.py &
python serve/grounded_medsam_worker.py &
python serve/biomedclip_worker.py &
python serve/chatcad_G_worker.py &
python serve/chatcad_R_worker.py &
sleep 5

# Start gradio
python llava/serve/gradio_web_server_mmedagent.py \
    --controller http://localhost:20001 \
    --model-list-mode reload

echo "All services started! Access http://localhost:7860"
```

### 7.5 Windows PowerShell

Mở nhiều PowerShell windows và chạy từng lệnh:

```powershell
# Window 1: Controller
conda activate mmedagent
python -m llava.serve.controller --host 0.0.0.0 --port 20001

# Window 2: Model Worker
conda activate mmedagent
python -m llava.serve.model_worker --host 0.0.0.0 --controller http://localhost:20001 --port 40000 --worker http://localhost:40000 --model-path .\llava_med_agent

# Window 3: Tool Workers (chạy lần lượt hoặc trong các window riêng)
conda activate mmedagent
python serve\grounding_dino_worker.py

# Window 4: Gradio
conda activate mmedagent
python llava\serve\gradio_web_server_mmedagent.py --controller http://localhost:20001 --model-list-mode reload
```

---

## 8. Cấu Trúc Project

```
MMedAgent/
├── 📄 README.md                    # Tài liệu chính
├── 📄 pyproject.toml               # Dependencies
├── 📄 tuning.sh                    # Script training
├── 📄 merge.sh                     # Script merge LoRA
├── 📄 eval.sh                      # Script evaluation
│
├── 📁 llava/                       # Core LLaVA code
│   ├── 📄 constants.py             # Hằng số
│   ├── 📄 conversation.py          # Conversation templates
│   ├── 📄 mm_utils.py              # Multi-modal utilities
│   ├── 📄 utils.py                 # Utilities
│   │
│   ├── 📁 model/                   # Model architecture
│   │   ├── 📄 builder.py           # Model loading
│   │   ├── 📄 llava_arch.py        # LLaVA architecture
│   │   └── 📄 apply_delta.py       # Apply delta weights
│   │
│   ├── 📁 train/                   # Training code
│   │   ├── 📄 train.py             # Training script
│   │   ├── 📄 train_mem.py         # Memory-efficient training
│   │   └── 📄 llava_trainer.py     # Custom trainer
│   │
│   ├── 📁 eval/                    # Evaluation code
│   │   ├── 📄 model_vqa.py         # VQA inference
│   │   ├── 📄 run_llava.py         # CLI inference
│   │   ├── 📄 eval_gpt4.py         # GPT-4 evaluation
│   │   └── 📄 eval_gpt4o.py        # GPT-4o inference
│   │
│   └── 📁 serve/                   # Serving code
│       ├── 📄 controller.py        # Request controller
│       ├── 📄 model_worker.py      # Model worker
│       ├── 📄 gradio_web_server_mmedagent.py  # Web UI
│       └── 📄 cli.py               # CLI interface
│
├── 📁 serve/                       # Tool workers
│   ├── 📄 grounding_dino_worker.py # Grounding DINO
│   ├── 📄 MedSAM_worker.py         # MedSAM segmentation
│   ├── 📄 grounded_medsam_worker.py# Combined G-Seg
│   ├── 📄 biomedclip_worker.py     # Classification
│   ├── 📄 chatcad_G_worker.py      # Medical report gen
│   └── 📄 chatcad_R_worker.py      # RAG
│
├── 📁 src/                         # External tools
│   ├── 📄 build_sam.py             # SAM builder (vit_b)
│   ├── 📁 ChatCAD_R/               # ChatCAD code
│   ├── 📁 GroundingDINO/           # (clone từ GitHub)
│   └── 📁 MedSAM/                  # (clone từ GitHub)
│
├── 📁 scripts/                     # Helper scripts
│   ├── 📄 merge_lora_weights.py    # Merge LoRA
│   ├── 📄 zero2.json               # DeepSpeed config
│   └── 📄 finetune_lora.sh         # Finetune script
│
├── 📁 data_processing/             # Data preprocessing
│   ├── 📄 dataset_loading.py       # Load datasets
│   ├── 📄 path_writing.py          # Path utilities
│   └── 📄 combine.ipynb            # Combine datasets
│
├── 📁 train_data_json/             # Training data
│   └── 📄 example.jsonl            # Example format
│
├── 📁 eval_data_json/              # Evaluation data
│   └── 📄 eval_example.jsonl       # Example format
│
├── 📁 train_images/                # Training images
└── 📁 eval_images/                 # Evaluation images
```

---

## 9. Troubleshooting

### 9.1 Lỗi Thường Gặp

#### ❌ CUDA Out of Memory (OOM)
```
RuntimeError: CUDA out of memory
```
**Giải pháp:**
```bash
# Giảm batch size
--per_device_train_batch_size 4

# Bật gradient checkpointing
--gradient_checkpointing True

# Dùng 8-bit loading
--load_in_8bit True
```

#### ❌ Flash Attention Installation Failed
```
error: command 'nvcc' failed
```
**Giải pháp:**
```bash
# Bỏ qua flash-attn nếu không cần thiết
# Hoặc cài từ pre-built wheel
pip install flash-attn --no-build-isolation
```

#### ❌ Module Not Found
```
ModuleNotFoundError: No module named 'llava'
```
**Giải pháp:**
```bash
# Cài lại package
pip install -e .
```

#### ❌ Connection Refused (Server)
```
requests.exceptions.ConnectionError: Connection refused
```
**Giải pháp:**
- Kiểm tra controller đã chạy chưa
- Kiểm tra port không bị block
- Đảm bảo khởi động theo đúng thứ tự

### 9.2 Kiểm Tra GPU

```bash
# Kiểm tra GPU
nvidia-smi

# Kiểm tra CUDA trong PyTorch
python -c "import torch; print(torch.cuda.device_count()); print(torch.cuda.get_device_name(0))"
```

### 9.3 Logs & Debug

```bash
# Xem logs chi tiết
export TRANSFORMERS_VERBOSITY=info
export CUDA_LAUNCH_BLOCKING=1

# Chạy với verbose
python llava/eval/model_vqa.py --verbose ...
```

---

## 8. 🎯 Chạy Trên Kaggle

### 8.1 Tổng Quan về Kaggle

**Ưu điểm:**
- ✅ **Miễn phí**: 30 giờ GPU mỗi tuần
- ✅ **GPU mạnh**: T4 (16GB) hoặc P100 (16GB), có thể chọn T4 x2 (32GB)
- ✅ **Không cần setup**: Python, CUDA, PyTorch có sẵn
- ✅ **Internet**: Tải datasets, models trực tiếp
- ✅ **Persistent storage**: Kaggle Datasets (100GB)

**Hạn chế:**
- ⚠️ **Session timeout**: 9-12 giờ/session (idle ~1h tắt)
- ⚠️ **CPU quota**: Chia sẻ với Notebooks khác
- ⚠️ **Web UI cần tunnel**: Kaggle không expose ports trực tiếp (cần ngrok/Gradio share)
- ⚠️ **Storage giới hạn**: 20GB working directory (cần Datasets cho persistent)

**Phù hợp cho:**
- Inference và evaluation
- Training nhỏ (LoRA với small dataset)
- Thử nghiệm và debug
- **Web UI demo** (dùng Gradio share=True hoặc ngrok)

**KHÔNG phù hợp cho:**
- Training lớn (>10 giờ)
- Production deployment
- Web UI với nhiều concurrent users (performance hạn chế)

### 8.2 Setup Ban Đầu

#### Bước 1: Tạo Kaggle Notebook

1. Đăng nhập [Kaggle](https://www.kaggle.com/)
2. Tạo New Notebook
3. Settings → Accelerator → **GPU T4 x2** (32GB recommended)
4. Settings → Internet → **On**
5. Settings → Persistence → **Variables and files**

#### Bước 2: Clone Repository

```python
!git clone https://github.com/Wangyixinxin/MMedAgent.git
%cd MMedAgent
!git lfs install
```

#### Bước 3: Cài Đặt Dependencies

```python
# Cài đặt package chính
!pip install -q -e .

# Check GPU
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB")

# Output mong đợi:
# CUDA available: True
# GPU count: 2
# GPU 0: Tesla T4
#   Memory: 15.0 GB
# GPU 1: Tesla T4
#   Memory: 15.0 GB
```

### 8.3 Tải Models Lên Kaggle Dataset

**⚠️ Quan trọng:** Kaggle working directory bị xóa sau mỗi session. Cần upload models lên **Kaggle Dataset** để persistent.

#### Option A: Upload từ Local (Khuyến nghị)

**Bước 1:** Tải models về local trước
```bash
# Trên máy local
git clone https://huggingface.co/andy0207/mmedagent
git clone https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
```

**Bước 2:** Tạo Kaggle Dataset
1. Vào [kaggle.com/datasets](https://www.kaggle.com/datasets)
2. Click **New Dataset**
3. Upload các thư mục:
   - `mmedagent/` → Dataset: **mmedagent-weights**
   - `BiomedCLIP-PubMedBERT_256-vit_base_patch16_224/` → Dataset: **biomedclip-weights**
4. Click **Create**

**Bước 3:** Add vào Notebook
- Notebook → Add Data → Your Datasets → Select datasets đã tạo

#### Option B: Tải trực tiếp trong Notebook

```python
# Tải MMedAgent weights
!git clone https://huggingface.co/andy0207/mmedagent /kaggle/working/mmedagent-weights

# Tải BiomedCLIP
!git clone https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224 \
    /kaggle/working/biomedclip-weights

# ⚠️ Lưu ý: Files sẽ BỊ XÓA sau khi session kết thúc
# Nên dùng Option A (upload Dataset) cho persistent storage
```

### 8.4 Inference Cơ Bản

#### Setup Model Paths

```python
import os
from pathlib import Path

# Nếu dùng Kaggle Datasets (khuyến nghị)
MMEDAGENT_PATH = "/kaggle/input/mmedagent-weights/final_model_lora"
BIOMEDCLIP_PATH = "/kaggle/input/biomedclip-weights"

# Nếu tải trực tiếp (temporary)
# MMEDAGENT_PATH = "/kaggle/working/mmedagent-weights/final_model_lora"
# BIOMEDCLIP_PATH = "/kaggle/working/biomedclip-weights"

# Verify paths
print(f"MMedAgent exists: {os.path.exists(MMEDAGENT_PATH)}")
print(f"BiomedCLIP exists: {os.path.exists(BIOMEDCLIP_PATH)}")
```

#### Simple Inference

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Load model
model_path = MMEDAGENT_PATH
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=True,  # ← Dùng 8-bit để tiết kiệm VRAM
    device_map="auto"
)

print(f"Model loaded successfully!")
print(f"Context length: {context_len}")

# Run inference
image_file = "/kaggle/input/your-medical-images/xray_001.jpg"
query = "Describe the findings in this chest X-ray"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": query,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

response = eval_model(args)
print(f"\n{'='*60}")
print(f"RESPONSE:\n{response}")
print(f"{'='*60}")
```

### 8.5 Batch Inference (Nhiều Ảnh)

```python
import pandas as pd
from tqdm import tqdm
from PIL import Image
import base64
from io import BytesIO

# Prepare dataset
image_folder = "/kaggle/input/medical-dataset/images"
questions = [
    "Describe the findings in this medical image",
    "What abnormalities can you see?",
    "Is this image normal or abnormal?"
]

image_files = list(Path(image_folder).glob("*.jpg"))[:10]  # Test với 10 ảnh
results = []

# Batch processing
for img_path in tqdm(image_files, desc="Processing images"):
    for question in questions:
        args.image_file = str(img_path)
        args.query = question
        
        try:
            response = eval_model(args)
            results.append({
                "image": img_path.name,
                "question": question,
                "response": response
            })
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            results.append({
                "image": img_path.name,
                "question": question,
                "response": f"ERROR: {str(e)}"
            })

# Save results
df = pd.DataFrame(results)
df.to_csv("inference_results.csv", index=False)
print(f"✅ Saved {len(results)} results to inference_results.csv")
df.head()
```

### 8.6 Evaluation Script

```python
# Chạy evaluation trên test set
!python llava/eval/model_vqa.py \
    --model-path {MMEDAGENT_PATH} \
    --question-file /kaggle/input/eval-data/eval_example.jsonl \
    --image-folder /kaggle/input/eval-data/eval_images \
    --answers-file /kaggle/working/eval_results.jsonl \
    --temperature 0 \
    --conv-mode llava_v1

# Xem results
import json
with open("/kaggle/working/eval_results.jsonl", "r") as f:
    results = [json.loads(line) for line in f]
    
print(f"Processed {len(results)} examples")
print(f"\nFirst result:")
print(json.dumps(results[0], indent=2))
```

### 8.7 Training (LoRA Fine-tuning)

**⚠️ Chỉ phù hợp cho small dataset (<500 samples) do thời gian session giới hạn**

```python
# Chuẩn bị training data
# Format: instruction_all.json (xem section 6.2)

# Training script
!python llava/train/train.py \
    --model_name_or_path {MMEDAGENT_PATH} \
    --version v1 \
    --data_path /kaggle/input/training-data/instruction_all.json \
    --image_folder /kaggle/input/training-data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir /kaggle/working/output_lora \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 2 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    --lora_enable True \
    --lora_r 64 \
    --lora_alpha 128

# ⏱️ Ước tính: ~1-2h cho 500 samples, 3 epochs trên T4 x2
```

### 8.8 Lưu Kết Quả

```python
# Option 1: Download trực tiếp từ Notebook
from IPython.display import FileLink

# Tạo file để download
!zip -r results.zip /kaggle/working/eval_results.jsonl /kaggle/working/output_lora

FileLink('results.zip')
```

```python
# Option 2: Push lên Kaggle Dataset (cho persistent storage)
# Cần cài Kaggle API
!pip install -q kaggle

# Upload API token (kaggle.json) lên Notebook
# Hoặc dùng Kaggle Secrets
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

# Create new dataset version
!kaggle datasets version -p /kaggle/working/output_lora -m "LoRA weights after fine-tuning"
```

### 8.9 Optimization Tips cho Kaggle

#### Memory Management

```python
# 1. Dùng 8-bit loading
load_8bit=True  # Giảm ~50% VRAM

# 2. Dùng smaller batch size
per_device_train_batch_size=1
gradient_accumulation_steps=8  # Equivalent to batch_size=8

# 3. Gradient checkpointing
gradient_checkpointing=True  # Giảm VRAM, tăng ~20% training time

# 4. Clear cache thường xuyên
import torch
torch.cuda.empty_cache()

# 5. Monitor VRAM
import GPUtil
GPUtil.showUtilization()
```

#### Speed Optimization

```python
# 1. Dùng T4 x2 thay vì single GPU
# 2. Tăng num_workers cho dataloader
dataloader_num_workers=4

# 3. Dùng mixed precision
bf16=True  # or fp16=True

# 4. Disable unnecessary logging
report_to="none"
logging_steps=50  # Thay vì 1

# 5. Use smaller context length nếu có thể
model_max_length=1024  # Thay vì 2048
```

### 8.10 Chạy Web UI trên Kaggle

**⚡ MỚI: Kaggle CÓ THỂ chạy Full Web UI từ repo!**

Có 2 options:

#### Option 1: Full Web UI với Controller + Workers (Khuyến nghị)

**Chạy đầy đủ như repo gốc** - bao gồm controller, model worker, và gradio server

```python
# =============================================================================
# MMedAgent Full Web UI on Kaggle
# =============================================================================

# ---- Step 1: Setup ----
!git clone https://github.com/Wangyixinxin/MMedAgent.git
%cd MMedAgent
!pip install -q -e .

import subprocess
import time
import os

# ---- Step 2: Start Controller ----
print("🎮 Starting Controller...")
controller_process = subprocess.Popen(
    ["python", "-m", "llava.serve.controller", "--host", "0.0.0.0", "--port", "20001"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(5)  # Wait for controller to start
print("✅ Controller started at http://0.0.0.0:20001")

# ---- Step 3: Start Model Worker ----
print("🤖 Starting Model Worker...")

MODEL_PATH = "/kaggle/input/mmedagent-weights/final_model_lora"

model_worker_process = subprocess.Popen([
    "python", "-m", "llava.serve.model_worker",
    "--host", "0.0.0.0",
    "--controller", "http://localhost:20001",
    "--port", "40000",
    "--worker", "http://localhost:40000",
    "--model-path", MODEL_PATH,
    "--load-8bit"  # ← Important cho Kaggle GPU
],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(20)  # Wait for model to load (this takes time!)
print("✅ Model Worker started at http://0.0.0.0:40000")

# ---- Step 4: Start Gradio Web Server ----
print("🌐 Starting Gradio Web Server...")

# Chạy gradio server với share=True
web_process = subprocess.Popen([
    "python", "-m", "llava.serve.gradio_web_server_mmedagent",
    "--controller", "http://localhost:20001",
    "--model-list-mode", "once",
    "--share",  # ← Tạo public link
    "--port", "7860"
],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    bufsize=1
)

# Monitor output để lấy share link
print("\n⏳ Waiting for Gradio share link...")
share_link = None
for line in web_process.stdout:
    print(line.strip())  # Print all output
    if "gradio.live" in line or "Running on public URL" in line:
        # Extract URL
        import re
        urls = re.findall(r'https://[^\s]+gradio\.live', line)
        if urls:
            share_link = urls[0]
            print(f"\n{'='*60}")
            print(f"🎉 SUCCESS! MMedAgent Web UI is ready!")
            print(f"🌐 Public URL: {share_link}")
            print(f"{'='*60}\n")
            break

# Keep processes running
if share_link:
    print("✅ Web UI is running. Press Ctrl+C to stop.")
    print(f"🔗 Access at: {share_link}")
    try:
        web_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping services...")
        web_process.terminate()
        model_worker_process.terminate()
        controller_process.terminate()
else:
    print("⚠️ Could not find share link. Check logs above.")
```

**Features của Full Web UI:**
- ✅ **Full interface** như local (examples, parameters, debug mode)
- ✅ **Tool integration** sẵn (nếu có tool workers)
- ✅ **Conversation history** được maintain
- ✅ **Public share link** tự động
- ⚠️ **Startup time**: ~30-60s (load model + register workers)

#### Option 2: Simplified Gradio Interface (Nhanh hơn)

**Nếu chỉ cần inference đơn giản**, không cần full controller/worker architecture:

```python
# =============================================================================
# MMedAgent Simplified Web UI on Kaggle
# =============================================================================

# ---- Setup ----
!git clone https://github.com/Wangyixinxin/MMedAgent.git
%cd MMedAgent
!pip install -q -e .

# ---- Load Model ----
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import gradio as gr
import torch

model_path = "/kaggle/input/mmedagent-weights/final_model_lora"
print("🔄 Loading model...")
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=True,
    device_map="auto"
)
print("✅ Model loaded!")

# ---- Inference Function ----
def inference(image, question, temperature=0.2, max_tokens=512):
    """Run inference on image + question"""
    if image is None:
        return "⚠️ Please upload an image first!"
    
    # Prepare conversation
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    # Process image
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    
    # Tokenize
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).cuda()
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor,
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            max_new_tokens=max_tokens,
            use_cache=True,
        )
    
    # Decode
    outputs = tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()
    return outputs

# ---- Build Gradio UI ----
with gr.Blocks(title="MMedAgent on Kaggle", theme=gr.themes.Base()) as demo:
    gr.Markdown("# 🏥 MMedAgent - Medical Image Analysis")
    gr.Markdown("Multi-modal Medical Agent running on Kaggle GPU")
    
    with gr.Row():
        with gr.Column(scale=3):
            image_input = gr.Image(type="pil", label="Upload Medical Image")
            question_input = gr.Textbox(
                label="Question",
                placeholder="Ask about the medical image...",
                value="Describe the findings in this medical image.",
                lines=3
            )
            
            with gr.Accordion("Parameters", open=False):
                temperature = gr.Slider(0, 1, value=0.2, step=0.1, label="Temperature")
                max_tokens = gr.Slider(128, 1024, value=512, step=64, label="Max Tokens")
            
            submit_btn = gr.Button("🔍 Analyze", variant="primary")
            clear_btn = gr.Button("🗑️ Clear")
        
        with gr.Column(scale=7):
            output = gr.Textbox(label="Analysis Result", lines=20, show_copy_button=True)
    
    # Examples
    gr.Examples(
        examples=[
            [None, "Describe the findings in this chest X-ray."],
            [None, "What abnormalities can you see in this image?"],
            [None, "Is this medical image normal or abnormal?"],
            [None, "Can you identify the anatomical structures in this image?"],
        ],
        inputs=[image_input, question_input],
        label="Example Questions"
    )
    
    # Events
    submit_btn.click(
        fn=inference,
        inputs=[image_input, question_input, temperature, max_tokens],
        outputs=output
    )
    clear_btn.click(
        fn=lambda: (None, "", ""),
        outputs=[image_input, question_input, output]
    )

# ---- Launch with Public Link ----
print("\n🚀 Launching Gradio interface...")
print("⏳ Generating public share link (10-30 seconds)...\n")

demo.launch(
    share=True,           # ← Tạo public link
    debug=True,
    server_name="0.0.0.0",
    server_port=7860,
    show_error=True
)

# ✅ Link sẽ hiện dạng:
# Running on public URL: https://abc123xyz.gradio.live
```

### Comparison: Full vs Simplified

| Feature | Full Web UI (Option 1) | Simplified (Option 2) |
|---------|------------------------|----------------------|
| **Setup complexity** | ⚠️ Phức tạp (3 processes) | ✅ Đơn giản (1 process) |
| **Startup time** | 🐌 30-60s | 🚀 10-20s |
| **Features** | ✅ Full (tools, debug, examples) | ⚠️ Basic VQA only |
| **Tool support** | ✅ Có (nếu setup workers) | ❌ Không |
| **Memory usage** | ⚠️ Cao hơn | ✅ Thấp hơn |
| **Recommended for** | Production demo | Quick testing |

**Lưu ý về Gradio Share:**
- ✅ Link dạng `https://xxxxx.gradio.live`
- ✅ Tồn tại ~72 giờ hoặc khi Kaggle session kết thúc
- ✅ Miễn phí, không cần API key
- ⚠️ Tốc độ có thể chậm hơn local (traffic qua Gradio servers)
- ✅ Share được cho người khác dùng

#### Option 2: Ngrok (Cho Port Forwarding)

**Ưu điểm:** Stable hơn, tự custom domain

```python
# ---- Cài đặt ngrok ----
!pip install -q pyngrok

# ---- Get ngrok token ----
# 1. Đăng ký miễn phí tại: https://dashboard.ngrok.com/signup
# 2. Copy token từ: https://dashboard.ngrok.com/get-started/your-authtoken

from pyngrok import ngrok
import os

# Set your ngrok token
NGROK_TOKEN = "YOUR_NGROK_TOKEN_HERE"  # ← Thay bằng token của bạn
os.environ["NGROK_AUTHTOKEN"] = NGROK_TOKEN
ngrok.set_auth_token(NGROK_TOKEN)

# ---- Launch Gradio in background ----
import threading

def launch_gradio():
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
        prevent_thread_lock=True  # ← Important!
    )

thread = threading.Thread(target=launch_gradio)
thread.start()

# Wait for server to start
import time
time.sleep(10)

# ---- Create ngrok tunnel ----
public_url = ngrok.connect(7860)
print(f"\n{'='*60}")
print(f"✅ MMedAgent Web UI is running!")
print(f"🌐 Public URL: {public_url}")
print(f"{'='*60}\n")

# Keep running
while True:
    time.sleep(60)
    print("🟢 Server running... (Ctrl+C to stop)")
```

**Lưu ý về Ngrok:**
- Cần đăng ký account miễn phí
- Free tier: 1 tunnel, tồn tại 2h, sau đó cần restart
- Link dạng `https://xxxxx.ngrok.io`
- Stable hơn Gradio share
- Paid ($8/tháng): unlimited tunnels, custom domain

#### Option 3: Localtunnel (Không cần đăng ký)

```python
# ---- Cài đặt localtunnel ----
!npm install -g localtunnel

# ---- Launch Gradio ----
import threading

def launch_gradio():
    demo.launch(
        share=False,
        server_name="0.0.0.0",
        server_port=7860,
        prevent_thread_lock=True
    )

thread = threading.Thread(target=launch_gradio)
thread.start()

import time
time.sleep(10)

# ---- Create tunnel ----
!lt --port 7860 --subdomain mmedagent

# Link sẽ là: https://mmedagent.loca.lt
```

### 8.11 So Sánh 3 Options

| Feature | Gradio Share | Ngrok | Localtunnel |
|---------|--------------|-------|-------------|
| **Setup** | ⭐⭐⭐ Dễ nhất | ⭐⭐ Cần token | ⭐⭐ Cần npm |
| **Tốc độ** | 🐌 Chậm | 🚀 Nhanh | 🚀 Nhanh |
| **Miễn phí** | ✅ Hoàn toàn | ⚠️ 2h/session | ✅ Hoàn toàn |
| **Thời gian tồn tại** | ~72h | 2h (free) | Unlimited |
| **Public link** | ✅ Có | ✅ Có | ✅ Có |
| **Khuyến nghị** | ⭐⭐⭐ Best | ⭐⭐ Nếu cần stable | ⭐ Alternative |

**Khuyến nghị:**
- **Demo nhanh**: Dùng **Gradio Share** (`share=True`) - 1 dòng code!
- **Stable/share lâu**: Dùng **Ngrok** (cần token)
- **Production**: Deploy trên Hugging Face Spaces hoặc local server

### 8.12 Kaggle Notebook Template (Simple Inference)

**Complete working example:**

```python
# =============================================================================
# MMedAgent Simple Inference on Kaggle
# =============================================================================

# ---- Setup ----
!git clone https://github.com/Wangyixinxin/MMedAgent.git
%cd MMedAgent
!pip install -q -e .

# ---- Check GPU ----
import torch
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Load Model ----
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path

model_path = "/kaggle/input/mmedagent-weights/final_model_lora"
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=True,
    device_map="auto"
)

# ---- Inference ----
from llava.eval.run_llava import eval_model

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": "Describe this medical image",
    "conv_mode": None,
    "image_file": "/kaggle/input/test-images/sample.jpg",
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

response = eval_model(args)
print(f"Response: {response}")

# ---- Save Results ----
import json
with open("inference_result.json", "w") as f:
    json.dump({"query": args.query, "response": response}, f, indent=2)

print("✅ Done! Download inference_result.json")
```

### 8.13 Troubleshooting Kaggle-Specific

| Issue | Cause | Solution |
|-------|-------|----------|
| **CUDA OOM** | Model + data > 15GB | Use `load_8bit=True`, reduce batch size |
| **Session timeout** | Idle >1h | Add periodic print statements |
| **Internet disabled** | Forgot to enable | Settings → Internet → On |
| **Module not found** | Package not installed | `!pip install -e .` trong notebook |
| **Dataset not found** | Path incorrect | Check `/kaggle/input/<dataset-name>/` |
| **Slow loading** | CPU bottleneck | Use `dataloader_num_workers=4` |
| **Gradio share failed** | Network issue | Restart kernel, try ngrok instead |
| **Ngrok tunnel died** | 2h free limit | Restart tunnel or upgrade to paid |
| **Port already in use** | Previous session | Kill process: `!kill $(lsof -t -i:7860)` |
| **Web UI slow** | Kaggle CPU limited | Normal, dùng local nếu cần fast |
| **⭐ NETWORK ERROR DUE TO HIGH TRAFFIC** | Gradio server quá tải / Model chưa load xong | Xem section 8.13.1 bên dưới |

#### 8.13.1 Khắc Phục Lỗi "NETWORK ERROR DUE TO HIGH TRAFFIC"

Lỗi này thường xảy ra khi chạy Full Web Demo. **Nguyên nhân:**
1. **Gradio share servers quá tải** - Server của Gradio đang có nhiều người dùng
2. **Model Worker chưa load xong** - Gradio khởi động trước khi model sẵn sàng
3. **Timeout kết nối** - Kaggle network không ổn định

**Giải pháp 1: Tăng thời gian chờ model load**

```python
import subprocess
import time
import requests

# 1. Start Controller
controller = subprocess.Popen(
    ["python", "-m", "llava.serve.controller", "--host", "0.0.0.0", "--port", "20001"],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE
)
time.sleep(10)

# 2. Start Model Worker
worker = subprocess.Popen([
    "python", "-m", "llava.serve.model_worker",
    "--host", "0.0.0.0",
    "--controller", "http://localhost:20001",
    "--port", "40000",
    "--worker", "http://localhost:40000",
    "--model-path", "/kaggle/input/mmedagent-weights/final_model_lora",
    "--load-8bit"
], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# ⭐ QUAN TRỌNG: Chờ 60-90s để model load xong
print("⏳ Đợi model load (60-90s)...")
time.sleep(90)

# 3. Kiểm tra model đã sẵn sàng
for i in range(10):
    try:
        r = requests.post("http://localhost:20001/list_models", json={}, timeout=30)
        if len(r.json().get("models", [])) > 0:
            print("✅ Model sẵn sàng!")
            break
    except:
        time.sleep(10)

# 4. Chỉ start Gradio SAU KHI model đã load
!python llava/serve/gradio_web_server_mmedagent.py \
    --controller http://localhost:20001 \
    --model-list-mode once \
    --share \
    --concurrency-count 2
```

**Giải pháp 2: Dùng Simplified Interface (Ổn định nhất)**

Không dùng Controller/Worker architecture, load model trực tiếp:

```python
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path, tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
import gradio as gr
import torch

# Load model trực tiếp
model_path = "/kaggle/input/mmedagent-weights/final_model_lora"
tokenizer, model, image_processor, _ = load_pretrained_model(
    model_path=model_path, model_base=None,
    model_name=get_model_name_from_path(model_path),
    load_8bit=True, device_map="auto"
)

def inference(image, question):
    if image is None: return "Upload ảnh trước!"
    conv = conv_templates["llava_v1"].copy()
    conv.append_message(conv.roles[0], f"<image>\n{question}")
    conv.append_message(conv.roles[1], None)
    
    image_tensor = image_processor.preprocess(image, return_tensors='pt')['pixel_values'].half().cuda()
    input_ids = tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).cuda()
    
    with torch.inference_mode():
        output_ids = model.generate(input_ids, images=image_tensor, max_new_tokens=512, use_cache=True)
    return tokenizer.decode(output_ids[0, input_ids.shape[1]:]).strip()

# Gradio đơn giản
with gr.Blocks() as demo:
    gr.Markdown("# 🏥 MMedAgent")
    image = gr.Image(type="pil")
    question = gr.Textbox(value="Describe this medical image")
    output = gr.Textbox(lines=10)
    gr.Button("Analyze").click(inference, [image, question], output)

demo.queue(max_size=5).launch(share=True, max_threads=2)
```

**Giải pháp 3: Dùng ngrok thay Gradio share**

Nếu Gradio share server vẫn lỗi:

```python
!pip install -q pyngrok
from pyngrok import ngrok

# Lấy token từ: https://dashboard.ngrok.com/signup
ngrok.set_auth_token("YOUR_TOKEN")

# Launch không share
import threading
threading.Thread(target=lambda: demo.launch(share=False, server_port=7860, prevent_thread_lock=True)).start()
import time; time.sleep(10)

# Tạo tunnel
print(f"🌐 URL: {ngrok.connect(7860)}")
```

### 8.14 Best Practices

✅ **DO:**
- Upload models to Kaggle Datasets (persistent)
- Use 8-bit loading to save VRAM
- Monitor GPU usage with `GPUtil.showUtilization()`
- Save results frequently (session timeout)
- Use T4 x2 for training
- **Use `share=True` cho Web UI demo** ← Cách dễ nhất!
- Test Web UI locally trước khi deploy lên Kaggle

❌ **DON'T:**
- Don't download large models every session (use Datasets)
- Don't train for >8 hours (session timeout risk)
- Don't use full precision (waste VRAM)
- Don't store results in /kaggle/working only (will be deleted)
- Don't expect high performance cho Web UI (Kaggle CPU limited)
- Don't share sensitive medical data qua public links

---

## 📝 Quick Reference

### Các Lệnh Thường Dùng

```bash
# Activate environment
conda activate mmedagent

# Quick inference test
python llava/eval/run_llava.py \
    --model-path ./llava_med_agent \
    --image-file ./eval_images/test.jpg \
    --query "Describe this medical image"

# Merge LoRA weights
bash merge.sh

# Start training
bash tuning.sh

# Run evaluation
bash eval.sh
```

### Ports Mặc Định

| Service | Port |
|---------|------|
| Controller | 20001 |
| Model Worker | 40000 |
| Tool Workers | 21001+ |
| Gradio Web UI | 7860 |

---

## 📚 Tài Liệu Tham Khảo

- [Paper (EMNLP 2024)](https://arxiv.org/abs/2407.02483)
- [HuggingFace Model](https://huggingface.co/andy0207/mmedagent)
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [LLaVA-Plus](https://llava-vl.github.io/llava-plus/)

---

*Tài liệu được tạo ngày 03/12/2025*
