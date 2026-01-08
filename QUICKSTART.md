# ⚡ Quick Start Guide - TriMed-Agent V2

## 🎯 Tổng quan

TriMed-Agent V2 hỗ trợ 3 chế độ:

| Mode | Description | GPU Required |
|------|-------------|--------------|
| **Demo** | Chạy inference với pretrained | T4 (16GB) |
| **SFT Training** | Fine-tune LoRA adapter | A100/L4 (40GB+) |
| **RL Training** | GRPO optimization | A100/L4 (40GB+) |

---

## 🚀 Mode 1: Demo (Inference Only)

### Step 1: Google Colab Setup

```bash
# Clone repo
!git clone https://github.com/ngnam1104/TriMedAgent.git
%cd TriMedAgent

# Install
!pip install -e .
```

### Step 2: Load Models

```python
from src import TriMedOrchestrator
from PIL import Image

# Initialize (sử dụng pretrained hoặc custom adapter)
config = {
    "device": {"llava_device": "cuda:0"},
    "models": {
        "llava": {"model_name": "chaoyinshe/llava-med-v1.5-mistral-7b-hf"},
        # Optional: Load your trained adapter
        # "lora_adapter": "ngnam1104/TriMedAgent-V2/adapters/rl-v1.0"
    }
}

agent = TriMedOrchestrator(config)
```

### Step 3: Run Inference

```python
# Diagnosis query
image = Image.open("chest_xray.jpg")
result = agent.process(image, "Có tổn thương phổi không?")
print(result.final_report)

# Theory query (uses RAG)
from src import MedicalRAG
rag = MedicalRAG()
answer = rag.query("Triệu chứng viêm phổi là gì?")
```

---

## 🎓 Mode 2: SFT Training (Stage 1)

### Step 1: Prepare Dataset

```bash
# Download VQA-RAD hoặc tự tạo dataset
# Format: JSON Lines với cấu trúc conversations

# data/sft_dataset/train.jsonl
{"image": "path/to/img.jpg", "conversations": [
    {"role": "user", "content": "<image>\nTim có to không?"},
    {"role": "assistant", "content": "{\"thought\": \"Cần đo CTR\", \"action\": \"GroundingDINO\", \"action_input\": {\"prompt\": \"heart\"}}"}
]}
```

### Step 2: Run SFT Training

```bash
# Trên Colab/Kaggle với A100
python scripts/train_sft.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --dataset "data/sft_dataset" \
    --output_dir "outputs/sft-v1" \
    --lora_r 64 \
    --lora_alpha 128 \
    --num_epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4
```

### Step 3: Push to HuggingFace

```bash
# Upload adapter
huggingface-cli login
huggingface-cli upload ngnam1104/TriMedAgent-V2 outputs/sft-v1 adapters/sft-v1.0
```

---

## 🎮 Mode 3: RL Training (Stage 2)

### Prerequisites
- Đã có SFT adapter từ Stage 1
- Dataset với ground truth boxes (cho IoU reward)

### Step 1: Configure Rewards

```yaml
# configs/rl_config.yaml
reward:
  iou_weight: 0.3      # IoU với ground truth
  acc_weight: 0.4      # Accuracy của answer
  format_weight: 0.2   # JSON valid
  step_penalty: -0.1   # Penalty mỗi step
```

### Step 2: Run GRPO Training

```bash
python scripts/train_rl_grpo.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --sft_adapter "outputs/sft-v1" \
    --dataset "data/rl_dataset" \
    --output_dir "outputs/rl-v1" \
    --group_size 4 \
    --num_epochs 1
```

### Step 3: Merge & Deploy

```bash
# Merge LoRA vào base model
python scripts/merge_adapters.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --adapter "outputs/rl-v1" \
    --output "outputs/merged-v1"

# Upload merged model
huggingface-cli upload ngnam1104/TriMedAgent-V2 outputs/merged-v1 adapters/merged-v1.0
```

---

## 📦 Dependencies

```bash
# Core
torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
bitsandbytes>=0.41.0
peft>=0.7.0           # LoRA
trl>=0.7.0            # GRPO training

# Vision Tools
pillow>=10.0.0
segment-anything
groundingdino-py

# RAG
sentence-transformers
faiss-cpu             # Vector DB
groq                  # LLM API

# UI
gradio>=4.0.0
```

---

## 🔧 Troubleshooting

### CUDA OOM during Training

```python
# Giảm batch size
batch_size = 1

# Enable gradient checkpointing
gradient_checkpointing = True

# Use 8-bit optimizer
use_8bit_adam = True
```

### LoRA not loading

```python
# Check adapter path
from peft import PeftModel
model = PeftModel.from_pretrained(
    base_model, 
    "path/to/adapter",
    is_trainable=False  # For inference
)
```

### JSON parsing fails

```python
# Model outputs malformed JSON
# → Cần thêm data SFT với nhiều examples hơn
# → Tăng format_weight trong RL reward
```

---

## 📊 Expected Results

| Stage | Metric | Target |
|-------|--------|--------|
| SFT | JSON Valid Rate | > 95% |
| SFT | Action Accuracy | > 80% |
| RL | IoU Score | > 0.5 |
| RL | Task Success | > 70% |

---

## 📚 Next Steps

1. **Đọc Architecture**: [docs/ARCHITECTURE_V2.md](docs/ARCHITECTURE_V2.md)
2. **Xem Training Details**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)
3. **Contribute**: Open PR on GitHub!
