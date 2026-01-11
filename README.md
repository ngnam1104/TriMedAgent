# 🏥 TriMed-Agent: State-based Medical AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-red.svg)](http://www.apache.org/licenses/LICENSE-2.0)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)
[![HuggingFace](https://img.shields.io/badge/🤗-Models-yellow.svg)](https://huggingface.co/)

> **TriMed-Agent** - Hệ thống AI Y tế với kiến trúc **State-based ReAct**
> 
> 🧠 **Logic Router** → 📚 **RAG/Planner** → 🔧 **ToolKit** → ✅ **Verification**

---

## 🌟 What's New 

| Feature  | (State-based V2) |
|---------|------------------|
| **Architecture** | Two-Path (Image-Aware Routing) |
| **Pre-Classification** | ✅ BiomedCLIP → Context |
| **Knowledge** | RAG (auto if no image) + Boost for hard cases |
| **Planning** | LLaVA-Med + LoRA + GRPO (4-field JSON) |
| **Training** | **SFT → RL (GRPO)** with RAG rewards |
| **Small Objects** | Smart Zoom + Fallback |
| **Image-Aware** | ✅ Penalty/Bonus based on image availability |

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         TriMed-Agent System                                  │
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   User Input (Query ± Image)                                                 │
│         │                                                                    │
│         ▼                                                                    │
│   ┌─────────────────────────────────────┐                                    │
│   │      IMAGE AVAILABILITY CHECK       │                                    │
│   │         has_image = ?               │                                    │
│   └──────────────┬──────────────────────┘                                    │
│                  │                                                           │
│     ┌────────────┴────────────┐                                              │
│     │                         │                                              │
│     ▼ (No Image)              ▼ (Has Image)                                  │
│ ┌─────────────┐    ┌──────────────────────────────────────────────────────┐  │
│ │ 🧠 Medical  │    │                 VISION PIPELINE                      │  │
│ │    RAG      │    │  ┌────────────────────────────────────────────────┐  │  │
│ │             │    │  │  Step 1: BiomedCLIP (Image Classification)     │  │  │
│ │  MedQuAD    │    │  │  → Modality: X-ray/CT/MRI                      │  │  │
│ │  + FAISS    │    │  │  → Anatomy: Chest/Brain/Abdomen                │  │  │
│ │             │    │  │  → Context for Planner                         │  │  │
│ │ "What are   │    │  └────────────────────┬───────────────────────────┘  │  │
│ │  symptoms   │    │                       │                              │  │
│ │  of pneumo- │    │                       ▼                              │  │
│ │  nia?"      │    │  ┌────────────────────────────────────────────────┐  │  │
│ │             │    │  │  Step 2: PLANNER (LLaVA-Med + LoRA + GRPO)     │  │  │
│ └──────┬──────┘    │  │  ┌──────────────────────────────────────────┐  │  │  │
│        │           │  │  │            ReAct Loop                    │  │  │  │
│        │           │  │  │  ┌─────────────────────────────────────┐ │  │  │  │
│        │           │  │  │  │ Thought: "Cần detect tổn thương..." │ │  │  │  │
│        │           │  │  │  │ Tool: "Vision"                      │ │  │  │  │
│        │           │  │  │  │ Action: "GroundingDINO"             │ │  │  │  │
│        │           │  │  │  │ Action_Input: {"prompt": "nodule"}  │ │  │  │  │
│        │           │  │  │  └─────────────────────────────────────┘ │  │  │  │
│        │           │  │  └──────────────────────────────────────────┘  │  │  │
│        │           │  └────────────────────┬───────────────────────────┘  │  │
│        │           │                       │                              │  │
│        │           │                       ▼                              │  │
│        │           │  ┌────────────────────────────────────────────────┐  │  │
│        │           │  │  Step 3: TOOLKIT (Execute Actions)             │  │  │
│        │           │  │  ┌──────────────────────────────────────────┐  │  │  │
│        │           │  │  │ GroundingDINO │ MedSAM │ Zoom │ Gate     │  │  │  │
│        │           │  │  └──────────────────────────────────────────┘  │  │  │
│        │           │  │              │                                 │  │  │
│        │           │  │              ▼ Observation                     │  │  │
│        │           │  │  ┌──────────────────────────────────────────┐  │  │  │
│        │           │  │  │ boxes: [[x1,y1,x2,y2]], masks: [...]     │  │  │  │
│        │           │  │  └──────────────────────────────────────────┘  │  │  │
│        │           │  └───────────────────────────────────────────────┘   │  │
│        │           │                       ↑                              │  │
│        │           │     ┌─────────────────┼                              │  │
│        │           │     │ (Hard/Medium)   │                              │  │
│        │           │     ▼                 │                              │  │
│        │     ┌───────────────┐             │                              │  │
│        │     │   RAG Boost   │             |                              │  │
│        │     │ "bilateral    │             │                              │  │
│        │     │  consolidation│────────────>┤                              │  │
│        │     │  differential"│             │                              │  │
│        │     └───────────────┘             │                              │  │
│        │           |                       |                              │  │
│        │           │                       |                              │  │
│        │           └───────────────────────┼──────────────────────────────┘  |
|        |                                   ▼                                 |
|        |              ──────────────────────────────────────────────────     | 
│        │             │  Step 4: LLaVA-Med (Response Synthesis)        │      │
│         ─────────>   │  Context + Observation + RAG → Final Answer    │      │
│                      └────────────────────┬───────────────────────────┘      │
│                                           │                                  │
│                                           │                                  |
│                                           ▼                                  │
│                                   ┌─────────────────┐                        │
│                                   │  Final Response │                        │
│                                   └─────────────────┘                        │
└──────────────────────────────────────────────────────────────────────────────┘
```

### 🔄 Pipeline Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        TWO-PATH ARCHITECTURE                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH A: No Image (Theory Questions)                                        │
│  ════════════════════════════════════                                       │
│  Query ──► Medical_RAG ──► MedQuAD/FAISS ──► Direct Answer                  │
│                                                                             │
│  Example: "Triệu chứng viêm phổi là gì?"                                    │
│           → RAG retrieves from medical knowledge base                       │
│           → Returns structured medical information                          │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  PATH B: With Image (Visual Diagnosis)                                      │
│  ═════════════════════════════════════                                      │
│                                                                             │
│  ┌─────────┐    ┌────────────┐    ┌──────────┐    ┌─────────┐    ┌───────┐ │
│  │ Image + │───►│ BiomedCLIP │───►│ Planner  │───►│ Toolkit │───►│LLaVA- │ │
│  │ Query   │    │ (Context)  │    │ (ReAct)  │    │ (Tools) │    │Med    │ │
│  └─────────┘    └────────────┘    └──────────┘    └─────────┘    └───────┘ │
│                       │                │               │             │      │
│                       ▼                ▼               ▼             ▼      │
│                 "chest_xray,      JSON Plan:      Observations:   Final    │
│                  lung_opacity"    {thought,       boxes, masks,   Report   │
│                                   tool, action,   confidence               │
│                                   action_input}                            │
│                                        │                                   │
│                              ┌─────────┴─────────┐                         │
│                              │  Difficulty?      │                         │
│                              │  Hard/Medium      │                         │
│                              └─────────┬─────────┘                         │
│                                        ▼                                   │
│                              ┌──────────────────┐                          │
│                              │  🧠 RAG Boost    │                          │
│                              │  +0.1~0.2 reward │                          │
│                              └──────────────────┘                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

📖 **Chi tiết kiến trúc**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)

---

## 🎯 Key Features

### 1. Two-Path Architecture (Image-Aware Routing)
```python
# Automatic routing based on image availability
if has_image:
    # PATH B: Visual Pipeline
    context = BiomedCLIP.classify(image)  # Step 1: Get image context
    plan = Planner.react(query, context)   # Step 2: Generate JSON plan
    obs = Toolkit.execute(plan)            # Step 3: Run tools
    response = LLaVA.synthesize(obs)       # Step 4: Final response
else:
    # PATH A: Knowledge Pipeline
    response = MedicalRAG.query(question)  # Direct RAG retrieval
```

### 2. BiomedCLIP Pre-Classification
```python
# Image context extraction BEFORE planning
from src.modules.vision import BiomedCLIP

clip = BiomedCLIP()
context = clip.classify(image)
# Returns: {"modality": "X-ray", "anatomy": "chest", "finding": "opacity"}

# Context is injected into Planner prompt
prompt = f"Image context: {context}\nQuery: {query}"
```

### 3. Planner with 4-Field JSON Output
```python
# LLaVA-Med (LoRA + GRPO) generates structured plans
{
    "thought": "Cần detect vùng tổn thương trong phổi",
    "tool": "Vision",
    "action": "GroundingDINO",
    "action_input": {"prompt": "lung opacity, consolidation"}
}
```

### 4. RAG Boost for Hard Cases
```python
# Automatic RAG augmentation for difficult cases
if difficulty in ["hard", "medium"] and needs_medical_knowledge:
    rag_context = MedicalRAG.query("bilateral consolidation differential")
    # RAG knowledge is combined with visual observations
    # Reward bonus: +0.1 (hard) / +0.05 (medium)
```

### 5. Two-Stage Training Pipeline
```
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: SFT (LoRA)                                            │
│  ─────────────────────                                          │
│  • Train LLaVA-Med to output 4-field JSON format                │
│  • Dataset: Medical VQA + Detection pairs                       │
│  • Output: JSON plans for tool orchestration                    │
├─────────────────────────────────────────────────────────────────┤
│  Stage 2: RL (GRPO)                                             │
│  ─────────────────────                                          │
│  • Optimize policy with reward function                         │
│  • RAG bonus for appropriate knowledge retrieval                │
│  • Image-aware penalties (no image → don't use vision tools)    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
TriMedAgent/
├── src/
│   ├── core/
│   │   ├── orchestrator.py      # Main ReAct loop
│   │   ├── logic_router.py      # Intent classification
│   │   └── state.py             # Agent state
│   ├── modules/
│   │   ├── brain/
│   │   │   └── planner.py       # LLaVA Planner
│   │   ├── knowledge/
│   │   │   └── rag_engine.py    # RAG with PubMedBERT
│   │   ├── vision/
│   │   │   ├── biomed_clip.py
│   │   │   ├── dino_detector.py
│   │   │   └── zoom_processor.py
│   │   └── segmentation/
│   │       └── medsam_wrapper.py
│   └── training/
│       ├── sft_trainer.py       # SFT with LoRA
│       └── grpo_trainer.py      # GRPO RL
├── scripts/
│   ├── train_sft.py             # 🆕 SFT training script
│   ├── train_rl_grpo.py         # 🆕 RL training script
│   └── merge_adapters.py
├── configs/
│   ├── config.yaml
│   ├── sft_config.yaml          # 🆕 SFT config
│   └── rl_config.yaml           # 🆕 RL config
├── docs/
│   └── ARCHITECTURE.md       # 🆕 Detailed architecture
└── demo_colab_1xt4.ipynb
```

---

## 🚀 Quick Start

### Option 1: Google Colab (Recommended)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/)

```bash
# Upload demo_colab_1xt4.ipynb to Colab
# Select Runtime → T4 GPU → Run all cells
```

### Option 2: Local Installation

```bash
git clone https://github.com/ngnam1104/TriMedAgent.git
cd TriMedAgent
pip install -e .
```

---

## 💻 Usage

### Basic Inference

```python
from src import TriMedOrchestrator
from PIL import Image

# Initialize
config = {"device": {"llava_device": "cuda:0"}}
agent = TriMedOrchestrator(config)

# Process
image = Image.open("chest_xray.jpg")
result = agent.process(image, "Có tổn thương phổi không?")

print(result.final_report)
```

### With RAG (Theoretical Questions)

```python
from src import MedicalRAG

rag = MedicalRAG(api_key="your-groq-key")
response = rag.query("Viêm phổi do vi khuẩn điều trị như thế nào?")
print(response.answer)
```

---

## 🤖 Load Trained Model from HuggingFace

### Installation

```bash
pip install transformers peft accelerate bitsandbytes torch
pip install sentence-transformers faiss-cpu  # For RAG
pip install Pillow  # For image processing
```

### Load TriMedAgent GRPO Model

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# ============================================
# 1. Load Base Model + GRPO Adapter
# ============================================
BASE_MODEL = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
GRPO_ADAPTER = "nhn309261/trimedagent-grpo-v1"

print("🚀 Loading TriMedAgent GRPO Model...")

# Quantization config (for GPU with limited VRAM)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)

# Load GRPO adapter
model = PeftModel.from_pretrained(base_model, GRPO_ADAPTER)
model.eval()

print("✅ Model loaded successfully!")
```

---

### 📷 Case 1: Visual Question Answering (VQA) - With Image

```python
import json
from PIL import Image

def inference_vqa(image_path: str, question: str) -> dict:
    """
    Visual Question Answering with TriMedAgent
    
    Args:
        image_path: Path to medical image (X-ray, CT, MRI)
        question: Medical question about the image
    
    Returns:
        JSON plan with thought, tool, action, action_input
    """
    # Load and preprocess image (if using full LLaVA pipeline)
    image = Image.open(image_path).convert("RGB")
    
    # Format prompt
    prompt = f"""<s>[INST] You are a medical AI assistant. Analyze the medical image and answer the question.

Question: {question}

Respond in JSON format with 4 fields:
- thought: Your reasoning process
- tool: "Vision" or "Knowledge"  
- action: Tool name (GroundingDINO, MedSAM, Medical_RAG, etc.)
- action_input: Parameters for the tool

[/INST]"""
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON from response
    import re
    match = re.search(r'\{.*\}', response, re.DOTALL)
    if match:
        try:
            plan = json.loads(match.group(0))
            return plan
        except json.JSONDecodeError:
            pass
    
    return {"raw_response": response}

# ============================================
# Example: Detect lung nodules
# ============================================
result = inference_vqa(
    image_path="chest_xray.jpg",
    question="Tìm các nốt mờ bất thường trong phổi và xác định vị trí"
)

print("🔍 VQA Result:")
print(json.dumps(result, indent=2, ensure_ascii=False))

# Expected output:
# {
#     "thought": "Cần phát hiện các nốt mờ trong ảnh X-quang phổi",
#     "tool": "Vision",
#     "action": "GroundingDINO",
#     "action_input": {"prompt": "lung nodule, pulmonary nodule"}
# }
```

---

### 📚 Case 2: RAG - Theoretical Medical Questions (No Image)

```python
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import numpy as np

# ============================================
# 2.1 Setup Medical RAG System
# ============================================
print("🧠 Setting up Medical RAG...")

# Load MedQuAD knowledge base
medquad = load_dataset("keivalya/MedQuad-MedicalQnADataset", split="train")
medquad_answers = [item['Answer'] for item in medquad]
medquad_questions = [item['Question'] for item in medquad]

# Load embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Build FAISS index
print("📦 Building FAISS index...")
answer_embeddings = embedding_model.encode(medquad_answers, show_progress_bar=True)
dimension = answer_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(answer_embeddings)
faiss_index.add(answer_embeddings)

print(f"✅ RAG System ready! Index size: {faiss_index.ntotal}")

def medical_rag_search(query: str, top_k: int = 3) -> list:
    """Search medical knowledge base"""
    query_embedding = embedding_model.encode([query])
    faiss.normalize_L2(query_embedding)
    scores, indices = faiss_index.search(query_embedding, top_k)
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        results.append({
            'question': medquad_questions[idx],
            'answer': medquad_answers[idx],
            'score': float(score)
        })
    return results

# ============================================
# 2.2 Inference with RAG
# ============================================
def inference_rag(question: str) -> dict:
    """
    Answer theoretical medical questions using RAG
    
    Args:
        question: Medical theory question (no image needed)
    
    Returns:
        JSON plan + RAG-augmented answer
    """
    # Step 1: Model generates plan
    prompt = f"""<s>[INST] You are a medical AI assistant. Answer the following medical question.

Question: {question}

Since no image is provided, use the Medical_RAG tool to retrieve relevant medical knowledge.

Respond in JSON format:
- thought: Your reasoning
- tool: "Knowledge"
- action: "Medical_RAG"
- action_input: {{"query": "search query for knowledge base"}}

[/INST]"""
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
            top_p=0.9
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract JSON plan
    import re
    match = re.search(r'\{.*\}', response, re.DOTALL)
    plan = {}
    if match:
        try:
            plan = json.loads(match.group(0))
        except:
            plan = {"raw": response}
    
    # Step 2: Execute RAG if action is Medical_RAG
    rag_results = []
    if plan.get('action', '').lower() in ['medical_rag', 'rag']:
        query = plan.get('action_input', {}).get('query', question)
        rag_results = medical_rag_search(query, top_k=3)
    
    # Step 3: Generate final answer with RAG context
    if rag_results:
        rag_context = "\n".join([f"- {r['answer'][:500]}" for r in rag_results[:2]])
        
        final_prompt = f"""<s>[INST] Based on the medical knowledge:
{rag_context}

Answer this question: {question}

Provide a clear, concise medical answer. [/INST]"""
        
        inputs = tokenizer(final_prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.5,
                do_sample=True
            )
        
        final_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # Extract only the answer part
        if "[/INST]" in final_answer:
            final_answer = final_answer.split("[/INST]")[-1].strip()
    else:
        final_answer = "Could not retrieve relevant medical information."
    
    return {
        "plan": plan,
        "rag_results": rag_results,
        "final_answer": final_answer
    }

# ============================================
# Example: Theory question
# ============================================
result = inference_rag("Triệu chứng của viêm phổi là gì và cách điều trị?")

print("📚 RAG Result:")
print(f"\n🔹 Plan: {json.dumps(result['plan'], indent=2, ensure_ascii=False)}")
print(f"\n🔹 RAG Retrieved {len(result['rag_results'])} documents")
print(f"\n🔹 Final Answer:\n{result['final_answer']}")

# Expected output:
# 🔹 Plan: {
#     "thought": "Đây là câu hỏi lý thuyết về viêm phổi, cần tra cứu kiến thức y khoa",
#     "tool": "Knowledge",
#     "action": "Medical_RAG",
#     "action_input": {"query": "pneumonia symptoms treatment"}
# }
# 
# 🔹 RAG Retrieved 3 documents
# 
# 🔹 Final Answer:
# Viêm phổi có các triệu chứng chính bao gồm:
# - Sốt cao, ớn lạnh
# - Ho có đờm (có thể có máu)
# - Khó thở, thở nhanh
# - Đau ngực khi hít thở sâu
# ...
```

---

### 🔄 Combined Pipeline: Auto-Routing

```python
def trimedagent_inference(question: str, image_path: str = None) -> dict:
    """
    Complete TriMedAgent inference with auto-routing
    
    - If image provided → VQA pipeline (Vision tools)
    - If no image → RAG pipeline (Knowledge retrieval)
    """
    if image_path:
        print("📷 Image detected → Using VQA Pipeline")
        return {"type": "vqa", "result": inference_vqa(image_path, question)}
    else:
        print("📚 No image → Using RAG Pipeline")
        return {"type": "rag", "result": inference_rag(question)}

# ============================================
# Examples
# ============================================

# Case 1: With image
result1 = trimedagent_inference(
    question="Có dấu hiệu viêm phổi không?",
    image_path="chest_xray.jpg"
)

# Case 2: Without image (theory)
result2 = trimedagent_inference(
    question="Phân biệt viêm phổi do virus và vi khuẩn như thế nào?"
)
```

---

### 📋 JSON Output Format Reference

| Field | Type | Description | Example |
|-------|------|-------------|---------|
| `thought` | string | Agent's reasoning | "Cần detect tổn thương phổi" |
| `tool` | string | Tool category | "Vision" \| "Knowledge" |
| `action` | string | Specific tool | "GroundingDINO", "Medical_RAG" |
| `action_input` | object | Tool parameters | `{"prompt": "lung nodule"}` |

**Vision Tools:**
- `GroundingDINO`: Object detection → `{"prompt": "target object"}`
- `MedSAM`: Segmentation → `{"boxes": [[x1,y1,x2,y2]]}`
- `ZoomProcessor`: Smart crop → `{"target": "small object", "mode": "smart_crop"}`
- `BiomedCLIP`: Classification → `{"task": "classify"}`

**Knowledge Tools:**
- `Medical_RAG`: Knowledge retrieval → `{"query": "medical question"}`

---

## 🎓 Training Pipeline

### Stage 1: SFT (Supervised Fine-Tuning)

```bash
# Train LoRA adapter to output structured JSON
python scripts/train_sft.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --dataset "data/sft_dataset" \
    --output_dir "outputs/sft-v1" \
    --lora_r 64 \
    --lora_alpha 128 \
    --epochs 3
```

### Stage 2: RL (GRPO)

```bash
# Continue training with RL to optimize policy
python scripts/train_rl_grpo.py \
    --base_model "chaoyinshe/llava-med-v1.5-mistral-7b-hf" \
    --sft_adapter "outputs/sft-v1" \
    --output_dir "outputs/rl-v1" \
    --reward_weights "0.3,0.4,0.2,0.1"
```

### Upload to HuggingFace

```bash
# Push adapter to HF Hub
huggingface-cli upload ngnam1104/TriMedAgent outputs/rl-v1
```

📖 **Chi tiết training**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

---

## 📊 Reward Function (RL)

$$R_{total} = w_1 \cdot R_{IoU} + w_2 \cdot R_{Acc} + w_3 \cdot R_{Format} + R_{Step} + R_{RAG} + R_{Image}$$

| Component | Weight | Description |
|-----------|--------|-------------|
| $R_{IoU}$ | 0.3 | IoU với ground truth boxes (only if has_image) |
| $R_{Acc}$ | 0.4 | Độ chính xác câu trả lời |
| $R_{Format}$ | 0.2 | JSON 4-field format validity |
| $R_{Step}$ | -0.1/step | Penalty cho nhiều bước (>4 steps) |
| $R_{RAG}$ | +0.1~0.2 | **Bonus** khi dùng RAG đúng lúc |
| $R_{Image}$ | -0.15 | **Penalty** dùng vision tool khi không có ảnh |

### Image-Aware Reward Logic

```python
# Smart rewards based on image availability
if not has_valid_image:
    if action == "Medical_RAG":
        reward += 0.2   # 🎁 Bonus: Smart choice!
    elif action in ["GroundingDINO", "MedSAM", "Zoom"]:
        reward -= 0.15  # ⚠️ Penalty: Wasted action
        
if has_valid_image and difficulty == "hard":
    if action == "Medical_RAG":
        reward += 0.1   # 🎁 Bonus: Good augmentation
```

---

## 🔧 Tools

| Tool | Model | Purpose | When Used |
|------|-------|---------|-----------|
| `BiomedCLIP` | microsoft/BiomedCLIP | Image classification & context | **Always first** (if has image) |
| `GroundingDINO` | IDEA-Research | Object detection | Planner requests detection |
| `MedSAM` | SAM-ViT-B | Segmentation | After detection, for masks |
| `ZoomProcessor` | Custom | Smart crop for small objects | Small targets (nodule, fracture) |
| `Medical_RAG` | MedQuAD + FAISS | Knowledge retrieval | No image OR hard cases |
| `Gatekeeper` | LLaVA-Med | Verification | Final check |

### Tool JSON Format (4-field)

```json
{
    "thought": "Reasoning about what to do...",
    "tool": "Vision | Knowledge",
    "action": "GroundingDINO | Medical_RAG | MedSAM | ZoomProcessor",
    "action_input": {"prompt": "...", "query": "..."}
}
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA 11.8+
- GPU: T4 (16GB) hoặc cao hơn

### Key Dependencies

```
torch>=2.0
transformers>=4.36
peft>=0.7  # LoRA
trl>=0.7   # GRPO training
accelerate
bitsandbytes
```

---

## 📝 License

Apache 2.0 License

---

## 🙏 Acknowledgments

- [ReAct](https://arxiv.org/abs/2210.03629) - Reasoning + Acting
- [GRPO](https://arxiv.org/abs/2402.03300) - DeepSeek-R1
- [LLaVA-Med](https://github.com/microsoft/LLaVA-Med)
- [PubMedBERT](https://huggingface.co/microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract)
- [LoRA](https://arxiv.org/abs/2106.09685)
