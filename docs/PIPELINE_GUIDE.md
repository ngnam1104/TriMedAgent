# 🔬 Hướng Dẫn Chi Tiết Pipeline MMedAgent

> Tài liệu phân tích toàn bộ code và pipeline của MMedAgent - Multi-modal Medical Agent

---

## 📋 Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Pipeline Inference](#2-pipeline-inference)
3. [Pipeline Training](#3-pipeline-training)
4. [Pipeline Serving (Web UI)](#4-pipeline-serving-web-ui)
5. [Tool Workers Chi Tiết](#5-tool-workers-chi-tiết)
6. [Core Components](#6-core-components)
7. [Data Flow Diagrams](#7-data-flow-diagrams)
8. [File Reference](#8-file-reference)

---

## 1. Tổng Quan Kiến Trúc

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              MMedAgent System                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  │
│  │   User      │───▶│  Gradio UI  │───▶│ Controller  │───▶│   Workers   │  │
│  │   Input     │    │  :7860      │    │  :20001     │    │  :40000+    │  │
│  └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  │
│                                              │                              │
│                           ┌──────────────────┼──────────────────┐          │
│                           ▼                  ▼                  ▼          │
│                    ┌───────────┐      ┌───────────┐      ┌───────────┐    │
│                    │  Model    │      │   Tool    │      │   Tool    │    │
│                    │  Worker   │      │  Worker 1 │      │  Worker N │    │
│                    │(LLaVA-Med)│      │(MedSAM)   │      │(DINO,etc) │    │
│                    └───────────┘      └───────────┘      └───────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Các Thành Phần Chính

| Component | Mô Tả | Files Chính |
|-----------|-------|-------------|
| **LLaVA Core** | Model backbone xử lý vision-language | `llava/model/`, `llava/mm_utils.py` |
| **Controller** | Điều phối requests giữa client và workers | `llava/serve/controller.py` |
| **Model Worker** | Chạy LLaVA-Med model | `llava/serve/model_worker.py` |
| **Tool Workers** | Các công cụ y tế chuyên biệt | `serve/*.py` |
| **Training** | Huấn luyện với LoRA/DeepSpeed | `llava/train/` |
| **Evaluation** | Đánh giá model | `llava/eval/` |

---

## 2. Pipeline Inference

### 2.1 Flow Diagram

```
┌────────────┐     ┌────────────────┐     ┌──────────────────┐
│   Input    │────▶│  Load Model    │────▶│ Process Image    │
│ (Image +   │     │  (builder.py)  │     │ (mm_utils.py)    │
│  Question) │     └────────────────┘     └──────────────────┘
└────────────┘              │                      │
                            ▼                      ▼
                   ┌────────────────┐     ┌──────────────────┐
                   │ Tokenizer      │────▶│ Build Prompt     │
                   │ (conversation) │     │ (conv_templates) │
                   └────────────────┘     └──────────────────┘
                                                   │
                                                   ▼
                   ┌────────────────┐     ┌──────────────────┐
                   │   Output       │◀────│ Model Generate   │
                   │   Answer       │     │ (LLaVA forward)  │
                   └────────────────┘     └──────────────────┘
```

### 2.2 Chi Tiết Từng Bước

#### Bước 1: Load Model (`llava/model/builder.py`)

```python
# File: llava/model/builder.py
def load_pretrained_model(model_path, model_base, model_name, 
                          load_8bit=False, load_4bit=False, 
                          device_map="auto", device="cuda"):
    """
    Load model với các options:
    - LoRA weights: Nếu có 'lora' trong model_name
    - Base model: Load từ model_base
    - Quantization: 8-bit hoặc 4-bit
    
    Returns:
        tokenizer: Tokenizer cho text
        model: LlavaLlamaForCausalLM 
        image_processor: Processor cho images
        context_len: Max context length
    """
```

**Flow chi tiết:**
```
model_path ─────┐
                │
model_base ─────┼───▶ load_pretrained_model()
                │           │
model_name ─────┘           ├──▶ Detect model type (llava/mpt/llama)
                            ├──▶ Load tokenizer
                            ├──▶ Load base model
                            ├──▶ Load LoRA weights (if any)
                            ├──▶ Merge LoRA into base
                            ├──▶ Initialize vision tower
                            └──▶ Return (tokenizer, model, processor, ctx_len)
```

#### Bước 2: Process Image (`llava/mm_utils.py`)

```python
# File: llava/mm_utils.py

def process_images(images, image_processor, model_cfg):
    """
    Xử lý images trước khi đưa vào model.
    
    Args:
        images: List[PIL.Image] - Input images
        image_processor: Processor từ vision tower
        model_cfg: Config chứa image_aspect_ratio
    
    Flow:
        1. Nếu aspect_ratio == 'pad': expand2square() 
        2. Preprocess với image_processor
        3. Return tensor [B, C, H, W]
    """

def expand2square(pil_img, background_color):
    """Pad image thành hình vuông với background color."""
    
def tokenizer_image_token(prompt, tokenizer, image_token_index, return_tensors):
    """
    Tokenize prompt với special image token.
    
    Flow:
        1. Split prompt theo '<image>'
        2. Tokenize từng chunk
        3. Insert IMAGE_TOKEN_INDEX giữa các chunks
        4. Return input_ids tensor
    """
```

#### Bước 3: Build Conversation (`llava/conversation.py`)

```python
# File: llava/conversation.py

class SeparatorStyle(Enum):
    """Các kiểu separator cho conversation"""
    SINGLE = auto()      # Dùng 1 separator
    TWO = auto()         # Dùng 2 separators xen kẽ
    MPT = auto()         # Style cho MPT models
    PLAIN = auto()       # Không có role prefix
    LLAMA_2 = auto()     # Style cho LLaMA-2

@dataclass
class Conversation:
    """
    Quản lý conversation history.
    
    Attributes:
        system: System prompt
        roles: ['USER', 'ASSISTANT'] 
        messages: List conversation turns
        sep_style: Kiểu separator
        sep, sep2: Separator strings
    """
    
    def get_prompt(self):
        """
        Build full prompt string từ messages.
        
        Flow theo sep_style:
        - SINGLE: system + sep + role: msg + sep + ...
        - TWO: system + sep + role: msg + sep/sep2 + ...
        - LLAMA_2: [INST] <<SYS>> sys <</SYS>> msg [/INST] response
        """
    
    def append_message(self, role, message):
        """Thêm message vào conversation."""
```

#### Bước 4: Model Generate (`llava/eval/model_vqa.py`)

```python
# File: llava/eval/model_vqa.py

def eval_model(args):
    """
    Main inference function.
    
    Flow:
    1. Load model: load_pretrained_model()
    2. Load questions từ JSONL
    3. For each question:
       a. Load image nếu có
       b. Build prompt với conversation template
       c. Tokenize với image token
       d. model.generate() 
       e. Decode output
       f. Save to answers file
    """
    
    # Key generation params:
    output_ids = model.generate(
        input_ids,
        images=image_tensor,
        do_sample=True if temperature > 0 else False,
        temperature=args.temperature,
        top_p=args.top_p,
        num_beams=args.num_beams,
        max_new_tokens=1024,
        use_cache=True,
    )
```

### 2.3 Inference Entry Points

| File | Mô Tả | Sử Dụng |
|------|-------|---------|
| `llava/eval/model_vqa.py` | Batch inference từ JSONL | `python llava/eval/model_vqa.py --model-path ... --question-file ...` |
| `llava/eval/run_llava.py` | Single image inference | `python llava/eval/run_llava.py --model-path ... --image-file ... --query ...` |
| `llava/serve/cli.py` | Interactive CLI | `python -m llava.serve.cli --model-path ...` |

---

## 3. Pipeline Training

### 3.1 Training Flow Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Training Data  │────▶│  LazySupervisedDataset  │────▶│  DataCollator   │
│  (JSONL + imgs) │     │  (train.py)     │     │  (padding/batch)│
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  Model Init     │────▶│  LoRA Setup     │────▶│  DeepSpeed      │
│  (LlavaLlama)   │     │  (peft)         │     │  (ZeRO Stage 2) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                         │
         ┌───────────────────────────────────────────────┘
         ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│  LLaVATrainer   │────▶│  Training Loop  │────▶│  Save Checkpoint│
│  (custom)       │     │  (forward/back) │     │  (LoRA weights) │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### 3.2 Chi Tiết Training Components

#### 3.2.1 Data Arguments (`llava/train/train.py`)

```python
# File: llava/train/train.py

@dataclass
class ModelArguments:
    model_name_or_path: str          # Base model path
    version: str = "v0"               # Conversation version
    freeze_backbone: bool = False     # Freeze LLM weights
    tune_mm_mlp_adapter: bool = False # Only tune projector
    vision_tower: str = None          # Vision encoder (CLIP)
    mm_projector_type: str = 'linear' # Projector type
    
@dataclass  
class DataArguments:
    data_path: str                    # Path to training JSONL
    image_folder: str                 # Path to images
    image_aspect_ratio: str = 'square'# pad/square/resize
    lazy_preprocess: bool = False     # Lazy loading

@dataclass
class TrainingArguments:
    # LoRA configs
    lora_enable: bool = False
    lora_r: int = 64                  # LoRA rank
    lora_alpha: int = 16              # LoRA alpha
    lora_dropout: float = 0.05
    
    # Training configs
    bits: int = 16                    # 4/8/16 bit training
    mm_projector_lr: float = None     # Separate LR for projector
```

#### 3.2.2 Dataset Class

```python
# File: llava/train/train.py

class LazySupervisedDataset(Dataset):
    """
    Lazy loading dataset cho instruction tuning.
    
    Data format (JSONL):
    {
        "id": "unique_id",
        "image": "image_name.jpg",
        "conversations": [
            {"from": "human", "value": "<image>\nQuestion"},
            {"from": "gpt", "value": "Answer with thoughts/actions/value"}
        ]
    }
    """
    
    def __getitem__(self, i):
        """
        Flow:
        1. Load conversation từ JSON
        2. Load image nếu có
        3. Preprocess conversation (tokenize)
        4. Build input_ids và labels
        5. Return dict với keys: input_ids, labels, image
        """
```

#### 3.2.3 Tool Use Format (`llava/mm_utils.py`)

```python
# File: llava/mm_utils.py

def reorganize_source_for_tool_use(source):
    """
    Chuyển đổi conversation format cho tool use training.
    
    Input format:
    {
        "from": "gpt",
        "thoughts": "Thinking about the task...",
        "actions": [{"tool": "MedSAM", "params": {...}}],
        "value": "Final response"
    }
    
    Output format:
    {
        "from": "gpt", 
        "value": '"thoughts🤔" Thinking...\n"actions🚀" [...]\n"value👉" Final response'
    }
    """
```

#### 3.2.4 Custom Trainer (`llava/train/llava_trainer.py`)

```python
# File: llava/train/llava_trainer.py

class LLaVATrainer(Trainer):
    """
    Custom trainer với:
    1. Support cho DeepSpeed ZeRO
    2. Modality-aware batching
    3. MM adapter saving
    """
    
    def _save_checkpoint(self, model, trial, metrics=None):
        """
        Save checkpoint với:
        - LoRA weights (nếu enable)
        - MM projector weights
        - Non-LoRA trainables
        """
    
class LengthGroupedSampler(Sampler):
    """
    Sampler nhóm samples theo length.
    Giúp giảm padding và tăng efficiency.
    """
```

### 3.3 Training Command

```bash
# File: tuning.sh

deepspeed llava/train/train_mem.py \
    # LoRA settings
    --lora_enable True \
    --lora_r 128 \
    --lora_alpha 256 \
    --mm_projector_lr 2e-5 \
    
    # DeepSpeed
    --deepspeed ./scripts/zero2.json \
    
    # Model
    --model_name_or_path ./base_model \
    --version v1 \
    
    # Data
    --data_path ./train_data_json/example.jsonl \
    --image_folder ./train_images \
    
    # Vision
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    
    # Training
    --bf16 True \
    --num_train_epochs 30 \
    --per_device_train_batch_size 12 \
    --learning_rate 2e-4 \
    --output_dir ./checkpoints/output_lora_weights
```

### 3.4 Merge LoRA Weights

```python
# File: scripts/merge_lora_weights.py

def merge_lora(args):
    """
    Merge LoRA weights vào base model.
    
    Flow:
    1. load_pretrained_model() - Tự động merge LoRA
    2. model.save_pretrained() - Save merged model
    3. tokenizer.save_pretrained() - Save tokenizer
    """
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        args.model_path, args.model_base, model_name, device_map='cpu')
    
    model.save_pretrained(args.save_model_path)
    tokenizer.save_pretrained(args.save_model_path)
```

---

## 4. Pipeline Serving (Web UI)

### 4.1 Serving Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Client Browser                                 │
│                          http://localhost:7860                          │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────┐
│                        Gradio Web Server                                 │
│              (gradio_web_server_mmedagent.py)                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐    │
│  │  Image      │  │  Text       │  │  Sketch     │  │  Chat       │    │
│  │  Upload     │  │  Input      │  │  Mask       │  │  History    │    │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘    │
└────────────────────────────────┬────────────────────────────────────────┘
                                 │ HTTP Requests
┌────────────────────────────────▼────────────────────────────────────────┐
│                           Controller                                     │
│                    (llava/serve/controller.py)                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Worker Registry:                                                │   │
│  │  - worker_info: Dict[worker_name -> WorkerInfo]                 │   │
│  │  - dispatch_method: LOTTERY | SHORTEST_QUEUE                     │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Endpoints:                                                      │   │
│  │  - /register_worker     - /list_models                          │   │
│  │  - /get_worker_address  - /receive_heart_beat                   │   │
│  └─────────────────────────────────────────────────────────────────┘   │
└──────┬─────────────────────┬─────────────────────┬──────────────────────┘
       │                     │                     │
       ▼                     ▼                     ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│ Model Worker │     │ Tool Worker  │     │ Tool Worker  │
│ (LLaVA-Med)  │     │ (MedSAM)     │     │ (DINO)       │
│ :40000       │     │ :21002       │     │ :21001       │
└──────────────┘     └──────────────┘     └──────────────┘
```

### 4.2 Controller Chi Tiết (`llava/serve/controller.py`)

```python
# File: llava/serve/controller.py

class DispatchMethod(Enum):
    LOTTERY = auto()        # Random weighted by speed
    SHORTEST_QUEUE = auto() # Pick worker with shortest queue

@dataclass
class WorkerInfo:
    model_names: List[str]  # Models worker can serve
    speed: int              # Relative speed
    queue_length: int       # Current queue length
    check_heart_beat: bool  # Enable heartbeat check
    last_heart_beat: str    # Last heartbeat timestamp

class Controller:
    """
    Central controller quản lý workers.
    
    Methods:
    - register_worker(): Đăng ký worker mới
    - get_worker_address(): Lấy địa chỉ worker cho model
    - receive_heart_beat(): Nhận heartbeat từ workers
    - list_models(): Liệt kê các models available
    """
    
    def get_worker_address(self, model_name):
        """
        Chọn worker để xử lý request.
        
        LOTTERY: Random theo tỉ lệ speed
        SHORTEST_QUEUE: Chọn worker có queue ngắn nhất
        """
```

### 4.3 Model Worker Chi Tiết (`llava/serve/model_worker.py`)

```python
# File: llava/serve/model_worker.py

class ModelWorker:
    """
    Worker chạy LLaVA-Med model.
    
    Init:
    1. Load model với load_pretrained_model()
    2. Register với controller
    3. Start heartbeat thread
    """
    
    def generate_stream(self, params):
        """
        Generate response streaming.
        
        Params:
        - prompt: Input prompt
        - images: List base64 images
        - temperature, top_p, max_new_tokens
        
        Flow:
        1. Process images
        2. Tokenize prompt với image tokens
        3. model.generate() với TextIteratorStreamer
        4. Yield chunks qua stream
        """

# FastAPI endpoints
@app.post("/worker_generate_stream")
async def generate_stream(request: Request):
    """Stream generation endpoint."""
    
@app.post("/worker_get_status")
async def get_status(request: Request):
    """Return worker status."""
```

### 4.4 Gradio Web Server (`llava/serve/gradio_web_server_mmedagent.py`)

```python
# File: llava/serve/gradio_web_server_mmedagent.py

class ImageMask(gr.components.Image):
    """
    Custom Gradio component cho image + sketch mask.
    Cho phép user vẽ bounding box trực tiếp.
    """

def add_text(state, text, image_dict, ref_image_dict, ...):
    """
    Xử lý input từ user.
    
    Flow:
    1. Validate input
    2. Process image (resize, pad)
    3. Extract sketch mask → bounding box
    4. Append message to conversation state
    """

def http_bot(state, model_selector, temperature, ...):
    """
    Gửi request đến model và stream response.
    
    Flow:
    1. Get worker address từ controller
    2. Build request payload
    3. Stream response chunks
    4. Parse tool calls (thoughts/actions/value)
    5. Execute tools nếu cần
    6. Update UI với results
    """

def get_worker_addr(controller_addr, worker_name):
    """
    Lấy địa chỉ worker từ controller.
    """

# Tool output parsing
def parse_tool_output(text):
    """
    Parse output format: 
    "thoughts🤔" ... "actions🚀" [...] "value👉" ...
    """
```

---

## 5. Tool Workers Chi Tiết

### 5.1 Overview

| Worker | File | Port | Chức năng |
|--------|------|------|-----------|
| **Grounding DINO** | `serve/grounding_dino_worker.py` | 21001 | Object detection/grounding |
| **MedSAM** | `serve/MedSAM_worker.py` | 21002 | Medical image segmentation |
| **Grounded MedSAM** | `serve/grounded_medsam_worker.py` | 21003 | DINO + MedSAM combined |
| **BiomedCLIP** | `serve/biomedclip_worker.py` | 21004 | Medical image classification |
| **ChatCAD-G** | `serve/chatcad_G_worker.py` | 21005 | Report generation |
| **ChatCAD-R** | `serve/chatcad_R_worker.py` | 21006 | RAG với medical knowledge |

### 5.2 Grounding DINO Worker

```python
# File: serve/grounding_dino_worker.py

class ModelWorker:
    """
    Grounding DINO cho medical image detection.
    
    Model: groundingdinomed-checkpoint0005_slim.pth
    """
    
    def __init__(...):
        # Load GroundingDINO model
        self.model = load_model(
            model_config_path=model_config,
            model_checkpoint_path=model_path,
            device=device,
        )
        
        # Image transform
        self.transform = T.Compose([
            T.RandomResize([800], max_size=1333),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
    
    def generate(self, params):
        """
        Detect objects trong medical image.
        
        Input:
        - image: base64 encoded image
        - text_prompt: Text description (e.g., "tumor", "lesion")
        - box_threshold, text_threshold
        
        Output:
        - boxes: [[x1,y1,x2,y2], ...]
        - logits: [confidence, ...]
        - phrases: ["tumor", ...]
        """
```

### 5.3 MedSAM Worker

```python
# File: serve/MedSAM_worker.py

class ModelWorker:
    """
    MedSAM cho medical image segmentation.
    
    Model: medsam_vit_b.pth
    """
    
    def __init__(...):
        # Load SAM model
        self.sam = build_sam(checkpoint=sam_path)
        self.sam_predictor = SamPredictor(self.sam)
    
    def generate(self, params):
        """
        Segment medical image với bbox prompt.
        
        Input:
        - image: base64 encoded image
        - boxes: [[x1,y1,x2,y2]] - Bounding box prompts
        
        Output:
        - masks_rle: RLE encoded masks
        """
        
        # Set image
        self.sam_predictor.set_image(image_np)
        
        # Predict với box prompt
        masks, scores, logits = self.sam_predictor.predict(
            box=input_box,
            multimask_output=True,
        )
```

### 5.4 BiomedCLIP Worker

```python
# File: serve/biomedclip_worker.py

class ModelWorker:
    """
    BiomedCLIP cho medical image classification.
    
    Model: microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224
    """
    
    def __init__(...):
        self.model, self.preprocess = create_model_from_pretrained(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        self.tokenizer = get_tokenizer(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
    
    def generate(self, params):
        """
        Classify medical image.
        
        Input:
        - image: base64 encoded image
        - labels: ["pneumonia", "normal", "tumor"]
        
        Output:
        - predictions: [(label, probability), ...]
        """
        
        # Encode image và text
        image_features = self.model.encode_image(image_input)
        text_features = self.model.encode_text(text_input)
        
        # Compute similarity
        logits = image_features @ text_features.T
```

### 5.5 ChatCAD-R Worker (RAG)

```python
# File: serve/chatcad_R_worker.py
# File: src/ChatCAD_R/chat_bot_RAG.py

class gpt_bot:
    """
    ChatCAD với RAG từ Merck Manual.
    
    Components:
    - JFchexpert model: Chest X-ray diagnosis
    - R2G model: Report generation
    - Sentence embedding: Query matching
    - GPT: Answer refinement
    """
    
    def __init__(...):
        # Load models
        self.img_model, self.imgcfg = JFinit(config_path, weights_path)
        self.reporter = reportGen()
        self.sent_model = SentenceModel()
        self.msd_dict = json.load(open(msd_path))  # Merck Manual index
    
    def chat(self, message, ref_record, force_generate=False):
        """
        RAG-based medical chat.
        
        Flow:
        1. Refine query với GPT
        2. Search Merck Manual với sentence embedding
        3. Retrieve relevant knowledge
        4. Generate response với GPT + knowledge
        """
        
    def report_cxr_zh(self, img):
        """
        Generate chest X-ray report.
        
        Flow:
        1. JFinfer: Disease probability
        2. reportGen: Generate report
        3. GPT: Refine report
        """

def RAG(prompt, api_key, chatbot=None):
    """
    Main RAG function.
    
    Flow:
    1. Initialize chatbot if None
    2. Process prompt
    3. Query knowledge base
    4. Generate response
    """
```

---

## 6. Core Components

### 6.1 Model Architecture (`llava/model/llava_arch.py`)

```python
# File: llava/model/llava_arch.py

class LlavaMetaModel:
    """
    Base class cho LLaVA model.
    
    Components:
    - vision_tower: CLIP ViT encoder
    - mm_projector: MLP projecting vision → language space
    """
    
    def initialize_vision_modules(self, model_args):
        """
        Initialize vision components.
        
        1. Build vision tower (CLIP)
        2. Build MM projector (MLP)
        3. Load pretrained weights if provided
        """
        
    def get_vision_tower(self):
        """Return vision encoder."""

class LlavaMetaForCausalLM:
    """
    Mixin class cho causal LM với multimodal.
    """
    
    def encode_images(self, images):
        """
        Encode images qua vision tower + projector.
        
        Flow:
        images → vision_tower → mm_projector → image_features
        """
        image_features = self.get_model().get_vision_tower()(images)
        image_features = self.get_model().mm_projector(image_features)
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, position_ids, attention_mask, 
        past_key_values, labels, images
    ):
        """
        Chuẩn bị inputs cho multimodal forward.
        
        Flow:
        1. Encode images
        2. Tìm vị trí IMAGE_TOKEN_INDEX trong input_ids
        3. Replace với image embeddings
        4. Adjust labels (ignore image positions)
        """
```

### 6.2 Constants (`llava/constants.py`)

```python
# File: llava/constants.py

# Server constants
CONTROLLER_HEART_BEAT_EXPIRATION = 30
WORKER_HEART_BEAT_INTERVAL = 15

# Model constants
IGNORE_INDEX = -100           # Label ignore index
IMAGE_TOKEN_INDEX = -200      # Special token for image
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"
IMAGE_PLACEHOLDER = "<image-placeholder>"
```

### 6.3 Utilities (`llava/utils.py`)

```python
# File: llava/utils.py

def build_logger(logger_name, logger_filename):
    """Build logger với file và console handlers."""

def disable_torch_init():
    """
    Disable torch default init.
    Tăng tốc model loading.
    """

def violates_moderation(text):
    """Check content moderation với OpenAI API."""

def server_error_msg():
    """Default server error message."""
```

---

## 7. Data Flow Diagrams

### 7.1 Inference Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        INFERENCE DATA FLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌─────────┐    ┌──────────────┐    ┌──────────────┐    ┌─────────┐    │
│  │ Image   │───▶│ CLIP ViT     │───▶│ MM Projector │───▶│ Image   │    │
│  │ (PIL)   │    │ Encoder      │    │ (MLP)        │    │ Embeds  │    │
│  └─────────┘    └──────────────┘    └──────────────┘    └────┬────┘    │
│                                                               │         │
│  ┌─────────┐    ┌──────────────┐                              │         │
│  │ Text    │───▶│ Tokenizer    │                              │         │
│  │ Prompt  │    │              │                              │         │
│  └─────────┘    └──────┬───────┘                              │         │
│                        │                                       │         │
│                        ▼                                       ▼         │
│                ┌──────────────────────────────────────────────────┐     │
│                │     Combine: Replace <image> với Image Embeds    │     │
│                │     [text_embeds, image_embeds, text_embeds]     │     │
│                └──────────────────────┬───────────────────────────┘     │
│                                       │                                  │
│                                       ▼                                  │
│                              ┌──────────────┐                           │
│                              │  LLaMA LLM   │                           │
│                              │  Decoder     │                           │
│                              └──────┬───────┘                           │
│                                     │                                    │
│                                     ▼                                    │
│                              ┌──────────────┐                           │
│                              │   Output     │                           │
│                              │   Tokens     │                           │
│                              └──────────────┘                           │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.2 Training Data Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        TRAINING DATA FLOW                                 │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  ┌──────────────┐                                                        │
│  │ JSONL File   │                                                        │
│  │ {            │                                                        │
│  │  "id": ...,  │                                                        │
│  │  "image": ...│                                                        │
│  │  "convs": ...│                                                        │
│  │ }            │                                                        │
│  └──────┬───────┘                                                        │
│         │                                                                 │
│         ▼                                                                 │
│  ┌──────────────────┐    ┌──────────────────┐                           │
│  │ Lazy Dataset     │───▶│ Data Collator    │                           │
│  │ - Load on demand │    │ - Padding        │                           │
│  │ - Preprocess     │    │ - Batching       │                           │
│  └──────────────────┘    └────────┬─────────┘                           │
│                                   │                                      │
│                                   ▼                                      │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                         Training Batch                          │     │
│  │  input_ids: [B, seq_len]   - Tokenized input                   │     │
│  │  labels: [B, seq_len]      - Target tokens (IGNORE human turn) │     │
│  │  images: [B, C, H, W]      - Processed images                  │     │
│  │  attention_mask: [B, seq_len]                                  │     │
│  └────────────────────────────┬───────────────────────────────────┘     │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    Forward Pass                                 │     │
│  │  1. encode_images(images) → image_features                     │     │
│  │  2. prepare_inputs_labels_for_multimodal()                     │     │
│  │  3. LLM forward → logits                                       │     │
│  │  4. CrossEntropyLoss(logits, labels)                           │     │
│  └────────────────────────────┬───────────────────────────────────┘     │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────────────┐     │
│  │                    Backward + Optimize                          │     │
│  │  - DeepSpeed ZeRO Stage 2                                      │     │
│  │  - LoRA gradients only (if enabled)                            │     │
│  │  - Gradient checkpointing                                       │     │
│  └────────────────────────────────────────────────────────────────┘     │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

### 7.3 Web UI Request Flow

```
┌──────────────────────────────────────────────────────────────────────────┐
│                        WEB UI REQUEST FLOW                                │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  User Action: Upload image + Type question + Click Submit                │
│                              │                                           │
│                              ▼                                           │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ 1. add_text() - Gradio callback                        │             │
│  │    - Process image (resize, pad)                       │             │
│  │    - Extract sketch mask → bbox                        │             │
│  │    - Update conversation state                         │             │
│  └────────────────────────────┬───────────────────────────┘             │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ 2. http_bot() - Main chat handler                      │             │
│  │    a. Get model worker address from controller         │             │
│  │    b. Build request: {prompt, images, temperature}     │             │
│  │    c. POST to /worker_generate_stream                  │             │
│  └────────────────────────────┬───────────────────────────┘             │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ 3. Model Worker processing                             │             │
│  │    - Decode base64 images                              │             │
│  │    - Process với image_processor                       │             │
│  │    - model.generate() với streaming                    │             │
│  └────────────────────────────┬───────────────────────────┘             │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ 4. Parse response & Execute tools                      │             │
│  │    Response format:                                    │             │
│  │    "thoughts🤔" Analysis of the image...               │             │
│  │    "actions🚀" [{"tool": "MedSAM", "params": {...}}]   │             │
│  │    "value👉" Final response to user                    │             │
│  │                                                        │             │
│  │    If actions detected:                                │             │
│  │    → Call tool worker (MedSAM/DINO/etc.)              │             │
│  │    → Get tool output (masks, boxes, etc.)             │             │
│  │    → Render results on image                          │             │
│  └────────────────────────────┬───────────────────────────┘             │
│                               │                                          │
│                               ▼                                          │
│  ┌────────────────────────────────────────────────────────┐             │
│  │ 5. Update UI                                           │             │
│  │    - Show text response                                │             │
│  │    - Display annotated image (boxes, masks)            │             │
│  │    - Update chat history                               │             │
│  └────────────────────────────────────────────────────────┘             │
│                                                                          │
└──────────────────────────────────────────────────────────────────────────┘
```

---

## 8. File Reference

### 8.1 Core Files Cần Đọc (Theo Thứ Tự)

| # | File | Mục đích | Thời gian |
|---|------|----------|-----------|
| 1 | `llava/constants.py` | Hiểu các constants | 5 phút |
| 2 | `llava/conversation.py` | Hiểu conversation format | 20 phút |
| 3 | `llava/mm_utils.py` | Hiểu image/text processing | 30 phút |
| 4 | `llava/model/builder.py` | Hiểu model loading | 30 phút |
| 5 | `llava/model/llava_arch.py` | Hiểu model architecture | 45 phút |
| 6 | `llava/eval/model_vqa.py` | Hiểu inference pipeline | 30 phút |
| 7 | `llava/train/train.py` | Hiểu training pipeline | 1 giờ |
| 8 | `llava/serve/controller.py` | Hiểu serving architecture | 30 phút |
| 9 | `llava/serve/model_worker.py` | Hiểu model serving | 30 phút |
| 10 | `serve/*_worker.py` | Hiểu tool workers | 1 giờ |

### 8.2 Config Files

| File | Mô Tả |
|------|-------|
| `pyproject.toml` | Dependencies |
| `scripts/zero2.json` | DeepSpeed ZeRO Stage 2 config |
| `scripts/zero3.json` | DeepSpeed ZeRO Stage 3 config |

### 8.3 Scripts

| Script | Mô Tả |
|--------|-------|
| `tuning.sh` | Training với LoRA |
| `merge.sh` | Merge LoRA weights |
| `eval.sh` | Run evaluation |
| `eval_gpt4.sh` | GPT-4 evaluation |

---

## 📝 Summary

### Pipeline Chính:

1. **Inference**: `Image + Text` → `mm_utils` → `conversation` → `model.generate()` → `Output`

2. **Training**: `JSONL + Images` → `Dataset` → `Trainer` → `LoRA weights`

3. **Serving**: `Gradio UI` → `Controller` → `Model Worker` + `Tool Workers` → `Response`

### Key Concepts:

- **IMAGE_TOKEN_INDEX (-200)**: Special token đánh dấu vị trí image trong prompt
- **LoRA**: Low-Rank Adaptation cho efficient fine-tuning
- **DeepSpeed ZeRO**: Distributed training optimization
- **Tool Use Format**: `thoughts🤔` + `actions🚀` + `value👉`

---

*Tài liệu được tạo ngày 03/12/2024*
