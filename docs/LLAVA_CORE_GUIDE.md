# 🧠 Hướng Dẫn Làm Chủ LLaVA Core Architecture

> **LLaVA** (Large Language and Vision Assistant) - Kiến trúc multimodal kết hợp Vision Encoder và Large Language Model để xử lý cả ảnh và văn bản.

---

## 📋 Mục Lục

1. [Tổng Quan Kiến Trúc](#1-tổng-quan-kiến-trúc)
2. [Components Chính](#2-components-chính)
3. [Data Flow & Processing](#3-data-flow--processing)
4. [Code Analysis Chi Tiết](#4-code-analysis-chi-tiết)
5. [Customize & Extend](#5-customize--extend)
6. [Training Pipeline](#6-training-pipeline)
7. [Inference Pipeline](#7-inference-pipeline)
8. [Best Practices](#8-best-practices)

---

## 1. Tổng Quan Kiến Trúc

### 1.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                       LLaVA Architecture                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Input Image                    Input Text                     │
│      │                              │                           │
│      ▼                              │                           │
│  ┌─────────────────┐                │                           │
│  │ Vision Encoder  │                │                           │
│  │ (CLIP ViT-L)    │                │                           │
│  │ 336×336         │                │                           │
│  └────────┬────────┘                │                           │
│           │                         │                           │
│           │ Vision Features         │                           │
│           │ [N, 1024]               │                           │
│           ▼                         │                           │
│  ┌─────────────────┐                │                           │
│  │  MM Projector   │                │                           │
│  │  (MLP 2-layer)  │                │                           │
│  │  1024 → 4096    │                │                           │
│  └────────┬────────┘                │                           │
│           │                         │                           │
│           │ Projected Features      │                           │
│           │ [N, 4096]               │                           │
│           │                         │                           │
│           └─────────┬───────────────┘                           │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │   Token Embedder    │                                │
│          │  Text → [M, 4096]   │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│          Merge Image + Text Embeddings                          │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │    LLaMA Model      │                                │
│          │    (7B params)      │                                │
│          │  Transformer Layers │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│                     ▼                                           │
│          ┌─────────────────────┐                                │
│          │   LM Head           │                                │
│          │   4096 → 32K vocab  │                                │
│          └──────────┬──────────┘                                │
│                     │                                           │
│                     ▼                                           │
│              Generated Text                                     │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | Role | Input | Output |
|-----------|------|-------|--------|
| **Vision Encoder** | Extract image features | Image (3×336×336) | Visual tokens [N, 1024] |
| **MM Projector** | Align vision-language | Vision features [N, 1024] | LLM embeds [N, 4096] |
| **Token Embedder** | Text tokenization | Text tokens | Text embeds [M, 4096] |
| **LLaMA Model** | Language understanding | Mixed embeds [N+M, 4096] | Hidden states [N+M, 4096] |
| **LM Head** | Token prediction | Hidden states | Logits [N+M, vocab_size] |

### 1.3 Model Sizes

```python
# LLaVA-7B (base)
Vision Encoder:    ~300M params (CLIP ViT-L/14)
MM Projector:      ~4M params (1024→4096, 2 layers)
LLaMA Backbone:    ~7B params
Total:             ~7.3B params

# Memory footprint (FP16)
Vision Encoder:    ~600MB
MM Projector:      ~8MB  
LLaMA:             ~14GB
Total:             ~14.6GB VRAM
```

---

## 2. Components Chính

### 2.1 Vision Encoder (CLIP ViT-L)

**File:** `llava/model/multimodal_encoder/clip_encoder.py`

```python
# Vision Encoder Architecture
class CLIPVisionTower:
    def __init__(self, vision_tower_name):
        # Load CLIP ViT-L/14@336px
        self.vision_tower = CLIPVisionModel.from_pretrained(
            "openai/clip-vit-large-patch14-336"
        )
        
    def forward(self, images):
        # Input: [B, 3, 336, 336]
        # Output: [B, 576, 1024]  # 576 = (336/14)^2
        
        image_features = self.vision_tower(
            images, 
            output_hidden_states=True
        )
        # Select layer -2 (second to last)
        image_features = image_features.hidden_states[-2]
        return image_features
```

**Đặc điểm:**
- Input resolution: 336×336 pixels
- Patch size: 14×14
- Number of patches: 24×24 = 576 tokens
- Hidden dimension: 1024
- Freeze weights: ✅ (không train)

### 2.2 Multimodal Projector

**File:** `llava/model/multimodal_projector/builder.py`

```python
# MM Projector: Align vision features to LLM space
class MLPProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 2-layer MLP with GELU activation
        self.linear_1 = nn.Linear(
            config.mm_hidden_size,      # 1024 (CLIP)
            config.hidden_size,         # 4096 (LLaMA)
            bias=True
        )
        self.gelu = nn.GELU()
        self.linear_2 = nn.Linear(
            config.hidden_size,         # 4096
            config.hidden_size,         # 4096
            bias=True
        )
    
    def forward(self, x):
        # x: [B, 576, 1024]
        x = self.linear_1(x)      # → [B, 576, 4096]
        x = self.gelu(x)
        x = self.linear_2(x)      # → [B, 576, 4096]
        return x
```

**Vai trò:**
- Chuyển đổi vision features (1024-dim) sang LLM embedding space (4096-dim)
- Trainable: ✅ (được train trong cả 2 giai đoạn)
- Architecture options:
  - `linear`: Single linear layer
  - `mlp2x_gelu`: 2-layer MLP (default)

### 2.3 LLaVA Meta Architecture

**File:** `llava/model/llava_arch.py`

#### 2.3.1 LlavaMetaModel

```python
class LlavaMetaModel:
    """Base model with vision components"""
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize vision components
        if hasattr(config, "mm_vision_tower"):
            self.vision_tower = build_vision_tower(config)
            self.mm_projector = build_vision_projector(config)
    
    def get_vision_tower(self):
        """Get vision encoder"""
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower
    
    def initialize_vision_modules(self, model_args, fsdp=None):
        """Initialize or load vision modules"""
        # Load vision tower
        vision_tower = build_vision_tower(model_args)
        self.vision_tower = vision_tower
        
        # Build projector
        self.mm_projector = build_vision_projector(self.config)
        
        # Load pretrained projector weights if available
        if model_args.pretrain_mm_mlp_adapter:
            mm_projector_weights = torch.load(
                model_args.pretrain_mm_mlp_adapter
            )
            self.mm_projector.load_state_dict(mm_projector_weights)
```

#### 2.3.2 LlavaMetaForCausalLM

```python
class LlavaMetaForCausalLM(ABC):
    """Causal LM with multimodal capabilities"""
    
    def encode_images(self, images):
        """Encode images to LLM embedding space"""
        # Step 1: Vision encoder
        image_features = self.get_vision_tower()(images)
        # [B, 576, 1024]
        
        # Step 2: MM projector
        image_features = self.mm_projector(image_features)
        # [B, 576, 4096]
        
        return image_features
    
    def prepare_inputs_labels_for_multimodal(
        self, input_ids, attention_mask, labels, images
    ):
        """Merge image and text embeddings"""
        
        # Encode images
        if images is not None:
            image_features = self.encode_images(images)
        
        # Find <image> token positions
        # IMAGE_TOKEN_INDEX = -200
        image_token_indices = torch.where(
            input_ids == IMAGE_TOKEN_INDEX
        )
        
        # Replace <image> tokens with actual image features
        new_input_embeds = []
        for batch_idx in range(input_ids.shape[0]):
            # Get text embeddings
            cur_input_embeds = self.embed_tokens(input_ids[batch_idx])
            
            # Find image token positions
            image_positions = image_token_indices[batch_idx]
            
            # Split at image positions
            text_parts = split_by_image_tokens(cur_input_embeds)
            
            # Insert image features
            merged = []
            for i, text_part in enumerate(text_parts):
                merged.append(text_part)
                if i < len(image_features):
                    merged.append(image_features[i])
            
            new_input_embeds.append(torch.cat(merged))
        
        # Pad to max length
        new_input_embeds = pad_sequence(new_input_embeds)
        
        return new_input_embeds, attention_mask, labels
```

### 2.4 LLaVA LLaMA Model

**File:** `llava/model/language_model/llava_llama.py`

```python
class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    """LLaVA with LLaMA backbone"""
    
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        
        # Initialize LLaMA model
        self.model = LlavaLlamaModel(config)
        self.lm_head = nn.Linear(
            config.hidden_size,    # 4096
            config.vocab_size,     # 32000
            bias=False
        )
    
    def forward(
        self,
        input_ids,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        images=None,
        **kwargs
    ):
        # Prepare multimodal inputs
        (
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            inputs_embeds,
            labels
        ) = self.prepare_inputs_labels_for_multimodal(
            input_ids,
            position_ids,
            attention_mask,
            past_key_values,
            labels,
            images
        )
        
        # Forward through LLaMA
        outputs = super().forward(
            input_ids=None,           # Don't use input_ids
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,  # Use merged embeds
            labels=labels,
            **kwargs
        )
        
        return outputs
```

---

## 3. Data Flow & Processing

### 3.1 Training Data Flow

```
┌────────────────────────────────────────────────────────────────┐
│                      Training Pipeline                         │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  1. Data Loading                                               │
│     ├─ Image: "medical_image.jpg"                             │
│     └─ Conversation: [                                         │
│         {"from": "human", "value": "<image>\nQuestion?"},      │
│         {"from": "gpt", "value": "Answer..."}                  │
│       ]                                                        │
│                                                                │
│  2. Preprocessing                                              │
│     ├─ Image → Resize to 336×336                              │
│     │         → Normalize with CLIP stats                     │
│     │         → Tensor [3, 336, 336]                          │
│     │                                                          │
│     └─ Text  → Tokenize with special tokens                   │
│               → "<image>" → IMAGE_TOKEN_INDEX (-200)          │
│               → Convert to input_ids                          │
│                                                                │
│  3. Forward Pass                                               │
│     ├─ Vision Encoder: Image → [576, 1024]                    │
│     ├─ MM Projector:   [576, 1024] → [576, 4096]              │
│     ├─ Token Embed:    text_ids → [M, 4096]                   │
│     ├─ Merge:          Insert image at <image> position       │
│     └─ LLaMA Forward:  Mixed embeds → Logits                  │
│                                                                │
│  4. Loss Calculation                                           │
│     └─ CrossEntropy on answer tokens only                     │
│        (image tokens & question masked with IGNORE_INDEX)     │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 Inference Data Flow

```python
# Inference Example
from llava.model.builder import load_pretrained_model
from llava.mm_utils import process_images, tokenizer_image_token
from llava.conversation import conv_templates

# 1. Load model
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path="./llava_med_agent",
    model_base=None,
    model_name="llava-med"
)

# 2. Prepare image
from PIL import Image
image = Image.open("medical_xray.jpg")
images_tensor = process_images([image], image_processor, model.config)
images_tensor = images_tensor.to(model.device, dtype=torch.float16)

# 3. Prepare text
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], "<image>\nWhat's in this X-ray?")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# 4. Tokenize
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors='pt'
).unsqueeze(0).to(model.device)

# 5. Generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        do_sample=False,
        max_new_tokens=512,
        use_cache=True
    )

# 6. Decode
outputs = tokenizer.batch_decode(
    output_ids, 
    skip_special_tokens=True
)[0]
```

### 3.3 Token Flow Diagram

```
Text:  "USER: <image> What's in the image? ASSISTANT:"
                 ↓
Tokenization:  [1, 3148, 29901, -200, 1724, 29915, 29879, ...]
                      ↓
Embedding Lookup:
    1      →  [4096-dim] (BOS)
    3148   →  [4096-dim] (USER)
    29901  →  [4096-dim] (:)
    -200   →  [576, 4096-dim] (IMAGE FEATURES!)
    1724   →  [4096-dim] (What)
    ...
                      ↓
Concatenated: [1 + 3 + 576 + M, 4096]
                      ↓
                  LLaMA Forward
                      ↓
                 Generate tokens
```

---

## 4. Code Analysis Chi Tiết

### 4.1 Model Builder

**File:** `llava/model/builder.py`

```python
def load_pretrained_model(
    model_path,
    model_base=None,
    model_name=None,
    load_8bit=False,
    load_4bit=False,
    device_map="auto"
):
    """
    Load LLaVA pretrained model
    
    Args:
        model_path: Path to LLaVA weights
        model_base: Path to base LLaMA model (for LoRA)
        load_8bit: Use 8-bit quantization
        load_4bit: Use 4-bit quantization
    """
    
    # Load tokenizer
    if model_base:
        tokenizer = AutoTokenizer.from_pretrained(model_base)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    if 'lora' in model_name.lower() and model_base:
        # LoRA model: Load base + LoRA weights
        print('Loading LLaVA from base model...')
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_base,
            load_in_8bit=load_8bit,
            device_map=device_map
        )
        
        # Load non-LoRA components (vision tower, projector)
        non_lora_trainables = torch.load(
            os.path.join(model_path, 'non_lora_trainables.bin')
        )
        model.load_state_dict(non_lora_trainables, strict=False)
        
        # Load LoRA weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, model_path)
        model = model.merge_and_unload()
        
    else:
        # Full model: Load directly
        model = LlavaLlamaForCausalLM.from_pretrained(
            model_path,
            load_in_8bit=load_8bit,
            device_map=device_map
        )
    
    # Load vision tower
    model.get_vision_tower().load_model()
    
    # Load image processor
    image_processor = CLIPImageProcessor.from_pretrained(
        model.config.mm_vision_tower
    )
    
    return tokenizer, model, image_processor, context_len
```

### 4.2 Image Processing

**File:** `llava/mm_utils.py`

```python
def process_images(images, image_processor, model_cfg):
    """
    Process images for LLaVA input
    
    Args:
        images: List of PIL images
        image_processor: CLIP image processor
        model_cfg: Model config
    
    Returns:
        Tensor of shape [B, 3, 336, 336]
    """
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    
    new_images = []
    if image_aspect_ratio == 'pad':
        # Pad to square
        for image in images:
            # Expand to square with mean color
            image = expand2square(
                image,
                tuple(int(x*255) for x in image_processor.image_mean)
            )
            # CLIP preprocessing
            image = image_processor.preprocess(
                image,
                return_tensors='pt'
            )['pixel_values'][0]
            new_images.append(image)
    else:
        # Direct resize
        return image_processor(
            images,
            return_tensors='pt'
        )['pixel_values']
    
    # Stack if same shape
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    
    return new_images


def expand2square(pil_img, background_color):
    """Expand image to square by padding"""
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
```

### 4.3 Tokenization

```python
def tokenizer_image_token(
    prompt,
    tokenizer,
    image_token_index=IMAGE_TOKEN_INDEX,
    return_tensors=None
):
    """
    Tokenize text with special handling for <image> token
    
    Args:
        prompt: Text with "<image>" placeholder
        tokenizer: LLaMA tokenizer
        image_token_index: Special token ID (-200)
    
    Returns:
        List of token IDs with image_token_index at <image> position
    """
    
    # Split by <image>
    prompt_chunks = [
        tokenizer(chunk).input_ids 
        for chunk in prompt.split('<image>')
    ]
    
    # Insert image token between chunks
    def insert_separator(X, sep):
        return [
            ele for sublist in zip(X, [sep]*len(X)) 
            for ele in sublist
        ][:-1]
    
    input_ids = []
    offset = 0
    
    # Handle BOS token
    if (len(prompt_chunks) > 0 and 
        len(prompt_chunks[0]) > 0 and 
        prompt_chunks[0][0] == tokenizer.bos_token_id):
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    
    # Merge chunks with image token
    for x in insert_separator(prompt_chunks, [image_token_index]):
        input_ids.extend(x[offset:])
    
    if return_tensors == 'pt':
        return torch.tensor(input_ids, dtype=torch.long)
    
    return input_ids
```

---

## 5. Customize & Extend

### 5.1 Thay Đổi Vision Encoder

```python
# Option 1: Thay đổi CLIP variant
# File: config.json
{
    "mm_vision_tower": "openai/clip-vit-large-patch14",  # 224px
    # hoặc
    "mm_vision_tower": "openai/clip-vit-large-patch14-336",  # 336px (default)
}

# Option 2: Dùng encoder khác (ví dụ: DINOv2)
class DinoVisionTower(nn.Module):
    def __init__(self, vision_tower_name):
        self.vision_model = torch.hub.load(
            'facebookresearch/dinov2',
            'dinov2_vitl14'
        )
        self.hidden_size = 1024
    
    def forward(self, images):
        # DINOv2 forward
        features = self.vision_model.forward_features(images)
        return features['x_norm_patchtokens']

# Đăng ký trong builder
def build_vision_tower(config):
    vision_tower = config.mm_vision_tower
    if 'dinov2' in vision_tower:
        return DinoVisionTower(vision_tower)
    elif 'clip' in vision_tower:
        return CLIPVisionTower(vision_tower)
```

### 5.2 Thay Đổi MM Projector

```python
# Option 1: Deeper MLP
class DeepMLPProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config.mm_hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
            nn.GELU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
    
    def forward(self, x):
        return self.layers(x)

# Option 2: Q-Former (như BLIP-2)
class QFormerProjector(nn.Module):
    def __init__(self, config):
        super().__init__()
        from transformers import Blip2QFormerModel
        self.qformer = Blip2QFormerModel.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.num_query_tokens = 32
        self.query_tokens = nn.Parameter(
            torch.zeros(1, self.num_query_tokens, config.hidden_size)
        )
    
    def forward(self, image_features):
        # Use Q-Former to compress vision features
        query_output = self.qformer(
            query_embeds=self.query_tokens,
            encoder_hidden_states=image_features
        )
        return query_output.last_hidden_state

# Đăng ký trong config
{
    "mm_projector_type": "qformer",  # or "mlp2x_gelu", "deep_mlp"
}
```

### 5.3 Thêm Image Token Học Được

```python
# File: llava/model/llava_arch.py

def initialize_vision_tokenizer(self, model_args, tokenizer):
    """Add learnable <im_start> and <im_end> tokens"""
    
    if model_args.mm_use_im_start_end:
        # Add special tokens
        num_new_tokens = tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN],
            special_tokens=True
        )
        self.resize_token_embeddings(len(tokenizer))
        
        # Initialize với average của existing embeddings
        if num_new_tokens > 0:
            input_embeddings = self.get_input_embeddings().weight.data
            output_embeddings = self.get_output_embeddings().weight.data
            
            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            
            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
        # Make them trainable
        if model_args.tune_mm_mlp_adapter:
            for p in self.get_input_embeddings().parameters():
                p.requires_grad = True

# Usage in conversation
prompt = f"{DEFAULT_IM_START_TOKEN}<image>{DEFAULT_IM_END_TOKEN}\n{question}"
```

### 5.4 Multi-Image Support

```python
# Hiện tại LLaVA đã hỗ trợ multi-image
# File: llava/model/llava_arch.py

def prepare_inputs_labels_for_multimodal(self, ..., images):
    # Handle multiple images
    if type(images) is list or images.ndim == 5:
        # images: List of [1, 3, 336, 336] or [B, N, 3, 336, 336]
        concat_images = torch.cat([image for image in images], dim=0)
        image_features = self.encode_images(concat_images)
        
        # Split back per image
        split_sizes = [image.shape[0] for image in images]
        image_features = torch.split(image_features, split_sizes, dim=0)
        
        # Flatten for insertion
        image_features = [
            x.flatten(0, 1).to(self.device) 
            for x in image_features
        ]

# Usage
prompt = "<image>\nFirst image. <image>\nSecond image. What's different?"
images = [image1, image2]  # List of PIL images
```

---

## 6. Training Pipeline

### 6.1 Two-Stage Training Strategy

```
┌──────────────────────────────────────────────────────────────┐
│                  LLaVA Training Strategy                     │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Stage 1: Pretraining (Feature Alignment)                   │
│  ├─ Data: Image-caption pairs (CC3M, ~600K)                 │
│  ├─ Frozen: Vision Encoder ✅, LLM ✅                        │
│  ├─ Trainable: MM Projector only                            │
│  ├─ Goal: Align vision features to LLM space                │
│  └─ Duration: ~4 hours on 8×A100                            │
│                                                              │
│  Stage 2: Instruction Tuning (Task Learning)                │
│  ├─ Data: Instruction-following (LLaVA-150K, Medical-60K)  │
│  ├─ Frozen: Vision Encoder ✅                                │
│  ├─ Trainable: MM Projector ✅, LLM (LoRA) ✅               │
│  ├─ Goal: Learn to follow instructions                      │
│  └─ Duration: ~10-20 hours on 8×A100                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 Training Configuration

```python
# training_args.py
from transformers import TrainingArguments

# Stage 1: Pretraining
pretrain_args = TrainingArguments(
    output_dir="./checkpoints/stage1",
    
    # Model
    model_name_or_path="./llama-7b",
    vision_tower="openai/clip-vit-large-patch14-336",
    mm_projector_type="mlp2x_gelu",
    
    # Freezing
    freeze_backbone=True,          # Freeze LLM
    tune_mm_mlp_adapter=True,      # Train projector only
    
    # Training
    num_train_epochs=1,
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
    learning_rate=2e-3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Optimization
    bf16=True,
    tf32=True,
    dataloader_num_workers=4,
    
    # Save
    save_strategy="steps",
    save_steps=500,
    save_total_limit=1,
)

# Stage 2: Instruction Tuning
finetune_args = TrainingArguments(
    output_dir="./checkpoints/stage2",
    
    # Model
    model_name_or_path="./llama-7b",
    pretrain_mm_mlp_adapter="./checkpoints/stage1/mm_projector.bin",
    
    # LoRA
    lora_enable=True,
    lora_r=128,
    lora_alpha=256,
    lora_dropout=0.05,
    lora_bias="none",
    
    # Freezing
    freeze_backbone=False,         # Unfreeze LLM (with LoRA)
    tune_mm_mlp_adapter=True,      # Continue training projector
    mm_projector_lr=2e-5,          # Lower LR for projector
    
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    learning_rate=2e-4,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    
    # Optimization
    bf16=True,
    tf32=True,
    gradient_checkpointing=True,   # Save memory
    dataloader_num_workers=4,
    
    # Save
    save_strategy="steps",
    save_steps=1000,
    save_total_limit=2,
)
```

### 6.3 Data Format

```python
# Stage 1: Pretrain data format
{
    "id": "sample_001",
    "image": "image_001.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nProvide a brief description of the given image."
        },
        {
            "from": "gpt",
            "value": "A medical chest X-ray showing clear lung fields..."
        }
    ]
}

# Stage 2: Instruction tuning format
{
    "id": "sample_002",
    "image": "xray_002.jpg",
    "conversations": [
        {
            "from": "human",
            "value": "<image>\nWhat abnormalities can you identify in this chest X-ray?"
        },
        {
            "from": "gpt",
            "value": "Based on the X-ray, I can identify the following abnormalities:\n1. ..."
        }
    ]
}

# Multi-turn conversation
{
    "id": "sample_003",
    "image": "ct_scan_003.jpg",
    "conversations": [
        {"from": "human", "value": "<image>\nWhat organ is this?"},
        {"from": "gpt", "value": "This is a CT scan of the brain."},
        {"from": "human", "value": "What abnormalities do you see?"},
        {"from": "gpt", "value": "I can see a hyperdense area suggesting..."}
    ]
}
```

### 6.4 Training Script

```python
# train.py
import torch
from transformers import Trainer
from llava.train.llava_trainer import LLaVATrainer
from llava.model import LlavaLlamaForCausalLM
from llava.train.train import make_supervised_data_module

def train():
    # Load model
    model = LlavaLlamaForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )
    
    # Initialize vision modules
    model.get_model().initialize_vision_modules(
        model_args=model_args,
        fsdp=training_args.fsdp
    )
    
    # Setup LoRA if enabled
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
    
    # Freeze vision encoder
    for p in model.get_model().get_vision_tower().parameters():
        p.requires_grad = False
    
    # Prepare data
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args
    )
    
    # Train
    trainer = LLaVATrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )
    
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
```

---

## 7. Inference Pipeline

### 7.1 Basic Inference

```python
import torch
from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

# Initialize
model_path = "./llava_med_agent"
model_name = get_model_name_from_path(model_path)

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=model_name,
    load_8bit=False,
    load_4bit=False,
    device="cuda"
)

# Inference
from PIL import Image
image = Image.open("chest_xray.jpg")

# Process image
images_tensor = process_images(
    [image],
    image_processor,
    model.config
).to(model.device, dtype=torch.float16)

# Prepare prompt
from llava.conversation import conv_templates
conv = conv_templates["v1"].copy()
conv.append_message(conv.roles[0], "<image>\nDescribe this medical image.")
conv.append_message(conv.roles[1], None)
prompt = conv.get_prompt()

# Tokenize
input_ids = tokenizer_image_token(
    prompt,
    tokenizer,
    IMAGE_TOKEN_INDEX,
    return_tensors='pt'
).unsqueeze(0).to(model.device)

# Generate
with torch.inference_mode():
    output_ids = model.generate(
        input_ids,
        images=images_tensor,
        do_sample=False,
        temperature=0.2,
        max_new_tokens=512,
        use_cache=True
    )

# Decode
response = tokenizer.batch_decode(
    output_ids,
    skip_special_tokens=True
)[0].strip()

print(response)
```

### 7.2 Batch Inference

```python
def batch_inference(model, tokenizer, image_processor, image_paths, prompts):
    """Batch inference for multiple images"""
    
    # Load images
    images = [Image.open(path) for path in image_paths]
    
    # Process images
    images_tensor = process_images(
        images,
        image_processor,
        model.config
    ).to(model.device, dtype=torch.float16)
    
    # Prepare prompts
    input_ids_list = []
    for prompt in prompts:
        input_ids = tokenizer_image_token(
            prompt,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors='pt'
        )
        input_ids_list.append(input_ids)
    
    # Pad to same length
    max_len = max(len(ids) for ids in input_ids_list)
    input_ids_padded = torch.full(
        (len(input_ids_list), max_len),
        tokenizer.pad_token_id,
        dtype=torch.long
    )
    for i, ids in enumerate(input_ids_list):
        input_ids_padded[i, :len(ids)] = ids
    input_ids_padded = input_ids_padded.to(model.device)
    
    # Generate
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids_padded,
            images=images_tensor,
            do_sample=False,
            max_new_tokens=512,
            use_cache=True
        )
    
    # Decode
    responses = tokenizer.batch_decode(
        output_ids,
        skip_special_tokens=True
    )
    
    return responses
```

### 7.3 Streaming Generation

```python
from transformers import TextIteratorStreamer
from threading import Thread

def streaming_inference(model, tokenizer, image, prompt):
    """Stream generated tokens in real-time"""
    
    # Prepare inputs
    images_tensor = process_images([image], image_processor, model.config)
    images_tensor = images_tensor.to(model.device, dtype=torch.float16)
    
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt'
    ).unsqueeze(0).to(model.device)
    
    # Setup streamer
    streamer = TextIteratorStreamer(
        tokenizer,
        skip_prompt=True,
        skip_special_tokens=True
    )
    
    # Generation kwargs
    generation_kwargs = dict(
        input_ids=input_ids,
        images=images_tensor,
        streamer=streamer,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        use_cache=True
    )
    
    # Start generation in thread
    thread = Thread(target=model.generate, kwargs=generation_kwargs)
    thread.start()
    
    # Stream tokens
    generated_text = ""
    for new_text in streamer:
        generated_text += new_text
        print(new_text, end="", flush=True)
    
    thread.join()
    return generated_text
```

---

## 8. Best Practices

### 8.1 Memory Optimization

```python
# 1. Use 8-bit quantization
tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    load_8bit=True,  # Giảm ~50% VRAM
    device_map="auto"
)

# 2. Use 4-bit quantization (even more aggressive)
from transformers import BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4'
)

model = LlavaLlamaForCausalLM.from_pretrained(
    model_path,
    quantization_config=quantization_config,
    device_map="auto"
)

# 3. Gradient checkpointing (training)
model.gradient_checkpointing_enable()

# 4. Clear cache between inferences
torch.cuda.empty_cache()

# 5. Use smaller batch size
per_device_train_batch_size=4  # instead of 16
gradient_accumulation_steps=4  # compensate
```

### 8.2 Quality Optimization

```python
# 1. Better prompting
GOOD_PROMPT = """<image>
Analyze this medical image in detail. Describe:
1. The imaging modality
2. Anatomical structures visible
3. Any abnormalities or findings
4. Clinical significance

Be specific and use medical terminology."""

BAD_PROMPT = "<image>\nWhat is this?"

# 2. Temperature tuning
# For factual tasks: temperature=0.2 (more deterministic)
# For creative tasks: temperature=0.7 (more diverse)
output = model.generate(
    ...,
    temperature=0.2,      # Lower for medical diagnosis
    top_p=0.9,            # Nucleus sampling
    do_sample=True
)

# 3. Use beam search for better quality (slower)
output = model.generate(
    ...,
    num_beams=5,
    do_sample=False,
    early_stopping=True
)

# 4. Constrained generation (force specific format)
from transformers import LogitsProcessor

class ForceStartTokenProcessor(LogitsProcessor):
    def __init__(self, start_token_id):
        self.start_token_id = start_token_id
    
    def __call__(self, input_ids, scores):
        if input_ids.shape[1] == 1:  # First token
            scores[:, :] = float('-inf')
            scores[:, self.start_token_id] = 0
        return scores

output = model.generate(
    ...,
    logits_processor=[ForceStartTokenProcessor(tokenizer.encode("The")[0])]
)
```

### 8.3 Debugging Tips

```python
# 1. Visualize attention on image tokens
def visualize_attention(model, input_ids, images):
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            images=images,
            output_attentions=True
        )
    
    # Get last layer attention
    attention = outputs.attentions[-1]  # [B, num_heads, seq_len, seq_len]
    
    # Average over heads
    attention = attention.mean(dim=1)  # [B, seq_len, seq_len]
    
    # Extract attention to image tokens
    image_token_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero()
    image_attention = attention[0, :, image_token_pos[0, 1]:image_token_pos[0, 1]+576]
    
    return image_attention

# 2. Check image encoding
def check_image_features(model, image):
    image_tensor = process_images([image], image_processor, model.config)
    image_tensor = image_tensor.to(model.device)
    
    with torch.no_grad():
        # Vision encoder output
        vision_features = model.get_vision_tower()(image_tensor)
        print(f"Vision features shape: {vision_features.shape}")
        print(f"Vision features mean: {vision_features.mean():.4f}")
        print(f"Vision features std: {vision_features.std():.4f}")
        
        # After projector
        projected_features = model.get_model().mm_projector(vision_features)
        print(f"Projected features shape: {projected_features.shape}")
        print(f"Projected features mean: {projected_features.mean():.4f}")

# 3. Debug tokenization
def debug_tokenization(tokenizer, prompt):
    input_ids = tokenizer_image_token(
        prompt, tokenizer, IMAGE_TOKEN_INDEX
    )
    
    print(f"Prompt: {prompt}")
    print(f"Token IDs: {input_ids}")
    
    # Decode each token
    for i, token_id in enumerate(input_ids):
        if token_id == IMAGE_TOKEN_INDEX:
            token_str = "<IMAGE>"
        else:
            token_str = tokenizer.decode([token_id])
        print(f"{i}: {token_id} → '{token_str}'")
```

### 8.4 Performance Monitoring

```python
import time
import torch

class PerformanceMonitor:
    def __init__(self):
        self.timings = {}
        self.memory = {}
    
    def start(self, name):
        self.timings[name] = time.time()
        self.memory[name] = torch.cuda.memory_allocated() / 1024**3
    
    def end(self, name):
        elapsed = time.time() - self.timings[name]
        memory_used = torch.cuda.memory_allocated() / 1024**3 - self.memory[name]
        print(f"{name}:")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Memory: {memory_used:.2f}GB")
        return elapsed, memory_used

# Usage
monitor = PerformanceMonitor()

monitor.start("image_encoding")
image_features = model.encode_images(images_tensor)
monitor.end("image_encoding")

monitor.start("generation")
output_ids = model.generate(...)
monitor.end("generation")
```

---

## 📚 Tài Liệu Tham Khảo

### Papers
- [LLaVA: Visual Instruction Tuning](https://arxiv.org/abs/2304.08485)
- [LLaVA-Med: Training a Large Language-and-Vision Assistant for Biomedicine](https://arxiv.org/abs/2306.00890)
- [CLIP: Learning Transferable Visual Models](https://arxiv.org/abs/2103.00020)

### Code Repositories
- [LLaVA Official](https://github.com/haotian-liu/LLaVA)
- [LLaVA-Med Official](https://github.com/microsoft/LLaVA-Med)
- [Transformers Library](https://github.com/huggingface/transformers)

### Tutorials
- [LLaVA Training Tutorial](https://github.com/haotian-liu/LLaVA/blob/main/docs/MODEL_ZOO.md)
- [Fine-tuning Guide](https://github.com/haotian-liu/LLaVA/blob/main/docs/Finetune_Custom_Data.md)

---

*Tài liệu được tạo ngày 13/12/2025*
*Phiên bản: 1.0 - MMedAgent codebase*
