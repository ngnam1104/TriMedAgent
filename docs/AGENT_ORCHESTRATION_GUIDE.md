# 🤖 Hướng Dẫn Làm Chủ Agent Orchestration

> **MMedAgent Orchestration** - Hệ thống điều phối LLM và tools để xử lý các tác vụ y tế phức tạp

---

## 📋 Mục Lục

1. [Tổng Quan Architecture](#1-tổng-quan-architecture)
2. [Conversation Memory System](#2-conversation-memory-system)
3. [Phần 1: Tool Dispatching (Điều Phối)](#3-phần-1-tool-dispatching-điều-phối)
4. [Phần 2: Result Aggregation (Tổng Hợp)](#4-phần-2-result-aggregation-tổng-hợp)
5. [Tool Worker Communication](#5-tool-worker-communication)
6. [Customize & Extend](#6-customize--extend)
7. [Best Practices](#7-best-practices)
8. [Debugging & Troubleshooting](#8-debugging--troubleshooting)

---

## 1. Tổng Quan Architecture

### 1.1 Agent Orchestration Flow

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT ORCHESTRATION PIPELINE                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  USER INPUT                                                         │
│  ┌──────────────────────────────────────────────┐                  │
│  │ "Can you detect the tumor in this brain MRI?"│                  │
│  └────────────────┬─────────────────────────────┘                  │
│                   │                                                 │
│                   ▼                                                 │
│  ┌────────────────────────────────────────────────────────┐        │
│  │         LLM (MMedAgent) - FIRST RESPONSE              │        │
│  │  Generates structured tool call:                       │        │
│  │  {                                                     │        │
│  │    "thoughts🤔": "I need detection model",             │        │
│  │    "actions🚀": [{                                     │        │
│  │      "API_name": "grounding_dino",                     │        │
│  │      "API_params": {"prompts": ["tumor"]}              │        │
│  │    }],                                                 │        │
│  │    "value👉": "Let me detect..."                       │        │
│  │  }                                                     │        │
│  └────────────────┬───────────────────────────────────────┘        │
│                   │                                                 │
│         ╔═════════▼════════════════════════════════╗               │
│         ║   PART 1: TOOL DISPATCHING              ║               │
│         ╚═════════╤════════════════════════════════╝               │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Parse Tool Call (gradio_web_server:454)          │            │
│  │  • Extract API_name: "grounding_dino"              │            │
│  │  • Extract API_params                              │            │
│  │  • Prepare image & prompt                          │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Find Tool Worker (controller.py)                  │            │
│  │  • Query controller for worker address             │            │
│  │  • Load balancing (if multiple workers)            │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Call Tool Worker API                               │            │
│  │  POST /worker_generate                              │            │
│  │  • Send image + parameters                          │            │
│  │  • Wait for response                                │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│                   ▼                                                 │
│  ┌──────────────────────────────────────────┐                      │
│  │  Tool Worker Response                     │                      │
│  │  {                                        │                      │
│  │    "boxes": [[0.2, 0.3, 0.5, 0.6]],       │                      │
│  │    "logits": [0.95],                      │                      │
│  │    "phrases": ["tumor"]                   │                      │
│  │  }                                        │                      │
│  └────────────────┬─────────────────────────┘                      │
│                   │                                                 │
│         ╔═════════▼════════════════════════════════╗               │
│         ║   PART 2: RESULT AGGREGATION            ║               │
│         ╚═════════╤════════════════════════════════╝               │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Process Tool Response (gradio_web_server:520)     │            │
│  │  • Clean response (round numbers, format)          │            │
│  │  • Extract masks/images if any                     │            │
│  │  • Format for display                              │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Prepare Context for LLM (gradio_web_server:572)  │            │
│  │  new_response = "grounding_dino outputs:           │            │
│  │    {'boxes': [...], 'phrases': ['tumor']}"         │            │
│  │  + "\n\nAnswer my first question: ..."             │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│  ┌────────────────▼────────────────────────────────────┐           │
│  │  LLM (MMedAgent) - SECOND RESPONSE                 │           │
│  │  Synthesizes tool output into natural language:     │           │
│  │  "Based on the detection results, there is a       │           │
│  │   tumor located at coordinates [0.2, 0.3, 0.5, 0.6]│           │
│  │   in the brain MRI with confidence 95%."           │           │
│  └────────────────┬────────────────────────────────────┘           │
│                   │                                                 │
│  ┌────────────────▼───────────────────────────────────┐            │
│  │  Visualize Results (gradio_web_server:650)         │            │
│  │  • Plot bounding boxes on image                    │            │
│  │  • Plot masks if segmentation                      │            │
│  │  • Combine with text response                      │            │
│  └────────────────┬───────────────────────────────────┘            │
│                   │                                                 │
│                   ▼                                                 │
│  ┌──────────────────────────────────────────────┐                  │
│  │  FINAL OUTPUT TO USER                        │                  │
│  │  • Natural language answer                   │                  │
│  │  • Annotated image                           │                  │
│  └──────────────────────────────────────────────┘                  │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.2 Key Components

| Component | File | Responsibility |
|-----------|------|----------------|
| **Web Server** | `gradio_web_server_mmedagent.py` | UI, orchestration logic |
| **Controller** | `controller.py` | Worker registry, load balancing |
| **Model Worker** | `model_worker.py` | LLM inference |
| **Tool Workers** | `serve/*_worker.py` | Specialized tools (DINO, SAM, etc.) |
| **Parser** | `conversation.py` | Parse LLM structured output |

---

## 2. Conversation Memory System

### 2.1 Tổng Quan về Memory

**⚠️ Quan trọng:** MMedAgent sử dụng **in-session memory** (RAM), KHÔNG có persistent storage lâu dài.

```
┌───────────────────────────────────────────────────────────────┐
│                    MEMORY ARCHITECTURE                        │
├───────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌─────────────────────────────────────────────────┐         │
│  │  Session-based Memory (RAM only)                │         │
│  │  ┌───────────────────────────────────────────┐  │         │
│  │  │ Conversation State                        │  │         │
│  │  │  • messages: List[List[str]]              │  │         │
│  │  │  • images: Base64 encoded                 │  │         │
│  │  │  • offset: Starting index                 │  │         │
│  │  │  • system prompt                          │  │         │
│  │  │  • roles: [USER, ASSISTANT]               │  │         │
│  │  └───────────────────────────────────────────┘  │         │
│  │                                                  │         │
│  │  Limitations:                                    │         │
│  │  ❌ No database/file persistence                │         │
│  │  ❌ No cross-session memory                     │         │
│  │  ❌ No sliding window (full truncation only)    │         │
│  │  ✅ Text hard limit: 1536 chars (1200 w/ image)│         │
│  │  ✅ Model context: 2048 tokens (LLaMA-7B)      │         │
│  └─────────────────────────────────────────────────┘         │
│                                                               │
│  Scope:                                                       │
│  • Fresh start on page reload/clear button                   │
│  • Works for 5-10 turn conversations                         │
│  • NOT suitable for long-term patient history                │
│                                                               │
└───────────────────────────────────────────────────────────────┘
```

### 2.2 Conversation State Implementation

**File:** [`llava/conversation.py`](llava/conversation.py) (Lines 62-75)

```python
@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    
    # System prompt (instructions for agent)
    system: str
    
    # Roles: ["USER", "ASSISTANT"] or ["Human", "Assistant"]
    roles: List[str]
    
    # Message history: [[role, content], [role, content], ...]
    # Content can be: str | tuple(str, PIL.Image, mode, mask)
    messages: List[List[str]]
    
    # Starting index for processing (skip system messages)
    offset: int
    
    # Separator style (SINGLE, TWO, LLAMA_2, etc.)
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None
    
    # Model version identifier
    version: str = "Unknown"
    
    # Skip flag for invalid inputs
    skip_next: bool = False

# Example conversation state
state = Conversation(
    system="You are a helpful medical AI assistant...",
    roles=["USER", "ASSISTANT"],
    messages=[
        ["USER", "What's in this X-ray?"],
        ["ASSISTANT", "Let me analyze..."]
    ],
    offset=0
)
```

### 2.3 Memory Operations

#### 2.3.1 Add Message

```python
def append_message(self, role, message):
    """
    Add a new message to conversation history
    
    Args:
        role: "USER" or "ASSISTANT"
        message: Text or tuple(text, image, mode, mask)
    """
    self.messages.append([role, message])

# Usage
state.append_message("USER", "Detect tumor in this CT scan")
state.append_message("ASSISTANT", None)  # Will be filled later
```

#### 2.3.2 Get Conversation Prompt

```python
def get_prompt(self):
    """
    Convert conversation history to prompt string
    
    Format depends on sep_style:
    - SINGLE: "USER: msg###ASSISTANT: msg###"
    - TWO: "USER: msg</s>ASSISTANT: msg</s>"
    - LLAMA_2: "[INST] msg [/INST] msg"
    
    Returns:
        Full prompt string for LLM
    """
    messages = self.messages
    
    # Handle image in first message
    if len(messages) > 0 and type(messages[0][1]) is tuple:
        messages = self.messages.copy()
        init_role, init_msg = messages[0].copy()
        init_msg = init_msg[0].replace("<image>", "").strip()
        messages[0] = (init_role, "<image>\n" + init_msg)
    
    # Build prompt based on separator style
    if self.sep_style == SeparatorStyle.SINGLE:
        ret = self.system + self.sep
        for role, message in messages:
            if message:
                ret += role + ": " + message + self.sep
            else:
                ret += role + ":"
        return ret
    # ... other styles

# Example output
prompt = state.get_prompt()
# "You are a medical assistant.###USER: Detect tumor###ASSISTANT:"
```

#### 2.3.3 Copy State (Start New Conversation)

**File:** [`gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 214, 228, 277)

```python
def copy(self):
    """
    Create a fresh copy of conversation state
    Used when:
    - New image uploaded (reset conversation)
    - Clear button clicked
    - Start new session
    """
    return Conversation(
        system=self.system,
        roles=self.roles,
        messages=[[x, y] for x, y in self.messages],  # Deep copy
        offset=self.offset,
        sep_style=self.sep_style,
        sep=self.sep,
        sep2=self.sep2,
        version=self.version
    )

# Usage: Reset conversation when new image uploaded
if image is not None:
    state = default_conversation.copy()  # ← Fresh state
    text = text + '\n<image>'
    state.append_message(state.roles[0], (text, image, mode))
```

### 2.4 Input Length Limitations

**File:** [`gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 272-275)

```python
def add_text(state, text, image_dict, ...):
    """Add user input to conversation"""
    
    # Hard text cut-off
    text = text[:1536]  # Maximum 1536 characters
    
    if image is not None:
        text = text[:1200]  # Maximum 1200 chars with image
        
    # No sliding window - just truncate
    # No warning to user about truncation
    
    state.append_message(state.roles[0], text)
    return state
```

**Limitations:**

| Scenario | Max Length | Behavior |
|----------|------------|----------|
| **Text only** | 1536 chars | Hard truncate, no warning |
| **Text + Image** | 1200 chars | Hard truncate, no warning |
| **Model context** | 2048 tokens | LLaMA-7B native limit |
| **Conversation turns** | Unlimited* | *But limited by model context |

### 2.5 Session Lifecycle

```python
# ===== SESSION START =====
# 1. Page load → load_demo()
state = default_conversation.copy()  # Fresh empty state

# 2. User uploads image + question
state.append_message("USER", (question, image, "Default"))
state.append_message("ASSISTANT", None)

# 3. LLM generates tool call
llm_response = "thoughts🤔...actions🚀[{...}]...value👉..."
state.messages[-1][-1] = llm_response

# 4. Tool executes → add tool output
tool_output = "grounding_dino outputs: {...}\n\nAnswer my first question: ..."
state.append_message("USER", tool_output)
state.append_message("ASSISTANT", None)

# 5. LLM synthesizes final answer
final_answer = "Based on detection results, there is a tumor at..."
state.messages[-1][-1] = final_answer

# 6. User asks follow-up (maintains history)
state.append_message("USER", "How big is it?")
state.append_message("ASSISTANT", None)
# → LLM sees full conversation history

# ===== SESSION END =====
# 7. User reloads page OR clicks "Clear" button
state = default_conversation.copy()  # ← All history LOST
```

### 2.6 Memory Limitations & Workarounds

#### ❌ Problem 1: No Persistent Storage

```python
# MMedAgent DOES NOT have:
# - Database storage (SQLite, PostgreSQL)
# - File-based history (JSON, CSV)
# - Redis cache
# - Session recovery after crash
```

**Workaround:** Implement custom storage

```python
import json
from datetime import datetime

class ConversationLogger:
    """Save conversations to disk"""
    
    def __init__(self, log_dir="conversation_logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
    
    def save_conversation(self, state, session_id):
        """Save conversation state to JSON"""
        timestamp = datetime.now().isoformat()
        filename = f"{session_id}_{timestamp}.json"
        
        data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "messages": state.messages,
            "system": state.system,
            "version": state.version
        }
        
        filepath = self.log_dir / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load_conversation(self, session_id):
        """Load most recent conversation for session"""
        files = list(self.log_dir.glob(f"{session_id}_*.json"))
        if not files:
            return None
        
        latest_file = max(files, key=lambda p: p.stat().st_mtime)
        with open(latest_file, 'r', encoding='utf-8') as f:
            return json.load(f)

# Usage in gradio_web_server.py
logger = ConversationLogger()

def http_bot(state, ...):
    # Before processing
    session_id = request.session.get('session_id', str(uuid.uuid4()))
    
    # ... process conversation ...
    
    # After processing
    logger.save_conversation(state, session_id)
```

#### ❌ Problem 2: No Context Window Management

```python
# MMedAgent DOES NOT have:
# - Sliding window (keep recent N messages)
# - Summarization (compress old messages)
# - Token counting (check context limit)
# - Graceful degradation when limit exceeded
```

**Workaround:** Implement sliding window

```python
def truncate_conversation(state, max_messages=10):
    """
    Keep only recent N message pairs
    Prevents context overflow for long conversations
    
    Args:
        state: Conversation state
        max_messages: Max message pairs to keep
    """
    # Keep system message + recent messages
    if len(state.messages) > max_messages * 2:
        # Keep first image if present
        first_msg = state.messages[0]
        has_image = isinstance(first_msg[1], tuple)
        
        if has_image:
            # Keep: first image + recent N pairs
            state.messages = [first_msg] + state.messages[-(max_messages * 2):]
        else:
            # Keep: recent N pairs only
            state.messages = state.messages[-(max_messages * 2):]
    
    return state

# Usage
def http_bot(state, ...):
    # Before LLM call
    state = truncate_conversation(state, max_messages=5)  # Keep last 5 turns
    
    # Continue processing...
```

#### ❌ Problem 3: No Multi-Session Management

```python
# MMedAgent DOES NOT have:
# - User authentication
# - Session isolation (multiple users)
# - Concurrent conversation tracking
```

**Workaround:** Use Gradio session state

```python
import uuid

# Track sessions in memory
active_sessions = {}  # {session_id: conversation_state}

def load_demo(url_params, request: gr.Request):
    # Generate or retrieve session ID
    session_id = request.session.get('session_id')
    if session_id is None:
        session_id = str(uuid.uuid4())
        request.session['session_id'] = session_id
    
    # Load or create conversation state
    if session_id in active_sessions:
        state = active_sessions[session_id]
    else:
        state = default_conversation.copy()
        active_sessions[session_id] = state
    
    return state, ...

def http_bot(state, ..., request: gr.Request):
    session_id = request.session['session_id']
    
    # Process conversation...
    
    # Update session state
    active_sessions[session_id] = state
    
    return state, ...
```

### 2.7 Best Practices for Memory Management

#### ✅ DO:

```python
# 1. Reset conversation when changing images
if new_image is not None:
    state = default_conversation.copy()

# 2. Check message count before processing
if len(state.messages) > 20:
    # Warn user or truncate
    state = truncate_conversation(state)

# 3. Clear state on logout/session end
def clear_conversation():
    return default_conversation.copy()

# 4. Save important conversations externally
if user_requested_save:
    save_to_database(state.messages)
```

#### ❌ DON'T:

```python
# 1. Assume conversation persists after reload
# state.messages will be EMPTY after page refresh

# 2. Store sensitive patient data in conversation
# No encryption, no access control, RAM only

# 3. Rely on conversation for critical context
# Long conversations → truncated input → model confusion

# 4. Mix conversations between different images
# Each image should have fresh conversation state
```

### 2.8 Context Window Calculation

```python
def estimate_token_count(state):
    """
    Rough estimation of token usage
    LLaMA tokenizer: ~1.3 tokens per word on average
    
    Returns:
        Approximate token count
    """
    total_text = ""
    
    # System prompt
    total_text += state.system
    
    # All messages
    for role, message in state.messages:
        if isinstance(message, str):
            total_text += message
        elif isinstance(message, tuple):
            # Text + image special tokens
            total_text += message[0]
            total_text += "<image>" * 256  # Image uses 256 tokens
    
    # Rough estimate: 1.3 tokens per word
    word_count = len(total_text.split())
    token_estimate = int(word_count * 1.3)
    
    return token_estimate

# Usage
tokens = estimate_token_count(state)
if tokens > 1800:  # Leave 248 tokens for response
    print(f"⚠️ Warning: {tokens} tokens (limit: 2048)")
    state = truncate_conversation(state)
```

### 2.9 Summary

| Feature | MMedAgent | Recommended |
|---------|-----------|-------------|
| **Memory Type** | In-session (RAM) | ✅ OK for demos |
| **Persistence** | ❌ None | ➕ Add file/DB logging |
| **Context Management** | ❌ Hard truncate only | ➕ Add sliding window |
| **Multi-User** | ❌ Single session | ➕ Add session isolation |
| **Max Input** | 1536 chars | ➕ Add token counting |
| **Max Context** | 2048 tokens | ⚠️ Cannot change (model limit) |
| **Cross-Session** | ❌ Not supported | ➕ Add persistent storage |

**Recommendation:** For production medical applications, implement:
1. Database storage (patient history, previous diagnoses)
2. Sliding window or summarization (long conversations)
3. Session management (multiple concurrent users)
4. Token counting (prevent context overflow)
5. Audit logging (regulatory compliance)

---

## 3. Phần 1: Tool Dispatching (Điều Phối)

### 2.1 Parse Tool Calls từ LLM Output

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 454-477)

```python
def parse_tool_call_from_llm_output(model_output_text):
    """
    Parse structured output from LLM to extract tool calls
    
    Expected format:
    "thoughts🤔" <reasoning>
    "actions🚀" [{"API_name": "...", "API_params": {...}}]
    "value👉" <message>
    
    Returns:
        tool_cfg: List of tool configurations or None
    """
    try:
        # Regex pattern to extract three parts
        pattern = r'"thoughts🤔"(.*)"actions🚀"(.*)"value👉"(.*)'
        matches = re.findall(pattern, model_output_text, re.DOTALL)
        
        if len(matches) > 0:
            # Extract actions (second group)
            actions_str = matches[0][1].strip()
            
            # Parse JSON (handle both single/double quotes)
            try:
                tool_cfg = json.loads(actions_str)
            except Exception as e:
                # Fallback: replace single quotes with double quotes
                tool_cfg = json.loads(actions_str.replace("\'", "\""))
            
            print(f"✅ Parsed tool config: {tool_cfg}")
            return tool_cfg
        else:
            print("❌ No tool pattern found in LLM output")
            return None
            
    except Exception as e:
        logger.error(f"Failed to parse tool config: {e}")
        return None

# Example Usage
model_output = '''
"thoughts🤔" I need to detect the tumor in this MRI scan using an object detection model.
"actions🚀" [{"API_name": "grounding_dino", "API_params": {"prompts": ["tumor", "lesion"]}}]
"value👉" Let me use the grounding_dino model to detect potential tumors in the image.
'''

tool_cfg = parse_tool_call_from_llm_output(model_output)
# Returns: [{"API_name": "grounding_dino", "API_params": {"prompts": ["tumor", "lesion"]}}]
```

### 2.2 Prepare Tool Call Parameters

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 476-510)

```python
def prepare_tool_parameters(tool_cfg, state, prompt, api_key):
    """
    Prepare parameters for tool API call
    
    Args:
        tool_cfg: Parsed tool configuration
        state: Conversation state (contains images, history)
        prompt: User prompt
        api_key: OpenAI API key (for some tools)
    
    Returns:
        api_name: Tool name
        api_paras: Complete parameters for tool
    """
    assert len(tool_cfg) == 1, f"Only one tool supported, got: {len(tool_cfg)}"
    
    # Extract tool name and user params
    api_name = tool_cfg[0]['API_name']
    user_params = tool_cfg[0]['API_params']
    
    # Remove 'image' if present (we'll add it from state)
    user_params.pop('image', None)
    
    # Get image from conversation state
    images = state.get_raw_images()
    image = images[0] if len(images) > 0 else None
    
    # Build complete parameters
    api_paras = {
        'image': image,              # PIL Image
        'prompt': prompt,            # Original user prompt
        'box_threshold': 0.3,        # Default detection threshold
        'text_threshold': 0.25,      # Default text threshold
        'openai_key': api_key,       # For RAG tools
        **user_params                # Merge user-specified params
    }
    
    # Handle special cases
    if api_name == 'inpainting':
        # Inpainting needs mask
        api_paras['mask'] = getattr(state, 'mask_rle', None)
        
    if api_name == 'controlnet':
        # ControlNet needs segmentation
        api_paras['mask'] = getattr(state, 'image_seg', None)
        api_paras['mode'] = 'controlnet'
        
    if api_name == 'seem':
        # SEEM needs reference image/mask
        api_paras['refimg'] = getattr(state, 'reference_image', None)
        api_paras['refmask'] = getattr(state, 'reference_mask', None)
    
    return api_name, api_paras

# Example
tool_cfg = [{"API_name": "grounding_dino", "API_params": {"prompts": ["tumor"]}}]
api_name, api_paras = prepare_tool_parameters(tool_cfg, state, prompt, api_key)

print(api_name)  # "grounding_dino"
print(api_paras)
# {
#   'image': <PIL.Image>,
#   'prompt': "Detect tumor in this MRI",
#   'box_threshold': 0.3,
#   'text_threshold': 0.25,
#   'prompts': ['tumor']
# }
```

### 2.3 Find Tool Worker Address

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Line 508)

```python
def get_worker_addr(controller_url, api_name):
    """
    Query controller to find worker address for specific tool
    
    Args:
        controller_url: Controller URL (e.g., "http://localhost:20001")
        api_name: Tool name (e.g., "grounding_dino")
    
    Returns:
        worker_addr: Worker URL (e.g., "http://localhost:21001")
    """
    # Query controller
    response = requests.post(
        controller_url + "/get_worker_address",
        json={"model": api_name}
    )
    
    if response.status_code == 200:
        worker_addr = response.json()["address"]
        return worker_addr
    else:
        raise ValueError(f"No worker found for tool: {api_name}")

# Example
controller_url = "http://localhost:20001"
api_name = "grounding_dino"

worker_addr = get_worker_addr(controller_url, api_name)
# Returns: "http://localhost:21001"
```

**Controller Implementation:**

**File:** [`llava/serve/controller.py`](llava/serve/controller.py)

```python
class Controller:
    def __init__(self):
        self.workers = {}  # {worker_name: WorkerInfo}
        
    def register_worker(self, worker_name, worker_info):
        """Register a tool worker"""
        self.workers[worker_name] = WorkerInfo(
            worker_addr=worker_info['worker_addr'],
            model_names=worker_info['model_names'],
            speed=worker_info['speed'],
            queue_length=0
        )
        
    def get_worker_address(self, model_name):
        """
        Get worker address for a specific model/tool
        
        Load balancing strategy:
        1. Find all workers that support this model
        2. Sort by queue_length (pick least busy)
        3. Return worker address
        """
        # Find workers that support this model
        workers = [
            (name, info) 
            for name, info in self.workers.items()
            if model_name in info.model_names
        ]
        
        if not workers:
            raise ValueError(f"No worker found for model: {model_name}")
        
        # Sort by queue length (load balancing)
        workers.sort(key=lambda x: x[1].queue_length)
        
        # Return least busy worker
        worker_name, worker_info = workers[0]
        return worker_info.worker_addr

# Usage in FastAPI endpoint
@app.post("/get_worker_address")
async def api_get_worker_address(request: Request):
    data = await request.json()
    model_name = data["model"]
    
    try:
        worker_addr = controller.get_worker_address(model_name)
        return {"address": worker_addr}
    except ValueError as e:
        return {"error": str(e)}, 404
```

### 2.4 Call Tool Worker

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 510-520)

```python
def call_tool_worker(worker_addr, api_params):
    """
    Make HTTP request to tool worker
    
    Args:
        worker_addr: Worker URL
        api_params: Parameters including image, prompt, etc.
    
    Returns:
        tool_response: Dict with tool outputs
    """
    headers = {"User-Agent": "MMedAgent Client"}
    
    # POST request to worker
    response = requests.post(
        worker_addr + "/worker_generate",
        headers=headers,
        json=api_params,
        timeout=300  # 5 minutes timeout
    )
    
    if response.status_code == 200:
        tool_response = response.json()
        return tool_response
    else:
        raise RuntimeError(f"Tool worker error: {response.text}")

# Example
worker_addr = "http://localhost:21001"
api_params = {
    'image': base64_encoded_image,
    'prompts': ['tumor', 'lesion'],
    'box_threshold': 0.3
}

tool_response = call_tool_worker(worker_addr, api_params)
print(tool_response)
# {
#   'boxes': [[0.2, 0.3, 0.5, 0.6], [0.7, 0.1, 0.9, 0.4]],
#   'logits': [0.95, 0.87],
#   'phrases': ['tumor', 'lesion']
# }
```

### 2.5 Complete Dispatching Flow

```python
def dispatch_tool_call(model_output_text, state, prompt, api_key, controller_url):
    """
    Complete tool dispatching pipeline
    
    Args:
        model_output_text: LLM output with tool call
        state: Conversation state
        prompt: User prompt
        api_key: OpenAI API key
        controller_url: Controller URL
    
    Returns:
        tool_response: Tool output
        api_name: Tool name used
    """
    # Step 1: Parse tool call
    tool_cfg = parse_tool_call_from_llm_output(model_output_text)
    
    if tool_cfg is None or len(tool_cfg) == 0:
        return None, None
    
    # Step 2: Prepare parameters
    api_name, api_params = prepare_tool_parameters(
        tool_cfg, state, prompt, api_key
    )
    
    # Step 3: Find worker
    worker_addr = get_worker_addr(controller_url, api_name)
    print(f"🔧 Calling {api_name} at {worker_addr}")
    
    # Step 4: Call worker
    tool_response = call_tool_worker(worker_addr, api_params)
    print(f"✅ Tool response received: {tool_response}")
    
    return tool_response, api_name

# Usage
tool_response, api_name = dispatch_tool_call(
    model_output_text=llm_output,
    state=conversation_state,
    prompt=user_prompt,
    api_key=openai_key,
    controller_url="http://localhost:20001"
)
```

---

## 3. Phần 2: Result Aggregation (Tổng Hợp)

### 3.1 Clean Tool Response

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 520-565)

```python
def clean_tool_response(tool_response, api_name):
    """
    Clean and format tool response for display
    
    Args:
        tool_response: Raw response from tool worker
        api_name: Tool name
    
    Returns:
        cleaned_response: Formatted response
        extracted_data: Special data (masks, images, etc.)
    """
    def round_number(x, decimals=4):
        """Round number to N decimals"""
        if isinstance(x, (int, float)):
            return round(x, decimals)
        return x
    
    cleaned_response = {}
    extracted_data = {
        'masks_rle': None,
        'edited_image': None,
        'image_seg': None,
        'iou_sort_masks': None
    }
    
    # Round numbers in boxes
    if 'boxes' in tool_response:
        try:
            cleaned_response['boxes'] = [
                [round_number(b) for b in box]
                for box in tool_response['boxes']
            ]
        except:
            cleaned_response['boxes'] = tool_response['boxes']
    
    # Round logits/scores
    if 'logits' in tool_response:
        try:
            cleaned_response['logits'] = [
                round_number(l) for l in tool_response['logits']
            ]
        except:
            cleaned_response['logits'] = tool_response['logits']
    
    if 'scores' in tool_response:
        try:
            cleaned_response['scores'] = [
                round_number(s) for s in tool_response['scores']
            ]
        except:
            cleaned_response['scores'] = tool_response['scores']
    
    # Keep phrases as-is
    if 'phrases' in tool_response:
        cleaned_response['phrases'] = tool_response['phrases']
    
    # Extract masks (don't show in text)
    if 'masks_rle' in tool_response:
        extracted_data['masks_rle'] = tool_response.pop('masks_rle')
    
    # Extract edited image
    if 'edited_image' in tool_response:
        extracted_data['edited_image'] = tool_response.pop('edited_image')
    
    # Extract segmentation
    if 'image_seg' in tool_response:
        extracted_data['image_seg'] = tool_response.pop('image_seg')
    
    # Extract sorted masks
    if 'iou_sort_masks' in tool_response:
        extracted_data['iou_sort_masks'] = tool_response.pop('iou_sort_masks')
    
    # Remove unnecessary fields
    tool_response.pop('size', None)
    
    # Handle special cases
    if api_name == 'easyocr':
        # OCR doesn't need boxes in text
        tool_response.pop('boxes', None)
        tool_response.pop('scores', None)
    
    if 'retrieval_results' in tool_response:
        # Format retrieval results
        cleaned_response['retrieval_results'] = [
            {
                'caption': item['caption'],
                'similarity': round_number(item['similarity'])
            }
            for item in tool_response['retrieval_results']
        ]
    
    # If empty response, add default message
    if len(cleaned_response) == 0:
        cleaned_response['message'] = f"The {api_name} has processed the image."
    
    return cleaned_response, extracted_data

# Example
tool_response = {
    'boxes': [[0.123456, 0.234567, 0.456789, 0.567890]],
    'logits': [0.951234],
    'phrases': ['tumor'],
    'masks_rle': '<base64_encoded_mask>',
    'size': [512, 512]
}

cleaned, extracted = clean_tool_response(tool_response, 'grounding_dino')
print(cleaned)
# {
#   'boxes': [[0.1235, 0.2346, 0.4568, 0.5679]],
#   'logits': [0.9512],
#   'phrases': ['tumor']
# }

print(extracted)
# {
#   'masks_rle': '<base64_encoded_mask>',
#   'edited_image': None,
#   ...
# }
```

### 3.2 Prepare Context for LLM

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 572-585)

```python
def prepare_llm_context_with_tool_output(
    api_name, 
    tool_response, 
    original_question,
    state
):
    """
    Prepare context for LLM to synthesize final answer
    
    Args:
        api_name: Tool name used
        tool_response: Cleaned tool response
        original_question: User's original question
        state: Conversation state
    
    Returns:
        new_prompt: Formatted prompt for LLM
    """
    # Format tool output as text
    tool_output_str = f"{api_name} model outputs: {tool_response}\n\n"
    
    # Extract original question from state
    # Handle tuple format (text, image, mode)
    if isinstance(original_question, tuple):
        original_question = original_question[0].replace("<image>", "")
    original_question = original_question.strip()
    
    # Build new prompt
    new_prompt = (
        f"{tool_output_str}"
        f"Answer my first question: {original_question}"
    )
    
    # Add to conversation state
    state.append_message(state.roles[0], new_prompt)  # USER role
    state.append_message(state.roles[1], None)         # ASSISTANT role (to be filled)
    
    return new_prompt

# Example
api_name = "grounding_dino"
tool_response = {
    'boxes': [[0.2, 0.3, 0.5, 0.6]],
    'logits': [0.95],
    'phrases': ['tumor']
}
original_question = "Can you detect the tumor in this MRI?"

new_prompt = prepare_llm_context_with_tool_output(
    api_name, tool_response, original_question, state
)

print(new_prompt)
# Output:
# grounding_dino model outputs: {'boxes': [[0.2, 0.3, 0.5, 0.6]], 
#   'logits': [0.95], 'phrases': ['tumor']}
#
# Answer my first question: Can you detect the tumor in this MRI?
```

### 3.3 LLM Synthesis

```python
def synthesize_final_answer(state, model_name, controller_url):
    """
    Call LLM again to synthesize tool output into natural language
    
    Args:
        state: Conversation state (with tool output added)
        model_name: LLM model name
        controller_url: Controller URL
    
    Returns:
        final_answer: Natural language answer
    """
    # Get LLM worker
    worker_addr = get_worker_addr(controller_url, model_name)
    
    # Prepare request
    prompt = state.get_prompt()  # Full conversation with tool output
    
    # Generate response
    response = requests.post(
        worker_addr + "/worker_generate",
        json={
            "prompt": prompt,
            "temperature": 0.2,
            "max_new_tokens": 512,
            "stop": state.sep if state.sep_style == SeparatorStyle.SINGLE else state.sep2
        }
    )
    
    final_answer = response.json()["text"]
    
    # Add to state
    state.messages[-1][-1] = final_answer
    
    return final_answer

# Example conversation flow:
# 
# User: "Can you detect the tumor?"
# LLM (1st): "thoughts🤔" ... "actions🚀" [grounding_dino] ...
# → Tool called
# System adds: "grounding_dino outputs: {'boxes': [...], 'phrases': ['tumor']}"
#              "Answer my first question: Can you detect the tumor?"
# LLM (2nd): "Based on the grounding_dino detection results, I can confirm 
#             there is a tumor detected in the brain MRI at coordinates 
#             [0.2, 0.3, 0.5, 0.6] with 95% confidence."
```

### 3.4 Visualize Results

**File:** [`llava/serve/gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) (Lines 640-690)

```python
def visualize_tool_results(
    api_name,
    tool_response,
    original_image,
    extracted_data,
    state
):
    """
    Create visualizations based on tool output
    
    Args:
        api_name: Tool name
        tool_response: Tool output
        original_image: Original input image
        extracted_data: Extracted masks/images
        state: Conversation state
    
    Returns:
        annotated_image: Image with annotations
    """
    from PIL import Image
    from io import BytesIO
    import base64
    
    annotated_image = None
    
    # 1. Detection boxes (Grounding DINO)
    if api_name in ['grounding_dino', 'ram+grounding_dino', 'grounding dino']:
        # Load original image
        img = Image.open(
            BytesIO(base64.b64decode(state.get_images()[0]))
        ).convert("RGB")
        
        # Plot bounding boxes
        annotated_image = plot_boxes(img, tool_response)
        
    # 2. Detection + Segmentation (Grounded SAM)
    elif api_name in ['grounding_dino+sam', 'grounded_sam', 'grounding dino + MedSAM']:
        img = Image.open(
            BytesIO(base64.b64decode(state.get_images()[0]))
        ).convert("RGB")
        
        # Plot boxes
        annotated_image = plot_boxes(img, tool_response)
        
        # Plot masks on top
        annotated_image = plot_masks(annotated_image, tool_response)
        
    # 3. Segmentation only (MedSAM)
    elif api_name == 'sam':
        img = Image.open(
            BytesIO(base64.b64decode(state.get_images()[0]))
        ).convert("RGB")
        
        if 'points' in tool_response:
            # Point-based segmentation
            annotated_image = plot_masks(img, tool_response)
            annotated_image = plot_points(annotated_image, tool_response)
        else:
            # Box-based segmentation
            annotated_image = plot_boxes(img, tool_response)
            annotated_image = plot_masks(annotated_image, tool_response)
    
    # 4. Edited image (Inpainting, ControlNet)
    elif extracted_data['edited_image'] is not None:
        annotated_image = Image.open(
            BytesIO(base64.b64decode(extracted_data['edited_image']))
        ).convert("RGB")
    
    # 5. Multiple masks (sorted by IoU)
    if extracted_data['iou_sort_masks'] is not None:
        annotated_images = [
            Image.open(BytesIO(base64.b64decode(mask))).convert("RGB")
            for mask in extracted_data['iou_sort_masks']
        ]
        return annotated_images  # Return list
    
    return annotated_image


def plot_boxes(image, response):
    """Draw bounding boxes on image"""
    from PIL import ImageDraw, ImageFont
    
    draw = ImageDraw.Draw(image)
    boxes = response.get('boxes', [])
    phrases = response.get('phrases', [])
    logits = response.get('logits', [])
    
    W, H = image.size
    
    for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
        # Denormalize coordinates
        x1, y1, x2, y2 = box
        x1, y1, x2, y2 = int(x1 * W), int(y1 * H), int(x2 * W), int(y2 * H)
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # Draw label
        label = f"{phrase} {logit:.2f}"
        draw.text((x1, y1 - 10), label, fill='red')
    
    return image


def plot_masks(image, response):
    """Overlay segmentation masks on image"""
    import numpy as np
    from PIL import Image
    
    if 'masks_rle' not in response:
        return image
    
    masks = response['masks_rle']
    image_array = np.array(image)
    
    for mask_rle in masks:
        # Decode RLE mask
        mask = decode_rle_mask(mask_rle)
        
        # Create colored overlay
        color = np.array([255, 0, 0])  # Red
        colored_mask = np.zeros_like(image_array)
        colored_mask[mask > 0] = color
        
        # Blend with original image
        alpha = 0.5
        image_array = (
            (1 - alpha) * image_array + 
            alpha * colored_mask
        ).astype(np.uint8)
    
    return Image.fromarray(image_array)
```

### 3.5 Complete Aggregation Flow

```python
def aggregate_tool_results(
    api_name,
    tool_response,
    original_question,
    state,
    model_name,
    controller_url
):
    """
    Complete result aggregation pipeline
    
    Args:
        api_name: Tool name
        tool_response: Raw tool response
        original_question: User's question
        state: Conversation state
        model_name: LLM model name
        controller_url: Controller URL
    
    Returns:
        final_answer: Natural language answer
        annotated_image: Visualized results
    """
    # Step 1: Clean response
    cleaned_response, extracted_data = clean_tool_response(
        tool_response, api_name
    )
    print(f"🧹 Cleaned response: {cleaned_response}")
    
    # Step 2: Prepare context for LLM
    new_prompt = prepare_llm_context_with_tool_output(
        api_name, cleaned_response, original_question, state
    )
    print(f"📝 Prepared prompt:\n{new_prompt}")
    
    # Step 3: LLM synthesizes answer
    final_answer = synthesize_final_answer(
        state, model_name, controller_url
    )
    print(f"💬 Final answer: {final_answer}")
    
    # Step 4: Visualize results
    annotated_image = visualize_tool_results(
        api_name,
        tool_response,
        state.get_raw_images()[0],
        extracted_data,
        state
    )
    print(f"🎨 Created visualization")
    
    # Step 5: Update state with visualization
    if annotated_image is not None:
        state.messages[-1][-1] = (
            state.messages[-1][-1],  # Text answer
            annotated_image,          # Image
            "Crop"                    # Display mode
        )
    
    return final_answer, annotated_image

# Usage
final_answer, viz = aggregate_tool_results(
    api_name="grounding_dino",
    tool_response=raw_tool_output,
    original_question="Detect tumor in MRI",
    state=conversation_state,
    model_name="llava_med_agent",
    controller_url="http://localhost:20001"
)
```

---

## 4. Tool Worker Communication

### 4.1 Tool Worker Interface

**File:** `serve/grounding_dino_worker.py` (Example)

```python
class ToolWorker:
    """Base class for tool workers"""
    
    def __init__(self, controller_addr, worker_addr, model_names):
        self.controller_addr = controller_addr
        self.worker_addr = worker_addr
        self.model_names = model_names
        
        # Load model
        self.model = self.load_model()
        
        # Register with controller
        self.register_to_controller()
        
    def register_to_controller(self):
        """Register this worker with controller"""
        url = self.controller_addr + "/register_worker"
        data = {
            "worker_name": self.worker_addr,
            "model_names": self.model_names,
            "check_heart_beat": True,
            "worker_status": self.get_status()
        }
        response = requests.post(url, json=data)
        assert response.status_code == 200
        
    def get_status(self):
        """Return worker status"""
        return {
            "model_names": self.model_names,
            "speed": 1,
            "queue_length": 0
        }
    
    @abstractmethod
    def load_model(self):
        """Load the tool model"""
        pass
    
    @abstractmethod
    def generate(self, params):
        """Run inference"""
        pass

# FastAPI endpoint
@app.post("/worker_generate")
async def api_generate(request: Request):
    """
    Handle inference request
    
    Expected params:
        - image: base64 encoded image or PIL image
        - prompts: List of text prompts
        - box_threshold: Detection threshold
        - ... (tool-specific params)
    
    Returns:
        - boxes: Detected boxes
        - logits: Confidence scores
        - phrases: Detected phrases
        - ... (tool-specific outputs)
    """
    params = await request.json()
    
    # Process image
    image = load_image(params['image'])
    
    # Run inference
    result = worker.generate(params)
    
    return result
```

### 4.2 Request/Response Format

**Request Format:**
```json
{
  "image": "<base64_encoded_image>",
  "prompts": ["tumor", "lesion"],
  "box_threshold": 0.3,
  "text_threshold": 0.25
}
```

**Response Format:**
```json
{
  "boxes": [[0.2, 0.3, 0.5, 0.6], [0.7, 0.1, 0.9, 0.4]],
  "logits": [0.95, 0.87],
  "phrases": ["tumor", "lesion"]
}
```

---

## 5. Customize & Extend

### 5.1 Add New Tool

**Step 1: Create Tool Worker**

```python
# serve/my_new_tool_worker.py
from base_worker import ToolWorker

class MyNewToolWorker(ToolWorker):
    def load_model(self):
        """Load your model"""
        from my_model import MyModel
        return MyModel.from_pretrained("my-model")
    
    def generate(self, params):
        """Run inference"""
        image = params['image']
        custom_param = params.get('custom_param', 'default')
        
        # Run your model
        output = self.model.predict(image, custom_param)
        
        # Format response
        return {
            'result': output,
            'confidence': 0.95
        }

# Start worker
if __name__ == "__main__":
    worker = MyNewToolWorker(
        controller_addr="http://localhost:20001",
        worker_addr="http://localhost:21010",
        model_names=["my_new_tool"]
    )
    
    # Run FastAPI server
    uvicorn.run(app, host="0.0.0.0", port=21010)
```

**Step 2: Train LLM to Use New Tool**

```json
// Training data
{
  "image": "sample.jpg",
  "conversations": [
    {
      "from": "human",
      "value": "<image>\nCan you analyze this with MyNewTool?"
    },
    {
      "from": "gpt",
      "thoughts": "I should use my_new_tool for this task",
      "actions": [{
        "API_name": "my_new_tool",
        "API_params": {"custom_param": "value"}
      }],
      "value": "Let me analyze this with my_new_tool..."
    },
    {
      "from": "human",
      "value": "my_new_tool outputs: {'result': '...', 'confidence': 0.95}\n\nAnswer my first question: ..."
    },
    {
      "from": "gpt",
      "thoughts": "Based on the tool output...",
      "actions": [],
      "value": "Based on my_new_tool analysis, ..."
    }
  ]
}
```

**Step 3: Add Visualization (Optional)**

```python
# In gradio_web_server_mmedagent.py, add:

if api_name == 'my_new_tool':
    # Custom visualization
    result = tool_response['result']
    annotated_image = my_custom_visualization(
        original_image, result
    )
    state.messages[-1][-1] = (
        state.messages[-1][-1],
        annotated_image,
        "Crop"
    )
```

### 5.2 Custom Tool Response Format

```python
def format_tool_response_for_llm(api_name, tool_response):
    """
    Custom formatting for different tools
    
    Args:
        api_name: Tool name
        tool_response: Raw tool output
    
    Returns:
        formatted_str: Formatted string for LLM
    """
    if api_name == 'grounding_dino':
        # Format detection results
        boxes = tool_response['boxes']
        phrases = tool_response['phrases']
        logits = tool_response['logits']
        
        formatted = f"Detected {len(boxes)} objects:\n"
        for i, (box, phrase, logit) in enumerate(zip(boxes, phrases, logits)):
            formatted += f"  {i+1}. {phrase} at {box} (confidence: {logit:.2%})\n"
        
        return formatted
        
    elif api_name == 'ChatCAD-G':
        # Format medical report
        report = tool_response['report']
        findings = tool_response.get('findings', [])
        
        formatted = f"Medical Report:\n{report}\n\n"
        if findings:
            formatted += "Key Findings:\n"
            for finding in findings:
                formatted += f"  - {finding}\n"
        
        return formatted
        
    else:
        # Default: JSON dump
        return json.dumps(tool_response, indent=2)
```

### 5.3 Multi-Tool Chaining

```python
def chain_tools(tool_sequence, state, prompt, controller_url):
    """
    Execute multiple tools in sequence
    
    Args:
        tool_sequence: List of tool configs
        state: Conversation state
        prompt: User prompt
        controller_url: Controller URL
    
    Returns:
        final_results: Combined results from all tools
    """
    results = {}
    
    for i, tool_cfg in enumerate(tool_sequence):
        api_name = tool_cfg['API_name']
        api_params = tool_cfg['API_params']
        
        # Use previous tool output if needed
        if i > 0 and 'use_previous_output' in api_params:
            prev_tool_name = tool_sequence[i-1]['API_name']
            api_params['previous_output'] = results[prev_tool_name]
        
        # Dispatch tool
        tool_response, _ = dispatch_tool_call(
            tool_cfg, state, prompt, controller_url
        )
        
        results[api_name] = tool_response
        
        print(f"✅ Tool {i+1}/{len(tool_sequence)} completed: {api_name}")
    
    return results

# Example: Detection → Segmentation → Classification
tool_sequence = [
    {
        "API_name": "grounding_dino",
        "API_params": {"prompts": ["abnormality"]}
    },
    {
        "API_name": "sam",
        "API_params": {"use_previous_output": "boxes"}
    },
    {
        "API_name": "biomedclip",
        "API_params": {"use_previous_output": "masks"}
    }
]

results = chain_tools(tool_sequence, state, prompt, controller_url)
```

---

## 6. Best Practices

### 6.1 Error Handling

```python
def robust_dispatch_tool_call(model_output_text, state, prompt, api_key, controller_url):
    """Dispatch with comprehensive error handling"""
    
    try:
        # Parse tool call
        tool_cfg = parse_tool_call_from_llm_output(model_output_text)
        
        if tool_cfg is None:
            logger.warning("No tool call found in LLM output")
            return None, None
        
        # Prepare parameters
        try:
            api_name, api_params = prepare_tool_parameters(
                tool_cfg, state, prompt, api_key
            )
        except Exception as e:
            logger.error(f"Failed to prepare parameters: {e}")
            return None, None
        
        # Find worker
        try:
            worker_addr = get_worker_addr(controller_url, api_name)
        except ValueError as e:
            logger.error(f"Worker not found: {e}")
            # Fallback: Ask LLM to try different tool
            return None, None
        
        # Call worker with timeout and retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tool_response = call_tool_worker(worker_addr, api_params)
                return tool_response, api_name
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout on attempt {attempt+1}/{max_retries}")
                if attempt == max_retries - 1:
                    raise
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                if attempt == max_retries - 1:
                    raise
    
    except Exception as e:
        logger.error(f"Tool dispatch failed: {e}", exc_info=True)
        return None, None
```

### 6.2 Performance Optimization

```python
# Cache tool responses
from functools import lru_cache
import hashlib

def get_cache_key(image, params):
    """Generate cache key for tool call"""
    # Hash image
    image_hash = hashlib.md5(image.tobytes()).hexdigest()
    # Hash params
    params_str = json.dumps(params, sort_keys=True)
    params_hash = hashlib.md5(params_str.encode()).hexdigest()
    return f"{image_hash}_{params_hash}"

# Cache decorator
def cached_tool_call(cache_size=100):
    cache = {}
    
    def decorator(func):
        def wrapper(api_name, api_params):
            cache_key = get_cache_key(api_params['image'], api_params)
            
            if cache_key in cache:
                logger.info(f"Cache hit for {api_name}")
                return cache[cache_key]
            
            result = func(api_name, api_params)
            
            if len(cache) >= cache_size:
                # Remove oldest
                cache.pop(next(iter(cache)))
            
            cache[cache_key] = result
            return result
        
        return wrapper
    return decorator

@cached_tool_call(cache_size=50)
def call_tool_worker_cached(api_name, api_params):
    return call_tool_worker(worker_addr, api_params)
```

### 6.3 Monitoring & Logging

```python
import time
from collections import defaultdict

class ToolMetrics:
    """Track tool usage metrics"""
    
    def __init__(self):
        self.call_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
    
    def record_call(self, api_name, response_time, success=True):
        self.call_counts[api_name] += 1
        self.response_times[api_name].append(response_time)
        if not success:
            self.error_counts[api_name] += 1
    
    def get_stats(self, api_name):
        times = self.response_times[api_name]
        return {
            'calls': self.call_counts[api_name],
            'errors': self.error_counts[api_name],
            'avg_time': sum(times) / len(times) if times else 0,
            'min_time': min(times) if times else 0,
            'max_time': max(times) if times else 0
        }

metrics = ToolMetrics()

def dispatch_with_metrics(api_name, api_params):
    start_time = time.time()
    success = False
    
    try:
        result = call_tool_worker(worker_addr, api_params)
        success = True
        return result
    finally:
        elapsed = time.time() - start_time
        metrics.record_call(api_name, elapsed, success)
        
        logger.info(f"Tool {api_name}: {elapsed:.2f}s (success={success})")
```

---

## 7. Debugging & Troubleshooting

### 7.1 Debug Mode

```python
DEBUG = True

def debug_print(stage, data):
    """Print debug information"""
    if DEBUG:
        print(f"\n{'='*60}")
        print(f"DEBUG [{stage}]")
        print(f"{'='*60}")
        if isinstance(data, dict):
            for key, value in data.items():
                print(f"  {key}: {value}")
        else:
            print(f"  {data}")
        print(f"{'='*60}\n")

# Usage in dispatching
def dispatch_tool_call_debug(model_output_text, state, prompt, api_key, controller_url):
    debug_print("LLM OUTPUT", model_output_text)
    
    tool_cfg = parse_tool_call_from_llm_output(model_output_text)
    debug_print("PARSED TOOL CONFIG", tool_cfg)
    
    api_name, api_params = prepare_tool_parameters(tool_cfg, state, prompt, api_key)
    debug_print("PREPARED PARAMS", {
        'api_name': api_name,
        'params_keys': list(api_params.keys())
    })
    
    worker_addr = get_worker_addr(controller_url, api_name)
    debug_print("WORKER ADDRESS", worker_addr)
    
    tool_response = call_tool_worker(worker_addr, api_params)
    debug_print("TOOL RESPONSE", tool_response)
    
    return tool_response, api_name
```

### 7.2 Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| **No tool pattern found** | LLM didn't generate tool call | Check prompt, retrain model |
| **Worker not found** | Tool worker not registered | Start worker, check controller |
| **Timeout** | Tool taking too long | Increase timeout, optimize tool |
| **Empty response** | Tool failed silently | Add error handling in worker |
| **Parse error** | Wrong JSON format | Handle both single/double quotes |

### 7.3 Testing

```python
import unittest

class TestToolOrchestration(unittest.TestCase):
    
    def test_parse_tool_call(self):
        """Test parsing tool calls from LLM output"""
        output = '''
        "thoughts🤔" Need detection
        "actions🚀" [{"API_name": "grounding_dino", "API_params": {"prompts": ["tumor"]}}]
        "value👉" Detecting...
        '''
        
        tool_cfg = parse_tool_call_from_llm_output(output)
        
        self.assertIsNotNone(tool_cfg)
        self.assertEqual(len(tool_cfg), 1)
        self.assertEqual(tool_cfg[0]['API_name'], 'grounding_dino')
        
    def test_clean_tool_response(self):
        """Test cleaning tool responses"""
        response = {
            'boxes': [[0.123456, 0.234567]],
            'logits': [0.951234],
            'size': [512, 512]
        }
        
        cleaned, extracted = clean_tool_response(response, 'grounding_dino')
        
        # Check rounding
        self.assertEqual(cleaned['boxes'][0][0], 0.1235)
        
        # Check size removed
        self.assertNotIn('size', cleaned)
    
    def test_visualization(self):
        """Test visualization creation"""
        from PIL import Image
        import numpy as np
        
        # Create dummy image
        img = Image.fromarray(np.zeros((512, 512, 3), dtype=np.uint8))
        
        response = {
            'boxes': [[0.2, 0.3, 0.5, 0.6]],
            'phrases': ['tumor'],
            'logits': [0.95]
        }
        
        annotated = plot_boxes(img, response)
        
        self.assertIsInstance(annotated, Image.Image)
        self.assertEqual(annotated.size, (512, 512))

if __name__ == '__main__':
    unittest.main()
```

---

## 📚 Tài Liệu Tham Khảo

### Papers
- [MMedAgent: Learning to Use Medical Tools with Multi-modal Agent](https://arxiv.org/abs/2407.02483)
- [ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/abs/2210.03629)
- [Toolformer: Language Models Can Teach Themselves to Use Tools](https://arxiv.org/abs/2302.04761)

### Code Files
- [`gradio_web_server_mmedagent.py`](llava/serve/gradio_web_server_mmedagent.py) - Main orchestration logic
- [`controller.py`](llava/serve/controller.py) - Worker management
- [`conversation.py`](llava/conversation.py) - Parsing utilities
- [`*_worker.py`](serve/) - Tool implementations

### Related
- [LLaVA-Plus](https://llava-vl.github.io/llava-plus/) - Tool-augmented visual agents
- [Gorilla](https://gorilla.cs.berkeley.edu/) - API-calling LLMs

---

*Tài liệu được tạo ngày 13/12/2025*
*Phiên bản: 1.0 - MMedAgent Agent Orchestration*
