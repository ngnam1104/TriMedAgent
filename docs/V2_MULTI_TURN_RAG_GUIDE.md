# 🚀 TriMedAgent V2 - Hướng Dẫn Multi-turn Chat & RAG

> Tài liệu hướng dẫn sử dụng TriMedAgent V2 với Multi-turn Conversation Memory và RAG Integration

---

## 📋 Mục Lục

1. [Tổng Quan](#1-tổng-quan)
2. [Kiến Trúc V2](#2-kiến-trúc-v2)
3. [Multi-turn Memory](#3-multi-turn-memory)
4. [RAG Integration](#4-rag-integration)
5. [Hướng Dẫn Sử Dụng](#5-hướng-dẫn-sử-dụng)
6. [API Reference](#6-api-reference)

---

## 1. Tổng Quan

### 1.1 Tính Năng Mới trong V2

| Feature | V1 | V2 |
|---------|----|----|
| Single-turn Q&A | ✅ | ✅ |
| Multi-turn Memory | ❌ | ✅ |
| Conversation Summarization | ❌ | ✅ |
| RAG Knowledge Retrieval | ❌ | ✅ |
| Multi-image Sessions | ❌ | ✅ |
| Session Export/Import | ❌ | ✅ |
| Groq API Integration | ❌ | ✅ |

### 1.2 File Structure

```
src/
├── trimed_orchestrator.py       # V1 - Basic orchestrator
├── trimed_orchestrator_v2.py    # V2 - Multi-turn + RAG (NEW)
└── ChatCAD_R/
    ├── rag_engine_groq.py       # RAG with Groq API (NEW)
    ├── chat_bot_RAG.py          # Legacy RAG with OpenAI
    └── search_engine/           # Vector search components
```

---

## 2. Kiến Trúc V2

### 2.1 Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          TriMedAgent V2 Architecture                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  User Input (Text + Image)                                                  │
│       │                                                                     │
│       ▼                                                                     │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                    TriMedOrchestratorV2                              │   │
│  │  ┌──────────────────────────────────────────────────────────────┐   │   │
│  │  │  ConversationState                                           │   │   │
│  │  │  ├─ session_id                                               │   │   │
│  │  │  ├─ messages: List[ChatMessage]  ◄── Multi-turn Memory       │   │   │
│  │  │  ├─ images: Dict[hash, ImageSession]                         │   │   │
│  │  │  ├─ summary: str  ◄── Auto-summarization                     │   │   │
│  │  │  └─ turn_count: int                                          │   │   │
│  │  └──────────────────────────────────────────────────────────────┘   │   │
│  │                           │                                          │   │
│  │  ┌────────────────────────┼────────────────────────┐                │   │
│  │  │                        ▼                        │                │   │
│  │  │  Stage 1: Triage (BiomedCLIP)                  │                │   │
│  │  │       │                                         │                │   │
│  │  │       ▼                                         │                │   │
│  │  │  Stage 2: Reasoning (LLaVA-Med + Context)      │                │   │
│  │  │       │                                         │                │   │
│  │  │       ▼                                         │                │   │
│  │  │  Stage 2.5: RAG Enhancement (Groq)  ◄─── NEW   │                │   │
│  │  │       │                                         │                │   │
│  │  │       ▼                                         │                │   │
│  │  │  Stage 3-5: Detection → Gatekeeper → Segment   │                │   │
│  │  └─────────────────────────────────────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  Workers (HTTP):                                                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐      │
│  │ BiomedCLIP   │ │   LLaVA-Med  │ │ Grounding    │ │   MedSAM     │      │
│  │ Port: 21006  │ │ Port: 21002  │ │ DINO: 21003  │ │ Port: 21004  │      │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2.2 Data Flow

```
User Query → Orchestrator.chat()
    │
    ├─1─→ Check/Create Session
    │
    ├─2─→ Handle Image (set_image if new)
    │
    ├─3─→ Add message to history
    │
    ├─4─→ Triage (if first query for image)
    │
    ├─5─→ Build context from history
    │
    ├─6─→ LLaVA reasoning with context
    │
    ├─7─→ RAG enhancement (if triggered)
    │
    ├─8─→ Detection/Gatekeeper/Segmentation (if action query)
    │
    ├─9─→ Add response to history
    │
    └─10→ Maybe summarize (every N turns)
```

---

## 3. Multi-turn Memory

### 3.1 Core Data Classes

```python
@dataclass
class ChatMessage:
    role: str           # "user", "assistant", "system"
    content: str
    timestamp: str
    image_hash: str     # Link to associated image
    metadata: Dict

@dataclass
class ImageSession:
    image_hash: str
    image_b64: str
    triage_result: Dict
    detection_results: List
    verified_boxes: List
    masks: List

@dataclass
class ConversationState:
    session_id: str
    messages: List[ChatMessage]
    images: Dict[str, ImageSession]
    current_image_hash: str
    summary: str        # Summarized old history
    turn_count: int
```

### 3.2 Memory Management

#### Context Window Strategy

```python
# Configuration
config = {
    "memory": {
        "max_turns": 20,           # Max turns to keep
        "summarize_after": 10,     # Summarize every N turns
        "context_window_tokens": 2048  # Max tokens for context
    }
}
```

#### Auto-summarization

Khi `turn_count % summarize_after == 0`:

1. Collect messages từ đầu đến `messages[-4]`
2. Gọi LLaVA để summarize
3. Lưu summary vào `state.summary`
4. Context mới = `summary + recent_4_messages`

```python
def _maybe_summarize(self):
    if self.state.turn_count % 10 == 0:
        self.state.summary = self._summarize_conversation()
```

### 3.3 Context Building

```python
def _build_conversation_context(self) -> str:
    parts = []
    
    # Add summary if exists
    if self.state.summary:
        parts.append(f"[Previous Summary]: {self.state.summary}")
    
    # Add recent conversation
    recent = self.state.get_recent_context(n_turns=5)
    parts.append(f"[Recent Conversation]:\n{recent}")
    
    return "\n\n".join(parts)
```

---

## 4. RAG Integration

### 4.1 Groq API Setup

```python
# Get API key from https://console.groq.com/keys
os.environ["GROQ_API_KEY"] = "your-api-key"

# Or in code
rag = MedicalRAG(api_key="your-api-key")
```

### 4.2 MedicalRAG Class

```python
from src.ChatCAD_R.rag_engine_groq import MedicalRAG, create_medical_rag

# Create instance
rag = create_medical_rag(
    api_key="your-key",
    model_name="llama3-70b-8192"  # or "mixtral-8x7b-32768"
)

# Load knowledge base
rag.load_knowledge_base("path/to/medical_kb.pkl")

# Query
response = rag.query("What is the treatment for pneumonia?")
print(response.answer)
print(response.retrieved_contexts)
```

### 4.3 RAG Trigger Keywords

RAG được kích hoạt tự động khi query chứa:

```python
RAG_KEYWORDS = [
    "treatment", "therapy", "prognosis",
    "cause", "symptom", "diagnose",
    "differential", "protocol", "guideline",
    "medication", "drug",
    # Vietnamese
    "phác đồ", "điều trị", "triệu chứng",
    "nguyên nhân", "tiên lượng"
]
```

### 4.4 Vector Search

```python
# Using KDTree (from ChatCAD_R)
from src.ChatCAD_R.rag_engine_groq import KDTreeVectorStore

store = KDTreeVectorStore()
store.load("medical_kb.pkl")
results = store.search(query_embedding, top_k=5)

# Using simple fallback
from src.ChatCAD_R.rag_engine_groq import SimpleVectorStore

store = SimpleVectorStore()
store.add(text, embedding, metadata)
results = store.search(query_embedding)
```

---

## 5. Hướng Dẫn Sử Dụng

### 5.1 Basic Usage

```python
from src.trimed_orchestrator_v2 import create_orchestrator
from src.ChatCAD_R.rag_engine_groq import create_medical_rag

# Create RAG engine
rag = create_medical_rag(api_key="your-groq-key")
rag.load_knowledge_base("medical_kb.pkl")

# Create orchestrator with RAG
orchestrator = create_orchestrator(rag_engine=rag)

# Multi-turn conversation
response1, _ = orchestrator.chat("What do you see?", image=my_image)
response2, _ = orchestrator.chat("Is it malignant?")  # Remembers context
response3, _ = orchestrator.chat("What's the treatment?")  # Triggers RAG
```

### 5.2 Session Management

```python
# Export session for persistence
session_data = orchestrator.export_session()
save_json(session_data, "session_backup.json")

# Load session later
orchestrator.load_session(load_json("session_backup.json"))

# Clear session
orchestrator.clear_session()
```

### 5.3 Gradio Interface

```python
import gradio as gr

def process_chat(message, history, image):
    response, result = orchestrator.chat(message, image)
    history.append([message, response])
    return history, draw_boxes(image, result.boxes)

demo = gr.Blocks()
with demo:
    chatbot = gr.Chatbot()
    image = gr.Image(type="pil")
    textbox = gr.Textbox()
    
    textbox.submit(process_chat, [textbox, chatbot, image], [chatbot])

demo.launch()
```

### 5.4 Demo Notebook

Xem `demo_trimedagent_rag.ipynb` để biết ví dụ đầy đủ với:
- Multi-turn conversation demo
- RAG testing
- Simulated mode (không cần workers)
- Gradio interface

---

## 6. API Reference

### 6.1 TriMedOrchestratorV2

```python
class TriMedOrchestratorV2:
    def __init__(
        self,
        worker_urls: Dict[str, str] = None,
        config: Dict[str, Any] = None,
        timeout: int = 120,
        rag_engine: MedicalRAG = None
    ): ...
    
    # Session Management
    def start_session(self, session_id: str = None) -> str: ...
    def load_session(self, state_dict: dict): ...
    def export_session(self) -> dict: ...
    def clear_session(self): ...
    def set_image(self, image) -> str: ...
    
    # Main Interface
    def chat(
        self,
        user_input: str,
        image = None,
        skip_gatekeeper: bool = False,
        skip_segmentation: bool = False,
        force_rag: bool = False
    ) -> Tuple[str, PipelineResult]: ...
    
    # Utilities
    def get_chat_history(self) -> List[List[str]]: ...
    def health_check(self) -> Dict[str, bool]: ...
```

### 6.2 MedicalRAG

```python
class MedicalRAG:
    def __init__(
        self,
        config: RAGConfig = None,
        groq_api_key: str = None
    ): ...
    
    # Knowledge Base
    def load_knowledge_base(self, path: str): ...
    def build_knowledge_base(
        self,
        texts: List[str],
        metadata: List[Dict] = None,
        save_path: str = None
    ): ...
    
    # Retrieval & Generation
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]: ...
    def generate(self, prompt: str, stream: bool = False) -> str: ...
    def query(self, question: str, top_k: int = 5) -> RAGResponse: ...
    
    # Integration helper
    def retrieve_and_generate(self, query: str) -> str: ...
    def enhance_visual_findings(
        self,
        visual_findings: str,
        user_question: str
    ) -> RAGResponse: ...
```

### 6.3 PipelineResult

```python
@dataclass
class PipelineResult:
    triage: TriageResult = None
    llava_response: str = ""
    context_injected: str = ""
    dino_raw_boxes: List[List[float]] = []
    dino_labels: List[str] = []
    dino_scores: List[float] = []
    verified_boxes: List[List[float]] = []
    rejected_boxes: List[Dict] = []
    gatekeeper_results: List[GatekeeperResult] = []
    masks: List = []
    rag_context: str = ""         # RAG-retrieved knowledge
    execution_time: float = 0.0
    pipeline_complete: bool = False
    errors: List[str] = []
```

---

## 📝 Notes

### Performance Tips

1. **Summarize thường xuyên**: Set `summarize_after` nhỏ (5-10) để tránh context quá dài
2. **Cache triage**: Triage result được cache per-image, không cần gọi lại
3. **Groq models**: `llama3-70b-8192` cho accuracy, `mixtral-8x7b-32768` cho speed

### Known Limitations

1. RAG knowledge base cần được pre-built
2. Summarization cần LLaVA worker
3. Multi-image tracking đơn giản (hash-based)

### Troubleshooting

```python
# Check worker status
print(orchestrator.health_check())

# Check session state
print(orchestrator.state.turn_count)
print(len(orchestrator.state.messages))

# Force RAG
response, _ = orchestrator.chat("Query", force_rag=True)
```

---

*Cập nhật: 2024*
