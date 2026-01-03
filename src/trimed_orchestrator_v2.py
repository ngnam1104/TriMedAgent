"""
TriMed Orchestrator V2 - Multi-turn Chatbot with Memory & RAG Integration
=========================================================================

Enhanced version of the Orchestrator with:
- Multi-turn conversation memory
- Conversation summarization for context management
- RAG integration points for medical knowledge retrieval
- Session management for multiple images

Author: TriMedAgent Team
Version: 2.0
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sys
import time
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
from collections import deque

import requests
from PIL import Image

# ==============================================================================
# Add llava to Python path
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

try:
    from llava.conversation import conv_templates, Conversation
except ImportError:
    conv_templates = {}
    Conversation = None

# ==============================================================================
# Logging Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TriMedOrchestrator.V2")


# ==============================================================================
# Configuration
# ==============================================================================
WORKER_MAP: Dict[str, str] = {
    "llava": "http://localhost:21002/worker_generate_stream",
    "dino": "http://localhost:21003/worker_generate",
    "medsam": "http://localhost:21004/worker_generate",
    "biomedclip": "http://localhost:21006/worker_generate"
}

DEFAULT_CONFIG: Dict[str, Any] = {
    "triage_labels": [
        "Chest X-ray", "Brain MRI", "Abdominal CT", "Histopathology",
        "Ultrasound", "Dermoscopy", "Gross pathology", "Bone X-ray",
        "Lung CT", "Retinal fundus", "Mammography"
    ],
    "gatekeeper_prompts": {
        "positive": "Pathological finding, lesion, tumor, abnormality",
        "negative": "Normal tissue, healthy anatomy, background noise"
    },
    "thresholds": {
        "triage_confidence": 0.5,
        "gatekeeper_confidence": 0.6,
        "low_confidence_fallback": 0.3
    },
    "action_keywords": [
        "find", "detect", "locate", "segment", "where is", "identify",
        "show", "mark", "highlight", "outline", "circle", "point"
    ],
    "rag_trigger_keywords": [
        "treatment", "therapy", "prognosis", "cause", "symptom", "diagnose",
        "differential", "protocol", "guideline", "medication", "drug",
        "phác đồ", "điều trị", "triệu chứng", "nguyên nhân", "tiên lượng"
    ],
    "memory": {
        "max_turns": 20,
        "summarize_after": 10,
        "context_window_tokens": 2048
    }
}


# ==============================================================================
# Data Classes
# ==============================================================================
@dataclass
class ChatMessage:
    """Single message in conversation."""
    role: str  # "user", "assistant", "system"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    image_hash: Optional[str] = None  # Hash of associated image
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ImageSession:
    """Session state for a single image."""
    image_hash: str
    image_b64: str
    triage_result: Optional[Dict] = None
    detection_results: List[Dict] = field(default_factory=list)
    segmentation_masks: List[Any] = field(default_factory=list)
    verified_boxes: List[List[float]] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass 
class ConversationState:
    """Full conversation state with memory."""
    session_id: str
    messages: List[ChatMessage] = field(default_factory=list)
    images: Dict[str, ImageSession] = field(default_factory=dict)  # hash -> session
    current_image_hash: Optional[str] = None
    summary: str = ""  # Summarized history for context management
    turn_count: int = 0
    
    def add_message(self, role: str, content: str, image_hash: str = None, **metadata):
        """Add a message to history."""
        msg = ChatMessage(
            role=role,
            content=content,
            image_hash=image_hash or self.current_image_hash,
            metadata=metadata
        )
        self.messages.append(msg)
        if role == "user":
            self.turn_count += 1
    
    def get_recent_context(self, n_turns: int = 5) -> str:
        """Get recent conversation context for prompt building."""
        recent = self.messages[-n_turns * 2:] if len(self.messages) > n_turns * 2 else self.messages
        
        context_lines = []
        for msg in recent:
            prefix = "User" if msg.role == "user" else "Assistant"
            context_lines.append(f"{prefix}: {msg.content[:500]}")
        
        return "\n".join(context_lines)
    
    def export_history(self) -> List[dict]:
        """Export history for Gradio chatbot format."""
        history = []
        for i in range(0, len(self.messages) - 1, 2):
            user_msg = self.messages[i] if i < len(self.messages) else None
            asst_msg = self.messages[i + 1] if i + 1 < len(self.messages) else None
            
            if user_msg and user_msg.role == "user":
                user_text = user_msg.content
                asst_text = asst_msg.content if asst_msg else ""
                history.append([user_text, asst_text])
        
        return history


@dataclass
class TriageResult:
    modality: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class DetectionResult:
    boxes: List[List[float]]
    labels: List[str]
    scores: List[float]
    success: bool = True
    error: Optional[str] = None


@dataclass
class GatekeeperResult:
    box: List[float]
    is_valid: bool
    pathology_score: float
    normal_score: float
    reason: str = ""


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    triage: Optional[TriageResult] = None
    llava_response: str = ""
    context_injected: str = ""
    dino_raw_boxes: List[List[float]] = field(default_factory=list)
    dino_labels: List[str] = field(default_factory=list)
    dino_scores: List[float] = field(default_factory=list)
    verified_boxes: List[List[float]] = field(default_factory=list)
    rejected_boxes: List[Dict] = field(default_factory=list)
    gatekeeper_results: List[GatekeeperResult] = field(default_factory=list)
    masks: List[Any] = field(default_factory=list)
    rag_context: str = ""  # RAG-retrieved knowledge
    execution_time: float = 0.0
    pipeline_complete: bool = False
    stopped_at_stage: str = ""
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# Utility Functions
# ==============================================================================
def image_to_base64(image: Union[Image.Image, str, Path]) -> str:
    """Convert image to base64 string."""
    if isinstance(image, str):
        if image.startswith("data:image"):
            return image.split(",", 1)[1]
        if len(image) > 500 and not Path(image).exists():
            return image
        image = Image.open(image)
    elif isinstance(image, Path):
        image = Image.open(image)
    
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        img_format = "PNG" if image.mode == "RGBA" else "JPEG"
        image.save(buffered, format=img_format)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    
    raise ValueError(f"Unsupported image type: {type(image)}")


def base64_to_image(b64_string: str) -> Image.Image:
    """Convert base64 string to PIL Image."""
    if b64_string.startswith("data:image"):
        b64_string = b64_string.split(",", 1)[1]
    img_bytes = base64.b64decode(b64_string)
    return Image.open(BytesIO(img_bytes))


def compute_image_hash(image: Union[Image.Image, str, bytes]) -> str:
    """Compute hash of image for session tracking."""
    if isinstance(image, Image.Image):
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        img_bytes = buffered.getvalue()
    elif isinstance(image, str):
        img_bytes = base64.b64decode(image) if len(image) > 500 else open(image, 'rb').read()
    else:
        img_bytes = image
    
    return hashlib.md5(img_bytes).hexdigest()[:12]


def crop_image_by_box(image: Union[Image.Image, str], box: List[float], padding: int = 0) -> Image.Image:
    """Crop image region defined by bounding box."""
    if isinstance(image, str):
        image = base64_to_image(image)
    
    x1, y1, x2, y2 = map(int, box)
    w, h = image.size
    
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image.crop((x1, y1, x2, y2))


# ==============================================================================
# Conversation Summarization Prompts
# ==============================================================================
SUMMARY_PROMPT = """Summarize the following medical conversation into a brief context paragraph.
Focus on: patient history, key findings, detected abnormalities, and any diagnoses mentioned.
Keep the summary under 200 words.

Conversation:
{conversation}

Summary:"""

MULTI_TURN_SYSTEM_PROMPT = """You are a medical AI assistant analyzing medical images.
You have access to the conversation history and can refer to previous findings.

Previous Context Summary:
{summary}

Current Image Analysis:
{triage_context}

Recent Conversation:
{recent_history}

Please respond to the user's query while maintaining context from the conversation."""


# ==============================================================================
# TriMed Orchestrator V2 - Main Class
# ==============================================================================
class TriMedOrchestratorV2:
    """
    Enhanced Orchestrator with Multi-turn Conversation Memory and RAG Integration.
    
    Features:
    - Multi-turn conversation history
    - Automatic context summarization
    - Multiple image session management
    - RAG integration hooks
    - State export/import for persistence
    """
    
    def __init__(
        self,
        worker_urls: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        conv_template: str = "llava_v1",
        rag_engine: Optional[Any] = None  # RAG engine instance (MedicalRAG)
    ):
        self.worker_urls = {**WORKER_MAP, **(worker_urls or {})}
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.timeout = timeout
        self.conv_template = conv_template
        self.rag_engine = rag_engine
        
        # Conversation state
        self.state: Optional[ConversationState] = None
        
        # Load config from file
        self._load_config_file()
        
        logger.info("TriMedOrchestrator V2 initialized with multi-turn memory")
    
    def _load_config_file(self):
        """Load configuration from labels.json if exists."""
        config_path = REPO_ROOT / "serve" / "labels.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
    
    # ==========================================================================
    # Session Management
    # ==========================================================================
    
    def start_session(self, session_id: str = None) -> str:
        """Start a new conversation session."""
        session_id = session_id or f"session_{int(time.time())}"
        self.state = ConversationState(session_id=session_id)
        logger.info(f"Started new session: {session_id}")
        return session_id
    
    def load_session(self, state_dict: dict):
        """Load session from exported state."""
        self.state = ConversationState(**state_dict)
        logger.info(f"Loaded session: {self.state.session_id}")
    
    def export_session(self) -> dict:
        """Export current session state for persistence."""
        if not self.state:
            return {}
        return asdict(self.state)
    
    def clear_session(self):
        """Clear current session."""
        if self.state:
            session_id = self.state.session_id
            self.state = ConversationState(session_id=session_id)
            logger.info(f"Cleared session: {session_id}")
    
    def set_image(self, image: Union[Image.Image, str, Path]) -> str:
        """
        Set the current image for analysis.
        Returns image hash for reference.
        """
        if not self.state:
            self.start_session()
        
        image_b64 = image_to_base64(image)
        image_hash = compute_image_hash(image_b64)
        
        # Create or get image session
        if image_hash not in self.state.images:
            self.state.images[image_hash] = ImageSession(
                image_hash=image_hash,
                image_b64=image_b64
            )
            logger.info(f"New image registered: {image_hash}")
        
        self.state.current_image_hash = image_hash
        return image_hash
    
    def get_current_image(self) -> Optional[ImageSession]:
        """Get current image session."""
        if self.state and self.state.current_image_hash:
            return self.state.images.get(self.state.current_image_hash)
        return None
    
    # ==========================================================================
    # API Wrappers (Same as V1)
    # ==========================================================================
    
    def _call_api(self, worker_name: str, payload: Dict, stream: bool = False) -> Dict:
        """Generic API caller with error handling."""
        url = self.worker_urls.get(worker_name)
        if not url:
            raise ValueError(f"Unknown worker: {worker_name}")
        
        try:
            if stream:
                response = requests.post(url, json=payload, stream=True, timeout=self.timeout)
                response.raise_for_status()
                
                full_text = ""
                for chunk in response.iter_lines(decode_unicode=True):
                    if chunk:
                        try:
                            data = json.loads(chunk.replace("data: ", ""))
                            if "text" in data:
                                full_text = data["text"]
                        except json.JSONDecodeError:
                            full_text += chunk
                
                return {"text": full_text, "success": True}
            else:
                response = requests.post(url, json=payload, timeout=self.timeout)
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(f"Worker '{worker_name}' unreachable") from e
        except requests.exceptions.Timeout as e:
            raise TimeoutError(f"Request to '{worker_name}' timed out") from e
    
    def _call_triage(self, image_b64: str) -> TriageResult:
        """Call BiomedCLIP for Triage."""
        try:
            payload = {"image": image_b64, "labels": self.config["triage_labels"]}
            response = self._call_api("biomedclip", payload)
            
            if "label" in response:
                return TriageResult(
                    modality=response["label"],
                    confidence=response.get("score", 0.0),
                    all_scores=response.get("all_scores", {}),
                    success=True
                )
            return TriageResult(modality="Unknown", confidence=0.0, success=False)
        except Exception as e:
            return TriageResult(modality="Unknown", confidence=0.0, success=False, error=str(e))
    
    def _call_dino(self, query: str, image_b64: str) -> DetectionResult:
        """Call Grounding DINO for detection."""
        try:
            payload = {"image": image_b64, "prompt": query}
            response = self._call_api("dino", payload)
            
            return DetectionResult(
                boxes=response.get("boxes", []),
                labels=response.get("labels", []),
                scores=response.get("scores", []),
                success=True
            )
        except Exception as e:
            return DetectionResult(boxes=[], labels=[], scores=[], success=False, error=str(e))
    
    def _call_gatekeeper(self, image_b64: str, box: List[float], target: str) -> GatekeeperResult:
        """Call BiomedCLIP Gatekeeper."""
        try:
            cropped_img = crop_image_by_box(image_b64, box, padding=5)
            cropped_b64 = image_to_base64(cropped_img)
            
            pos_label = f"Pathology of {target}"
            neg_label = "Normal tissue"
            
            payload = {"image": cropped_b64, "labels": [pos_label, neg_label]}
            response = self._call_api("biomedclip", payload)
            
            all_scores = response.get("all_scores", {})
            path_score = all_scores.get(pos_label, 0.0)
            norm_score = all_scores.get(neg_label, 0.0)
            
            threshold = self.config["thresholds"]["gatekeeper_confidence"]
            is_valid = path_score > threshold
            
            return GatekeeperResult(
                box=box, is_valid=is_valid,
                pathology_score=path_score, normal_score=norm_score,
                reason=f"Path: {path_score:.1%}, Normal: {norm_score:.1%}"
            )
        except Exception as e:
            return GatekeeperResult(box=box, is_valid=False, pathology_score=0.0, normal_score=0.0, reason=str(e))
    
    def _call_medsam(self, image_b64: str, boxes: List[List[float]]) -> List[Any]:
        """Call MedSAM for segmentation."""
        try:
            if not boxes:
                return []
            payload = {"image": image_b64, "boxes": boxes}
            response = self._call_api("medsam", payload)
            return response.get("masks", [])
        except Exception as e:
            logger.error(f"MedSAM failed: {e}")
            return []
    
    # ==========================================================================
    # Context Building with Memory
    # ==========================================================================
    
    def _build_conversation_context(self) -> str:
        """Build context from conversation history."""
        if not self.state:
            return ""
        
        parts = []
        
        # Add summary if exists
        if self.state.summary:
            parts.append(f"[Previous Summary]: {self.state.summary}")
        
        # Add recent conversation
        recent = self.state.get_recent_context(n_turns=5)
        if recent:
            parts.append(f"[Recent Conversation]:\n{recent}")
        
        return "\n\n".join(parts)
    
    def _build_triage_context(self, triage: TriageResult) -> str:
        """Build system context from Triage."""
        if not triage.success:
            return "[System: Unable to identify image modality.]"
        
        threshold = self.config["thresholds"]["triage_confidence"]
        
        if triage.confidence >= threshold:
            return f"[System: Image identified as {triage.modality} ({triage.confidence:.0%} confidence)]"
        else:
            return f"[System: Image modality unclear (best guess: {triage.modality}, {triage.confidence:.0%})]"
    
    def _build_llava_prompt_with_history(self, user_query: str, triage_context: str) -> str:
        """Build LLaVA prompt with conversation history."""
        conv_context = self._build_conversation_context()
        
        # Combine contexts
        full_context = f"{conv_context}\n\n{triage_context}" if conv_context else triage_context
        
        # Get conversation template
        conv = conv_templates.get(self.conv_template)
        if conv is None:
            conv = conv_templates.get("llava_v1", None)
        
        if conv:
            conv = conv.copy()
            conv.append_message(conv.roles[0], f"<image>\n{full_context}\n\nUser: {user_query}")
            conv.append_message(conv.roles[1], None)
            return conv.get_prompt()
        else:
            # Fallback format
            return f"USER: <image>\n{full_context}\n\n{user_query}\nASSISTANT:"
    
    def _call_llava_with_context(self, query: str, image_b64: str, triage_context: str) -> str:
        """Call LLaVA with conversation context."""
        prompt = self._build_llava_prompt_with_history(query, triage_context)
        
        payload = {
            "prompt": prompt,
            "images": [image_b64],
            "temperature": 0.2,
            "max_new_tokens": 512,
            "stop": "</s>"
        }
        
        response = self._call_api("llava", payload, stream=True)
        return response.get("text", "").strip()
    
    # ==========================================================================
    # RAG Integration
    # ==========================================================================
    
    def _should_trigger_rag(self, query: str, llava_response: str = "") -> bool:
        """Check if RAG should be triggered."""
        if not self.rag_engine:
            return False
        
        combined = (query + " " + llava_response).lower()
        rag_keywords = self.config.get("rag_trigger_keywords", [])
        
        return any(kw in combined for kw in rag_keywords)
    
    def _enhance_with_rag(self, query: str, context: str = "") -> str:
        """Enhance response with RAG knowledge."""
        if not self.rag_engine:
            return ""
        
        try:
            # Build RAG query
            rag_query = f"{context}\n\nUser query: {query}"
            
            # Retrieve and generate
            knowledge = self.rag_engine.retrieve_and_generate(rag_query)
            return knowledge
        except Exception as e:
            logger.error(f"RAG enhancement failed: {e}")
            return ""
    
    # ==========================================================================
    # Conversation Summarization
    # ==========================================================================
    
    def _summarize_conversation(self) -> str:
        """Summarize conversation history to manage context length."""
        if not self.state or len(self.state.messages) < 6:
            return ""
        
        # Build conversation text
        conv_text = ""
        for msg in self.state.messages[:-4]:  # Keep last 4 messages out of summary
            role = "User" if msg.role == "user" else "Assistant"
            conv_text += f"{role}: {msg.content[:300]}\n"
        
        # Use LLaVA to summarize if available, otherwise simple truncation
        try:
            summary_prompt = SUMMARY_PROMPT.format(conversation=conv_text)
            payload = {
                "prompt": summary_prompt,
                "images": [],
                "temperature": 0.3,
                "max_new_tokens": 256
            }
            response = self._call_api("llava", payload, stream=True)
            summary = response.get("text", "")
            return summary.strip()
        except Exception as e:
            logger.warning(f"Summarization failed: {e}")
            # Fallback: simple truncation
            return conv_text[:500] + "..."
    
    def _maybe_summarize(self):
        """Summarize if conversation is getting long."""
        if not self.state:
            return
        
        summarize_after = self.config.get("memory", {}).get("summarize_after", 10)
        
        if self.state.turn_count > 0 and self.state.turn_count % summarize_after == 0:
            logger.info("Summarizing conversation history...")
            self.state.summary = self._summarize_conversation()
            logger.info(f"Summary updated: {len(self.state.summary)} chars")
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _extract_target_entity(self, query: str) -> str:
        """Extract target entity from user query."""
        patterns = [
            r"find (?:the |a |an )?(\w+)",
            r"detect (?:the |a |an )?(\w+)",
            r"locate (?:the |a |an )?(\w+)",
            r"segment (?:the |a |an )?(\w+)",
        ]
        
        query_lower = query.lower()
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        return "abnormality"
    
    def _should_detect(self, query: str) -> bool:
        """Check if detection should be triggered."""
        keywords = self.config.get("action_keywords", [])
        return any(kw in query.lower() for kw in keywords)
    
    # ==========================================================================
    # Main Chat Interface
    # ==========================================================================
    
    def chat(
        self,
        user_input: str,
        image: Optional[Union[Image.Image, str, Path]] = None,
        skip_gatekeeper: bool = False,
        skip_segmentation: bool = False,
        force_rag: bool = False
    ) -> Tuple[str, PipelineResult]:
        """
        Main chat interface with multi-turn memory.
        
        Args:
            user_input: User's message
            image: Optional new image (uses current if None)
            skip_gatekeeper: Skip verification step
            skip_segmentation: Skip segmentation step
            force_rag: Force RAG enhancement
            
        Returns:
            Tuple of (response_text, PipelineResult)
        """
        start_time = time.time()
        
        # Ensure session exists
        if not self.state:
            self.start_session()
        
        # Handle image
        if image is not None:
            self.set_image(image)
        
        img_session = self.get_current_image()
        if not img_session:
            return "Please upload an image first.", PipelineResult()
        
        result = PipelineResult()
        
        # Add user message to history
        self.state.add_message("user", user_input)
        
        try:
            # =================================================================
            # Stage 1: Triage (if not done for this image)
            # =================================================================
            if not img_session.triage_result:
                logger.info("=== Stage 1: Triage ===")
                triage = self._call_triage(img_session.image_b64)
                img_session.triage_result = asdict(triage)
                result.triage = triage
            else:
                result.triage = TriageResult(**img_session.triage_result)
            
            triage_context = self._build_triage_context(result.triage)
            result.context_injected = triage_context
            
            # =================================================================
            # Stage 2: Reasoning with Context
            # =================================================================
            logger.info("=== Stage 2: Reasoning (LLaVA) ===")
            
            try:
                llava_response = self._call_llava_with_context(
                    user_input, img_session.image_b64, triage_context
                )
                result.llava_response = llava_response
            except Exception as e:
                result.llava_response = f"[Error: {str(e)}]"
                result.errors.append(str(e))
            
            # =================================================================
            # Stage 2.5: RAG Enhancement (if triggered)
            # =================================================================
            if force_rag or self._should_trigger_rag(user_input, result.llava_response):
                logger.info("=== Stage 2.5: RAG Enhancement ===")
                rag_knowledge = self._enhance_with_rag(user_input, result.llava_response)
                if rag_knowledge:
                    result.rag_context = rag_knowledge
                    result.llava_response += f"\n\n📚 **Medical Reference**:\n{rag_knowledge}"
            
            # =================================================================
            # Stage 3-5: Detection, Gatekeeper, Segmentation (if needed)
            # =================================================================
            if self._should_detect(user_input):
                logger.info("=== Stage 3: Detection ===")
                target = self._extract_target_entity(user_input)
                
                detection = self._call_dino(target, img_session.image_b64)
                result.dino_raw_boxes = detection.boxes
                result.dino_labels = detection.labels
                result.dino_scores = detection.scores
                
                if detection.boxes:
                    # Gatekeeper
                    if not skip_gatekeeper:
                        logger.info("=== Stage 4: Gatekeeper ===")
                        for box in detection.boxes:
                            gk = self._call_gatekeeper(img_session.image_b64, box, target)
                            result.gatekeeper_results.append(gk)
                            if gk.is_valid:
                                result.verified_boxes.append(box)
                            else:
                                result.rejected_boxes.append({"box": box, "reason": gk.reason})
                    else:
                        result.verified_boxes = detection.boxes
                    
                    # Segmentation
                    if result.verified_boxes and not skip_segmentation:
                        logger.info("=== Stage 5: Segmentation ===")
                        result.masks = self._call_medsam(img_session.image_b64, result.verified_boxes)
                    
                    # Update image session
                    img_session.verified_boxes = result.verified_boxes
                    img_session.segmentation_masks = result.masks
                    
                    # Append detection summary to response
                    det_summary = f"\n\n🔍 **Detection Results**:\n"
                    det_summary += f"- Found {len(result.dino_raw_boxes)} regions\n"
                    det_summary += f"- Verified: {len(result.verified_boxes)} regions\n"
                    if result.masks:
                        det_summary += f"- Segmented: {len(result.masks)} masks"
                    
                    result.llava_response += det_summary
            
            result.pipeline_complete = True
            result.stopped_at_stage = "complete"
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            result.errors.append(str(e))
            result.llava_response = f"Sorry, an error occurred: {str(e)}"
        
        result.execution_time = time.time() - start_time
        
        # Add assistant response to history
        self.state.add_message("assistant", result.llava_response)
        
        # Maybe summarize
        self._maybe_summarize()
        
        return result.llava_response, result
    
    # ==========================================================================
    # Convenience Methods
    # ==========================================================================
    
    def triage(self, image: Union[Image.Image, str, Path]) -> TriageResult:
        """Run only Triage stage."""
        image_b64 = image_to_base64(image)
        return self._call_triage(image_b64)
    
    def detect(self, image: Union[Image.Image, str, Path], query: str) -> DetectionResult:
        """Run only Detection stage."""
        image_b64 = image_to_base64(image)
        return self._call_dino(query, image_b64)
    
    def get_chat_history(self) -> List[List[str]]:
        """Get chat history in Gradio format."""
        if not self.state:
            return []
        return self.state.export_history()
    
    def health_check(self) -> Dict[str, bool]:
        """Check worker connectivity."""
        status = {}
        for name, url in self.worker_urls.items():
            try:
                base_url = url.rsplit("/", 1)[0]
                response = requests.get(base_url, timeout=5)
                status[name] = response.status_code < 500
            except:
                status[name] = False
        return status


# ==============================================================================
# Factory Function for Easy Initialization
# ==============================================================================
def create_orchestrator(
    rag_engine=None,
    session_id: str = None,
    **kwargs
) -> TriMedOrchestratorV2:
    """
    Factory function to create and initialize orchestrator.
    
    Args:
        rag_engine: Optional RAG engine instance
        session_id: Optional session ID
        **kwargs: Additional config options
        
    Returns:
        Initialized TriMedOrchestratorV2
    """
    orchestrator = TriMedOrchestratorV2(rag_engine=rag_engine, **kwargs)
    orchestrator.start_session(session_id)
    return orchestrator


# ==============================================================================
# Test
# ==============================================================================
if __name__ == "__main__":
    print("TriMed Orchestrator V2 - Multi-turn Chatbot")
    print("=" * 50)
    
    orchestrator = create_orchestrator(session_id="test_session")
    print(f"Session: {orchestrator.state.session_id}")
    print(f"Workers: {list(orchestrator.worker_urls.keys())}")
    
    # Health check
    print("\nWorker Status:")
    for worker, status in orchestrator.health_check().items():
        print(f"  {'✓' if status else '✗'} {worker}")
