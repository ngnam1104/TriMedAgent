"""
TriMed Orchestrator - Thin Client for Medical AI Pipeline
=========================================================

This module implements the Orchestrator Pattern for TriMedAgent.
It is a lightweight client that coordinates data flow between
multiple AI workers via HTTP API calls.

Architecture:
- LLaVA-Med Worker (21002): VQA, medical conversation
- Grounding DINO Worker (21003): Object detection
- MedSAM Worker (21004): Segmentation
- BiomedCLIP Worker (21006): Triage + Gatekeeper

Pipeline Flow: Perceive → Reason → Verify → Act

Author: TriMedAgent Team
"""

from __future__ import annotations

import base64
import json
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

# ==============================================================================
# Add llava to Python path for conversation templates
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from llava.conversation import conv_templates, Conversation

# ==============================================================================
# Logging Setup
# ==============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("TriMedOrchestrator")


# ==============================================================================
# Worker Configuration - Centralized URL Management
# ==============================================================================
WORKER_MAP: Dict[str, str] = {
    "llava": "http://localhost:21002/worker_generate_stream",
    "dino": "http://localhost:21003/worker_generate",
    "medsam": "http://localhost:21004/worker_generate",
    "biomedclip": "http://localhost:21006/worker_generate"
}

# Default configuration from serve/labels.json
DEFAULT_CONFIG: Dict[str, Any] = {
    "triage_labels": [
        "Chest X-ray", "Brain MRI", "Abdominal CT", "Histopathology",
        "Ultrasound", "Dermoscopy", "Gross pathology", "Bone X-ray",
        "Lung CT", "Retinal fundus", "Mammography"
    ],
    "gatekeeper_prompts": {
        "positive": "Pathological finding, lesion, tumor, abnormality",
        "negative": "Normal tissue, healthy anatomy, background noise, blurry area"
    },
    "thresholds": {
        "triage_confidence": 0.5,
        "gatekeeper_confidence": 0.6,
        "low_confidence_fallback": 0.3
    },
    "action_keywords": [
        "find", "detect", "locate", "segment", "where is", "identify",
        "show", "mark", "highlight", "outline", "circle", "point"
    ]
}


# ==============================================================================
# Data Classes for Pipeline Results
# ==============================================================================
@dataclass
class TriageResult:
    """Result from BiomedCLIP Triage stage."""
    modality: str
    confidence: float
    all_scores: Dict[str, float] = field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None


@dataclass
class DetectionResult:
    """Result from Grounding DINO detection."""
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]
    labels: List[str]
    scores: List[float]
    success: bool = True
    error: Optional[str] = None


@dataclass
class GatekeeperResult:
    """Result from BiomedCLIP Gatekeeper verification."""
    box: List[float]
    is_valid: bool
    pathology_score: float
    normal_score: float
    reason: str = ""


@dataclass
class SegmentationResult:
    """Result from MedSAM segmentation."""
    masks: List[Any]  # numpy arrays or base64 encoded
    boxes_used: List[List[float]]
    success: bool = True
    error: Optional[str] = None


@dataclass
class PipelineResult:
    """Complete pipeline execution result."""
    # Stage 1: Triage
    triage: Optional[TriageResult] = None
    
    # Stage 2: LLaVA Reasoning
    llava_response: str = ""
    context_injected: str = ""
    
    # Stage 3: Detection (DINO)
    dino_raw_boxes: List[List[float]] = field(default_factory=list)
    dino_labels: List[str] = field(default_factory=list)
    dino_scores: List[float] = field(default_factory=list)
    
    # Stage 4: Gatekeeper Verification
    verified_boxes: List[List[float]] = field(default_factory=list)
    rejected_boxes: List[Dict[str, Any]] = field(default_factory=list)
    gatekeeper_results: List[GatekeeperResult] = field(default_factory=list)
    
    # Stage 5: Segmentation (MedSAM)
    masks: List[Any] = field(default_factory=list)
    
    # Metadata
    execution_time: float = 0.0
    pipeline_complete: bool = False
    stopped_at_stage: str = ""
    errors: List[str] = field(default_factory=list)


# ==============================================================================
# Image Utilities
# ==============================================================================
def image_to_base64(image: Union[Image.Image, str, Path]) -> str:
    """
    Convert image to base64 string.
    
    Args:
        image: PIL Image, file path, or base64 string
        
    Returns:
        Base64 encoded image string
    """
    if isinstance(image, str):
        # Check if already base64
        if image.startswith("data:image"):
            return image.split(",", 1)[1]
        if len(image) > 500 and not Path(image).exists():
            return image  # Assume it's already base64
        # Load from file path
        image = Image.open(image)
    elif isinstance(image, Path):
        image = Image.open(image)
    
    # Convert PIL Image to base64
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


def crop_image_by_box(
    image: Union[Image.Image, str], 
    box: List[float],
    padding: int = 0
) -> Image.Image:
    """
    Crop image region defined by bounding box.
    
    Args:
        image: Source image
        box: [x1, y1, x2, y2] coordinates
        padding: Extra pixels around the box
        
    Returns:
        Cropped PIL Image
    """
    if isinstance(image, str):
        image = base64_to_image(image)
    
    x1, y1, x2, y2 = map(int, box)
    w, h = image.size
    
    # Apply padding with bounds check
    x1 = max(0, x1 - padding)
    y1 = max(0, y1 - padding)
    x2 = min(w, x2 + padding)
    y2 = min(h, y2 + padding)
    
    return image.crop((x1, y1, x2, y2))


# ==============================================================================
# TriMed Orchestrator - Main Class
# ==============================================================================
class TriMedOrchestrator:
    """
    Thin Client Orchestrator for TriMedAgent Pipeline.
    
    This class coordinates data flow between AI workers without
    loading any models locally. All inference happens on workers.
    
    Example:
        >>> orchestrator = TriMedOrchestrator()
        >>> result = orchestrator.run_full_chain(image, "Find the tumor")
        >>> print(result.verified_boxes)
    """
    
    def __init__(
        self,
        worker_urls: Optional[Dict[str, str]] = None,
        config: Optional[Dict[str, Any]] = None,
        timeout: int = 120,
        conv_template: str = "llava_v1"
    ):
        """
        Initialize Orchestrator.
        
        Args:
            worker_urls: Custom worker URL mapping
            config: Custom configuration (labels, thresholds, etc.)
            timeout: HTTP request timeout in seconds
            conv_template: Conversation template name for LLaVA
        """
        self.worker_urls = {**WORKER_MAP, **(worker_urls or {})}
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        self.timeout = timeout
        self.conv_template = conv_template
        
        # Load config from file if exists
        config_path = REPO_ROOT / "serve" / "labels.json"
        if config_path.exists():
            try:
                with open(config_path, "r") as f:
                    file_config = json.load(f)
                self.config.update(file_config)
                logger.info(f"Loaded config from {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        logger.info("TriMedOrchestrator initialized")
        logger.info(f"Workers: {list(self.worker_urls.keys())}")
    
    # ==========================================================================
    # Private API Wrappers - HTTP Communication
    # ==========================================================================
    
    def _call_api(
        self,
        worker_name: str,
        payload: Dict[str, Any],
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        Generic API caller with error handling.
        
        Args:
            worker_name: Key in worker_urls dict
            payload: Request payload
            stream: Whether to use streaming response
            
        Returns:
            Response data as dict
            
        Raises:
            ConnectionError: If worker is unreachable
            TimeoutError: If request times out
        """
        url = self.worker_urls.get(worker_name)
        if not url:
            raise ValueError(f"Unknown worker: {worker_name}")
        
        try:
            logger.debug(f"Calling {worker_name} at {url}")
            
            if stream:
                # Streaming response (for LLaVA)
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    stream=True,
                    timeout=self.timeout
                )
                response.raise_for_status()
                
                # Collect streamed text
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
                # Standard response
                response = requests.post(
                    url,
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout
                )
                response.raise_for_status()
                return response.json()
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Worker '{worker_name}' is not reachable at {url}"
            logger.error(error_msg)
            raise ConnectionError(error_msg) from e
        except requests.exceptions.Timeout as e:
            error_msg = f"Request to '{worker_name}' timed out after {self.timeout}s"
            logger.error(error_msg)
            raise TimeoutError(error_msg) from e
        except requests.exceptions.HTTPError as e:
            error_msg = f"HTTP error from '{worker_name}': {e.response.status_code}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _call_triage(self, image_b64: str) -> TriageResult:
        """
        Call BiomedCLIP for image modality classification (Triage).
        
        Args:
            image_b64: Base64 encoded image
            
        Returns:
            TriageResult with modality and confidence
        """
        try:
            payload = {
                "image": image_b64,
                "labels": self.config["triage_labels"]
            }
            
            response = self._call_api("biomedclip", payload)
            
            # Parse response - expect {"label": ..., "score": ..., "all_scores": {...}}
            if "label" in response:
                return TriageResult(
                    modality=response["label"],
                    confidence=response.get("score", response.get("confidence", 0.0)),
                    all_scores=response.get("all_scores", {}),
                    success=True
                )
            elif "predictions" in response:
                # Alternative format
                preds = response["predictions"]
                if preds:
                    best = max(preds, key=lambda x: x.get("score", 0))
                    return TriageResult(
                        modality=best["label"],
                        confidence=best["score"],
                        all_scores={p["label"]: p["score"] for p in preds},
                        success=True
                    )
            
            return TriageResult(
                modality="Unknown",
                confidence=0.0,
                success=False,
                error="Unexpected response format"
            )
            
        except Exception as e:
            logger.error(f"Triage failed: {e}")
            return TriageResult(
                modality="Unknown",
                confidence=0.0,
                success=False,
                error=str(e)
            )
    
    def _build_llava_prompt(
        self,
        user_query: str,
        system_context: str = ""
    ) -> str:
        """
        Build properly formatted prompt for LLaVA using conversation template.
        
        Args:
            user_query: User's question
            system_context: Optional system context from Triage
            
        Returns:
            Formatted prompt string
        """
        # Get conversation template
        conv = conv_templates.get(self.conv_template)
        if conv is None:
            conv = conv_templates.get("llava_v1")
        
        # Make a copy to avoid modifying the template
        conv = conv.copy()
        
        # Inject system context if provided
        if system_context:
            full_query = f"{system_context}\n\n{user_query}"
        else:
            full_query = user_query
        
        # Add image token and user message
        conv.append_message(conv.roles[0], f"<image>\n{full_query}")
        conv.append_message(conv.roles[1], None)
        
        return conv.get_prompt()
    
    def _call_llava(
        self,
        prompt: str,
        image_b64: str,
        temperature: float = 0.2,
        max_new_tokens: int = 512
    ) -> str:
        """
        Call LLaVA-Med for visual question answering.
        
        Args:
            prompt: Formatted prompt (use _build_llava_prompt)
            image_b64: Base64 encoded image
            temperature: Sampling temperature
            max_new_tokens: Maximum response length
            
        Returns:
            LLaVA's text response
        """
        try:
            payload = {
                "prompt": prompt,
                "images": [image_b64],
                "temperature": temperature,
                "max_new_tokens": max_new_tokens,
                "stop": "</s>"
            }
            
            response = self._call_api("llava", payload, stream=True)
            return response.get("text", "").strip()
            
        except Exception as e:
            logger.error(f"LLaVA call failed: {e}")
            raise
    
    def _call_dino(
        self,
        query: str,
        image_b64: str
    ) -> DetectionResult:
        """
        Call Grounding DINO for object detection.
        
        Args:
            query: Text query describing objects to find
            image_b64: Base64 encoded image
            
        Returns:
            DetectionResult with boxes, labels, scores
        """
        try:
            payload = {
                "image": image_b64,
                "prompt": query
            }
            
            response = self._call_api("dino", payload)
            
            # Parse response
            boxes = response.get("boxes", [])
            labels = response.get("labels", [query] * len(boxes))
            scores = response.get("scores", [1.0] * len(boxes))
            
            return DetectionResult(
                boxes=boxes,
                labels=labels,
                scores=scores,
                success=True
            )
            
        except Exception as e:
            logger.error(f"DINO detection failed: {e}")
            return DetectionResult(
                boxes=[],
                labels=[],
                scores=[],
                success=False,
                error=str(e)
            )
    
    def _call_gatekeeper(
        self,
        image_b64: str,
        box: List[float],
        target_label: str
    ) -> GatekeeperResult:
        """
        Call BiomedCLIP Gatekeeper to verify a detection box.
        
        The Gatekeeper crops the image region and classifies it
        as either pathological or normal tissue.
        
        Args:
            image_b64: Full image in base64
            box: [x1, y1, x2, y2] bounding box
            target_label: What we're looking for (e.g., "tumor")
            
        Returns:
            GatekeeperResult with validation status
        """
        try:
            # Crop the region from the image
            cropped_img = crop_image_by_box(image_b64, box, padding=5)
            cropped_b64 = image_to_base64(cropped_img)
            
            # Build gatekeeper labels
            positive_label = f"Pathology of {target_label}"
            negative_label = "Normal tissue"
            
            payload = {
                "image": cropped_b64,
                "labels": [positive_label, negative_label]
            }
            
            response = self._call_api("biomedclip", payload)
            
            # Parse scores
            all_scores = response.get("all_scores", {})
            pathology_score = all_scores.get(positive_label, 0.0)
            normal_score = all_scores.get(negative_label, 0.0)
            
            # Alternative parsing
            if not all_scores and "predictions" in response:
                for pred in response["predictions"]:
                    if positive_label in pred["label"]:
                        pathology_score = pred["score"]
                    elif negative_label in pred["label"]:
                        normal_score = pred["score"]
            
            # If response has direct label/score
            if not all_scores and "label" in response:
                if positive_label in response["label"]:
                    pathology_score = response.get("score", 0.0)
                    normal_score = 1 - pathology_score
                else:
                    normal_score = response.get("score", 0.0)
                    pathology_score = 1 - normal_score
            
            # Determine if valid
            threshold = self.config["thresholds"]["gatekeeper_confidence"]
            is_valid = pathology_score > threshold
            
            reason = (
                f"Pathology: {pathology_score:.2%}, Normal: {normal_score:.2%}, "
                f"Threshold: {threshold:.0%}"
            )
            
            return GatekeeperResult(
                box=box,
                is_valid=is_valid,
                pathology_score=pathology_score,
                normal_score=normal_score,
                reason=reason
            )
            
        except Exception as e:
            logger.error(f"Gatekeeper verification failed: {e}")
            return GatekeeperResult(
                box=box,
                is_valid=False,
                pathology_score=0.0,
                normal_score=0.0,
                reason=f"Error: {str(e)}"
            )
    
    def _call_medsam(
        self,
        image_b64: str,
        boxes: List[List[float]]
    ) -> SegmentationResult:
        """
        Call MedSAM for segmentation.
        
        Args:
            image_b64: Base64 encoded image
            boxes: List of bounding boxes [[x1, y1, x2, y2], ...]
            
        Returns:
            SegmentationResult with masks
        """
        try:
            if not boxes:
                return SegmentationResult(
                    masks=[],
                    boxes_used=[],
                    success=True
                )
            
            payload = {
                "image": image_b64,
                "boxes": boxes
            }
            
            response = self._call_api("medsam", payload)
            
            masks = response.get("masks", [])
            
            return SegmentationResult(
                masks=masks,
                boxes_used=boxes,
                success=True
            )
            
        except Exception as e:
            logger.error(f"MedSAM segmentation failed: {e}")
            return SegmentationResult(
                masks=[],
                boxes_used=boxes,
                success=False,
                error=str(e)
            )
    
    # ==========================================================================
    # Helper Methods
    # ==========================================================================
    
    def _extract_target_entity(self, query: str) -> str:
        """
        Extract the target entity from user query.
        
        Example: "Find the tumor in the brain" -> "tumor"
        """
        # Common patterns to extract nouns
        patterns = [
            r"find (?:the |a |an )?(\w+)",
            r"detect (?:the |a |an )?(\w+)",
            r"locate (?:the |a |an )?(\w+)",
            r"where is (?:the |a |an )?(\w+)",
            r"segment (?:the |a |an )?(\w+)",
            r"identify (?:the |a |an )?(\w+)",
            r"show (?:the |a |an )?(\w+)",
        ]
        
        query_lower = query.lower()
        
        for pattern in patterns:
            match = re.search(pattern, query_lower)
            if match:
                return match.group(1)
        
        # Fallback: extract last noun-like word
        words = query_lower.split()
        for word in reversed(words):
            if word not in ["the", "a", "an", "in", "on", "at", "of", "is", "are"]:
                return word
        
        return "abnormality"
    
    def _should_proceed_to_detection(self, llava_response: str, user_query: str) -> bool:
        """
        Determine if pipeline should proceed to detection stage.
        
        Returns True if user query or LLaVA response suggests
        visual localization is needed.
        """
        action_keywords = self.config.get("action_keywords", DEFAULT_CONFIG["action_keywords"])
        
        combined_text = (user_query + " " + llava_response).lower()
        
        for keyword in action_keywords:
            if keyword in combined_text:
                return True
        
        return False
    
    def _build_triage_context(self, triage_result: TriageResult) -> str:
        """Build system context from Triage result."""
        threshold = self.config["thresholds"]["triage_confidence"]
        low_threshold = self.config["thresholds"]["low_confidence_fallback"]
        
        if triage_result.confidence >= threshold:
            # Get recommended tools
            tool_mapping = self.config.get("conditional_execution", {}).get("tool_mapping", {})
            tools = tool_mapping.get(triage_result.modality, tool_mapping.get("default", []))
            tools_str = ", ".join(tools) if tools else "general analysis"
            
            return (
                f"[System Context: The input image is identified as {triage_result.modality} "
                f"with {triage_result.confidence:.0%} confidence. "
                f"Act as a specialist for this modality. Recommended tools: {tools_str}.]"
            )
        elif triage_result.confidence >= low_threshold:
            return (
                f"[System Context: Image modality unclear (best guess: {triage_result.modality}, "
                f"conf: {triage_result.confidence:.0%}). Proceed with general medical analysis.]"
            )
        else:
            return "[System Context: Unable to determine image modality. Proceed with caution.]"
    
    # ==========================================================================
    # Public Pipeline - Main Entry Point
    # ==========================================================================
    
    def run_full_chain(
        self,
        image: Union[Image.Image, str, Path],
        user_query: str,
        skip_gatekeeper: bool = False,
        skip_segmentation: bool = False
    ) -> PipelineResult:
        """
        Execute the full Perceive-Reason-Verify-Act pipeline.
        
        Pipeline Stages:
        1. Perceive (Triage): Classify image modality with BiomedCLIP
        2. Reason (LLaVA): Answer query with context injection
        3. Act (Detection): If action keywords detected, run Grounding DINO
        4. Verify (Gatekeeper): Filter detections with BiomedCLIP
        5. Act (Segmentation): Run MedSAM on verified boxes
        
        Args:
            image: Input image (PIL, path, or base64)
            user_query: User's question or command
            skip_gatekeeper: Skip Gatekeeper verification
            skip_segmentation: Skip MedSAM segmentation
            
        Returns:
            PipelineResult with all stage outputs
        """
        start_time = time.time()
        result = PipelineResult()
        
        logger.info(f"Starting pipeline for query: '{user_query}'")
        
        try:
            # Convert image to base64
            image_b64 = image_to_base64(image)
            logger.info("Image converted to base64")
            
            # =================================================================
            # Stage 1: Perceive (Triage)
            # =================================================================
            logger.info("=== Stage 1: Triage (BiomedCLIP) ===")
            result.triage = self._call_triage(image_b64)
            
            if not result.triage.success:
                result.errors.append(f"Triage failed: {result.triage.error}")
                logger.warning(f"Triage failed, continuing with default context")
            else:
                logger.info(
                    f"Triage: {result.triage.modality} "
                    f"(confidence: {result.triage.confidence:.2%})"
                )
            
            # Build context from Triage
            result.context_injected = self._build_triage_context(result.triage)
            
            # =================================================================
            # Stage 2: Reason (LLaVA)
            # =================================================================
            logger.info("=== Stage 2: Reasoning (LLaVA-Med) ===")
            
            try:
                prompt = self._build_llava_prompt(user_query, result.context_injected)
                result.llava_response = self._call_llava(prompt, image_b64)
                logger.info(f"LLaVA response received ({len(result.llava_response)} chars)")
            except Exception as e:
                result.errors.append(f"LLaVA failed: {str(e)}")
                result.llava_response = f"[LLaVA Error: {str(e)}]"
                logger.error(f"LLaVA failed: {e}")
            
            # =================================================================
            # Decision: Should we proceed to detection?
            # =================================================================
            if not self._should_proceed_to_detection(result.llava_response, user_query):
                logger.info("No action keywords detected. Pipeline complete at Stage 2.")
                result.stopped_at_stage = "reasoning"
                result.pipeline_complete = True
                result.execution_time = time.time() - start_time
                return result
            
            logger.info("Action keywords detected. Proceeding to detection...")
            
            # =================================================================
            # Stage 3: Act - Detection (Grounding DINO)
            # =================================================================
            logger.info("=== Stage 3: Detection (Grounding DINO) ===")
            
            target_entity = self._extract_target_entity(user_query)
            logger.info(f"Target entity extracted: '{target_entity}'")
            
            detection_result = self._call_dino(target_entity, image_b64)
            
            if not detection_result.success:
                result.errors.append(f"DINO failed: {detection_result.error}")
                result.stopped_at_stage = "detection"
                result.execution_time = time.time() - start_time
                return result
            
            result.dino_raw_boxes = detection_result.boxes
            result.dino_labels = detection_result.labels
            result.dino_scores = detection_result.scores
            
            logger.info(f"DINO found {len(result.dino_raw_boxes)} boxes")
            
            if not result.dino_raw_boxes:
                logger.info("No detections found. Pipeline complete at Stage 3.")
                result.stopped_at_stage = "detection"
                result.pipeline_complete = True
                result.execution_time = time.time() - start_time
                return result
            
            # =================================================================
            # Stage 4: Verify (Gatekeeper)
            # =================================================================
            if skip_gatekeeper:
                logger.info("Gatekeeper skipped (user request)")
                result.verified_boxes = result.dino_raw_boxes.copy()
            else:
                logger.info("=== Stage 4: Gatekeeper Verification (BiomedCLIP) ===")
                
                for i, box in enumerate(result.dino_raw_boxes):
                    logger.info(f"Verifying box {i+1}/{len(result.dino_raw_boxes)}")
                    
                    gk_result = self._call_gatekeeper(image_b64, box, target_entity)
                    result.gatekeeper_results.append(gk_result)
                    
                    if gk_result.is_valid:
                        result.verified_boxes.append(box)
                        logger.info(f"  ✓ Box {i+1} VERIFIED: {gk_result.reason}")
                    else:
                        result.rejected_boxes.append({
                            "box": box,
                            "reason": gk_result.reason,
                            "pathology_score": gk_result.pathology_score
                        })
                        logger.info(f"  ✗ Box {i+1} REJECTED: {gk_result.reason}")
                
                logger.info(
                    f"Gatekeeper: {len(result.verified_boxes)}/{len(result.dino_raw_boxes)} "
                    f"boxes verified"
                )
            
            if not result.verified_boxes:
                logger.info("No boxes passed Gatekeeper. Pipeline complete at Stage 4.")
                result.stopped_at_stage = "gatekeeper"
                result.pipeline_complete = True
                result.execution_time = time.time() - start_time
                return result
            
            # =================================================================
            # Stage 5: Act - Segmentation (MedSAM)
            # =================================================================
            if skip_segmentation:
                logger.info("Segmentation skipped (user request)")
            else:
                logger.info("=== Stage 5: Segmentation (MedSAM) ===")
                
                seg_result = self._call_medsam(image_b64, result.verified_boxes)
                
                if seg_result.success:
                    result.masks = seg_result.masks
                    logger.info(f"MedSAM generated {len(result.masks)} masks")
                else:
                    result.errors.append(f"MedSAM failed: {seg_result.error}")
                    logger.error(f"MedSAM failed: {seg_result.error}")
            
            # =================================================================
            # Pipeline Complete
            # =================================================================
            result.stopped_at_stage = "segmentation"
            result.pipeline_complete = True
            result.execution_time = time.time() - start_time
            
            logger.info(f"Pipeline completed in {result.execution_time:.2f}s")
            
            return result
            
        except Exception as e:
            logger.error(f"Pipeline failed with unexpected error: {e}")
            result.errors.append(f"Unexpected error: {str(e)}")
            result.execution_time = time.time() - start_time
            return result
    
    # ==========================================================================
    # Convenience Methods for Individual Stages
    # ==========================================================================
    
    def triage(self, image: Union[Image.Image, str, Path]) -> TriageResult:
        """Run only the Triage stage."""
        image_b64 = image_to_base64(image)
        return self._call_triage(image_b64)
    
    def ask_llava(
        self,
        image: Union[Image.Image, str, Path],
        question: str,
        system_context: str = ""
    ) -> str:
        """Ask LLaVA a question about an image."""
        image_b64 = image_to_base64(image)
        prompt = self._build_llava_prompt(question, system_context)
        return self._call_llava(prompt, image_b64)
    
    def detect(
        self,
        image: Union[Image.Image, str, Path],
        query: str
    ) -> DetectionResult:
        """Run object detection with Grounding DINO."""
        image_b64 = image_to_base64(image)
        return self._call_dino(query, image_b64)
    
    def segment(
        self,
        image: Union[Image.Image, str, Path],
        boxes: List[List[float]]
    ) -> SegmentationResult:
        """Run segmentation with MedSAM."""
        image_b64 = image_to_base64(image)
        return self._call_medsam(image_b64, boxes)
    
    def health_check(self) -> Dict[str, bool]:
        """Check connectivity to all workers."""
        status = {}
        
        for worker_name, url in self.worker_urls.items():
            try:
                # Simple ping - just check if URL is reachable
                base_url = url.rsplit("/", 1)[0]
                response = requests.get(base_url, timeout=5)
                status[worker_name] = response.status_code < 500
            except Exception:
                status[worker_name] = False
        
        return status


# ==============================================================================
# Module Entry Point (for testing)
# ==============================================================================
if __name__ == "__main__":
    # Simple test
    orchestrator = TriMedOrchestrator()
    
    print("TriMed Orchestrator initialized")
    print(f"Workers: {orchestrator.worker_urls}")
    print(f"Config loaded: {list(orchestrator.config.keys())}")
    
    # Health check
    print("\nWorker Health Check:")
    status = orchestrator.health_check()
    for worker, is_healthy in status.items():
        icon = "✓" if is_healthy else "✗"
        print(f"  {icon} {worker}: {'Online' if is_healthy else 'Offline'}")
