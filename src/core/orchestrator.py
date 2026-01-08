"""
TriMed-Agent Orchestrator V2
============================

Core logic implementing the Hybrid ReAct loop with:
1. Logic Router - Intent classification (Theory vs Diagnosis)
2. RAG Module - For theory questions  
3. Planner (Brain) - Generates structured JSON plans
4. Vision Engine - Executes tools (BiomedCLIP, DINO, MedSAM)
5. Verification - Self-correction loop
6. Synthesis - Final report generation

Architecture:
    Query → Logic Router → {Theory → RAG, Diagnosis → Planner}
    Planner → Action → Observation → (loop) → Answer
"""

import logging
import time
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from PIL import Image
import traceback

from ..types.models import PipelineResult, StrategicPlan, Strategy
from ..modules.brain.llava_brain import LLaVABrain
from ..modules.vision.dino_detector import GroundingDINO
from ..modules.vision.zoom_processor import ZoomManager
from ..modules.segmentation.medsam_wrapper import MedSAMWrapper
from ..utils.math_utils import get_box_area_ratio
from ..utils.visualization import draw_boxes_on_image, draw_masks_on_image

# Import new V2 modules
from .logic_router import LogicRouter, Intent
from ..modules.knowledge.rag_engine import MedicalRAG

logger = logging.getLogger(__name__)


# =============================================================================
# State Machine for ReAct Loop
# =============================================================================

@dataclass
class AgentState:
    """Tracks agent state during ReAct execution"""
    step: int = 0
    max_steps: int = 5
    
    # ReAct trajectory
    thoughts: List[str] = field(default_factory=list)
    actions: List[str] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    
    # Results
    detections: List[Dict[str, Any]] = field(default_factory=list)
    verified_boxes: List[List[float]] = field(default_factory=list)
    masks: List[Any] = field(default_factory=list)
    
    # Flags
    should_terminate: bool = False
    used_fallback: bool = False
    

# =============================================================================
# Main Orchestrator
# =============================================================================

class TriMedOrchestrator:
    """
    TriMed-Agent V2 Orchestrator
    
    Implements full pipeline:
    1. Logic Router → Decide RAG or Planner path
    2. For Theory → RAG direct answer
    3. For Diagnosis → ReAct loop (Plan → Act → Observe → Verify)
    4. Segmentation if boxes found
    5. Synthesis and report
    
    Example:
        config = {"device": {"llava_device": "cuda:0"}, ...}
        agent = TriMedOrchestrator(config)
        result = agent.process(image, "Tim có to không?")
    """
    
    # Configuration defaults
    DEFAULT_CONFIDENCE_THRESHOLD = 0.25
    DEFAULT_MAX_BOX_RATIO = 0.30
    DEFAULT_MAX_STEPS = 5
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Initialize Logic Router (lightweight, always on)
        self.router = LogicRouter(
            device=config.get('device', {}).get('router_device', 'cpu'),
            diagnosis_threshold=config.get('router', {}).get('threshold', 0.7)
        )
        
        # Initialize RAG Module (lazy load)
        self._rag = None
        
        # Initialize Brain (Planner)
        self.brain = LLaVABrain(
            device=config['device']['llava_device'],
            quantize_4bit=config.get('models', {}).get('llava', {}).get('quantize', True)
        )
        
        # Initialize Vision Tools
        self.vision = GroundingDINO(
            device=config['device']['tool_device'],
            config_path=config['models']['grounding_dino']['config_path'],
            checkpoint_path=config['models']['grounding_dino']['weights_path']
        )
        self.zoomer = ZoomManager(
            zoom_factor=config.get('vision', {}).get('zoom_factor', 2.0)
        )
        
        # Initialize Segmentation
        self.segmenter = MedSAMWrapper(
            device=config['device']['tool_device'],
            checkpoint_path=config['models']['medsam']['checkpoint']
        )
        
        logger.info("✓ TriMedOrchestrator V2 initialized")
        
    @property
    def rag(self) -> MedicalRAG:
        """Lazy load RAG module"""
        if self._rag is None:
            self._rag = MedicalRAG()
        return self._rag
        
    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------
    
    def process(
        self,
        image: Optional[Image.Image],
        query: str,
        force_segmentation: bool = False,
        max_steps: int = None
    ) -> PipelineResult:
        """
        Main processing pipeline with Logic Router.
        
        Args:
            image: Input medical image (can be None for theory questions)
            query: User's question
            force_segmentation: Always run segmentation
            max_steps: Override max ReAct steps
            
        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()
        result = PipelineResult()
        
        logger.info(f"Processing: {query[:50]}...")
        
        try:
            # ---------------------------------------------------------
            # Step 0: ROUTE (Logic Router)
            # ---------------------------------------------------------
            routing = self.router.route(query, has_image=(image is not None))
            result.steps_executed.append(f"route: {routing['intent']}")
            
            logger.info(f"Routing decision: {routing['reasoning']}")
            
            # ---------------------------------------------------------
            # Path A: THEORY → RAG
            # ---------------------------------------------------------
            if routing['use_rag'] and not routing['use_planner']:
                logger.info("Executing RAG path for theory question...")
                result = self._handle_theory_query(query, result)
                
            # ---------------------------------------------------------
            # Path B: DIAGNOSIS → ReAct Loop
            # ---------------------------------------------------------
            elif routing['use_planner'] and image is not None:
                logger.info("Executing ReAct path for diagnosis...")
                max_steps = max_steps or self.DEFAULT_MAX_STEPS
                result = self._handle_diagnosis_query(
                    image, query, result, 
                    force_segmentation, max_steps
                )
                
            # ---------------------------------------------------------
            # Path C: HYBRID → RAG + ReAct
            # ---------------------------------------------------------
            elif routing['use_rag'] and routing['use_planner'] and image is not None:
                logger.info("Executing Hybrid path...")
                # Get context from RAG first
                rag_response = self.rag.query(query, top_k=2)
                context = "\n".join([r.text for r in rag_response.retrieved_contexts])
                
                # Pass context to ReAct
                result = self._handle_diagnosis_query(
                    image, query, result,
                    force_segmentation, max_steps or self.DEFAULT_MAX_STEPS,
                    rag_context=context
                )
                
            # ---------------------------------------------------------
            # Path D: No image but diagnosis intent → Error
            # ---------------------------------------------------------
            else:
                result.final_report = "Cần cung cấp ảnh để thực hiện chẩn đoán."
                result.errors.append("No image provided for diagnosis query")
                
        except Exception as e:
            logger.error(f"Pipeline error: {traceback.format_exc()}")
            result.errors.append(str(e))
            result.final_report = f"Đã xảy ra lỗi: {str(e)}"
            
        result.execution_time = time.time() - start_time
        result.success = len(result.errors) == 0
        
        return result
    
    # -------------------------------------------------------------------------
    # Theory Query Handler (RAG Path)
    # -------------------------------------------------------------------------
    
    def _handle_theory_query(
        self,
        query: str,
        result: PipelineResult
    ) -> PipelineResult:
        """Handle theory questions using RAG"""
        result.steps_executed.append("rag_retrieve")
        
        rag_response = self.rag.query(query)
        
        result.final_report = rag_response.answer
        result.steps_executed.append("rag_generate")
        
        # Store retrieved context for reference
        if rag_response.retrieved_contexts:
            sources = [r.source for r in rag_response.retrieved_contexts]
            result.steps_executed.append(f"sources: {sources}")
            
        return result
    
    # -------------------------------------------------------------------------
    # Diagnosis Query Handler (ReAct Path)
    # -------------------------------------------------------------------------
    
    def _handle_diagnosis_query(
        self,
        image: Image.Image,
        query: str,
        result: PipelineResult,
        force_segmentation: bool,
        max_steps: int,
        rag_context: Optional[str] = None
    ) -> PipelineResult:
        """
        Handle diagnosis questions using ReAct loop.
        
        ReAct Flow:
            1. Thought: Brain reasons about the query
            2. Action: Select and execute a tool
            3. Observation: Get tool output
            4. Repeat until done or max_steps
        """
        state = AgentState(max_steps=max_steps)
        
        # ---------------------------------------------------------
        # 1. PLAN (First Thought)
        # ---------------------------------------------------------
        result.steps_executed.append("plan")
        
        # Add RAG context if available
        augmented_query = query
        if rag_context:
            augmented_query = f"Context: {rag_context}\n\nQuery: {query}"
        
        plan_dict = self.brain.plan(image, augmented_query)
        
        target = plan_dict.get("target", "abnormality")
        action = plan_dict.get("action", "detect")
        thought = plan_dict.get("thought", "Analyzing image...")
        
        state.thoughts.append(thought)
        state.actions.append(f"{action}({target})")
        
        # Map to internal strategy
        strategy_enum = Strategy.ZOOM_IN if action == "zoom" else Strategy.GLOBAL_SCAN
        
        result.strategic_plan = StrategicPlan(
            target_object=target,
            strategy=strategy_enum,
            reasoning=thought
        )
        
        # ---------------------------------------------------------
        # 2. ACT (Execute Vision Tools)
        # ---------------------------------------------------------
        state.detections = self._execute_detection(
            image, target, strategy_enum, state, result
        )
        
        # ---------------------------------------------------------
        # 3. VERIFY (Self-Correction)
        # ---------------------------------------------------------
        result.steps_executed.append("verify")
        state.verified_boxes, scores = self._verify_detections(
            state.detections, image.size
        )
        
        state.observations.append(f"Found {len(state.verified_boxes)} verified boxes")
        
        # ---------------------------------------------------------
        # 4. FALLBACK (If needed)
        # ---------------------------------------------------------
        if len(state.verified_boxes) == 0 and strategy_enum == Strategy.ZOOM_IN:
            result.steps_executed.append("fallback_global")
            logger.warning("Zoom failed. Triggering Global Fallback...")
            
            fallback_dets = self.vision.detect(image, target)
            state.verified_boxes, scores = self._verify_detections(
                fallback_dets, image.size
            )
            state.used_fallback = True
            state.observations.append(f"Fallback found {len(state.verified_boxes)} boxes")
        
        result.verified_boxes = state.verified_boxes
        result.detection_scores = scores
        
        # ---------------------------------------------------------
        # 5. SEGMENT (MedSAM)
        # ---------------------------------------------------------
        if state.verified_boxes and (force_segmentation or True):
            result.steps_executed.append("segment")
            try:
                state.masks = self.segmenter.segment(image, state.verified_boxes)
                result.masks = state.masks
            except Exception as e:
                logger.error(f"Segmentation error: {e}")
                result.errors.append(f"Segmentation failed: {str(e)}")
        
        # ---------------------------------------------------------
        # 6. SYNTHESIZE (Generate Report)
        # ---------------------------------------------------------
        result.steps_executed.append("synthesize")
        
        # Draw annotations
        annotated = image.copy()
        if state.verified_boxes:
            annotated = draw_boxes_on_image(annotated, state.verified_boxes)
        if state.masks:
            annotated = draw_masks_on_image(annotated, state.masks)
        result.annotated_image = annotated
        
        # Generate final report
        result.final_report = self._generate_report(
            image, target, state, rag_context
        )
        
        return result
    
    # -------------------------------------------------------------------------
    # Helper: Execute Detection
    # -------------------------------------------------------------------------
    
    def _execute_detection(
        self,
        image: Image.Image,
        target: str,
        strategy: Strategy,
        state: AgentState,
        result: PipelineResult
    ) -> List[Dict[str, Any]]:
        """Execute detection based on strategy"""
        detections = []
        
        if strategy == Strategy.ZOOM_IN:
            result.steps_executed.append("zoom_search")
            logger.info("Executing Smart Zoom Strategy...")
            
            crops = self.zoomer.generate_crops(image)
            
            for crop_img, crop_box in crops:
                local_dets = self.vision.detect(crop_img, target)
                global_mapped = self.zoomer.process_zoom_results(
                    local_dets, crop_box, image.size
                )
                detections.extend(global_mapped)
                
        else:
            result.steps_executed.append("global_scan")
            logger.info("Executing Global Scan Strategy...")
            detections = self.vision.detect(image, target)
            
        state.observations.append(f"Raw detections: {len(detections)}")
        return detections
    
    # -------------------------------------------------------------------------
    # Helper: Verify Detections
    # -------------------------------------------------------------------------
    
    def _verify_detections(
        self,
        detections: List[Dict[str, Any]],
        image_size: Tuple[int, int]
    ) -> Tuple[List[List[float]], List[float]]:
        """
        Verify and filter detections.
        
        Rules:
        1. Confidence > threshold
        2. Box area < max_ratio of image
        3. Non-zero area
        """
        valid_boxes = []
        valid_scores = []
        
        for det in detections:
            box = det.get('box', det.get('bbox', []))
            score = det.get('score', det.get('confidence', 0))
            
            # Rule 1: Confidence
            if score < self.DEFAULT_CONFIDENCE_THRESHOLD:
                continue
                
            # Rule 2: Size ratio
            ratio = get_box_area_ratio(box, image_size)
            if ratio > self.DEFAULT_MAX_BOX_RATIO:
                logger.debug(f"Rejected large box: ratio={ratio:.2f}")
                continue
                
            # Rule 3: Non-zero area
            if len(box) >= 4:
                if (box[2] - box[0]) <= 0 or (box[3] - box[1]) <= 0:
                    continue
                    
            valid_boxes.append(box)
            valid_scores.append(score)
            
        return valid_boxes, valid_scores
    
    # -------------------------------------------------------------------------
    # Helper: Generate Report
    # -------------------------------------------------------------------------
    
    def _generate_report(
        self,
        image: Image.Image,
        target: str,
        state: AgentState,
        rag_context: Optional[str] = None
    ) -> str:
        """Generate final diagnostic report"""
        if state.verified_boxes:
            num_boxes = len(state.verified_boxes)
            prompt = f"""Based on my analysis:
- Found {num_boxes} instances of "{target}"
- Strategy: {'Zoom + Fallback' if state.used_fallback else 'Standard'}
- Reasoning: {state.thoughts[0] if state.thoughts else 'N/A'}

{'Context: ' + rag_context if rag_context else ''}

Please provide a concise diagnostic summary."""
            
            return self.brain.query(image, prompt)
        else:
            return f"Không phát hiện {target} với độ tin cậy cao. Có thể cần kiểm tra lại với các phương pháp khác."
    
    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------
    
    def get_available_tools(self) -> List[str]:
        """List available tools"""
        return [
            "GroundingDINO - Object detection",
            "MedSAM - Medical image segmentation",
            "BiomedCLIP - Image-text similarity",
            "MedicalRAG - Knowledge retrieval",
        ]
        
    def get_system_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            "router": "active",
            "brain": "loaded" if self.brain else "not loaded",
            "vision": "loaded" if self.vision else "not loaded",
            "rag": "loaded" if self._rag else "lazy",
            "config": self.config.get('device', {})
        }
