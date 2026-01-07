"""
Hybrid ReAct Orchestrator
=========================

SOTA Medical AI Orchestrator implementing:
- Planning Phase (LLaVA as Brain/Planner)
- Coarse-to-Fine Zooming (solve "box too large" problem)
- Self-Correction / Verification Loop
- ReAct-style reasoning loop
"""

from __future__ import annotations

import logging
import time
from typing import List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from ..types.models import (
    Strategy, TargetSize, StrategicPlan, 
    DetectionResult, AgentState, PipelineResult
)
from ..utils.image import base64_to_image
from ..utils.visualization import draw_boxes_on_image, draw_masks_on_image
from .planner import StrategicPlanner
from .zoomer import ImageZoomer
from .validator import DetectionValidator

logger = logging.getLogger(__name__)


class HybridReActOrchestrator:
    """
    SOTA Medical AI Orchestrator with Hybrid ReAct Architecture.
    
    Implements Plan → Zoom/Act → Verify loop for robust detection
    of small pathologies like lung nodules.
    
    Example:
        orchestrator = HybridReActOrchestrator()
        orchestrator.load_tools()
        result = orchestrator.process(image, "Find any nodules in the right lung")
    """
    
    def __init__(
        self,
        device: str = "cuda",
        llava_device: str = "cuda:0",
        tool_device: str = "cuda:0",
        max_iterations: int = 3,
        enable_verification: bool = True,
        enable_rag: bool = True
    ):
        """
        Initialize orchestrator.
        
        Args:
            device: Default device
            llava_device: Device for LLaVA
            tool_device: Device for other tools
            max_iterations: Max ReAct loop iterations
            enable_verification: Enable LLaVA verification
            enable_rag: Enable RAG for knowledge queries
        """
        self.device = device
        self.llava_device = llava_device
        self.tool_device = tool_device
        self.max_iterations = max_iterations
        self.enable_verification = enable_verification
        self.enable_rag = enable_rag
        
        # Tools
        self.biomedclip = None
        self.grounding_dino = None
        self.medsam = None
        self.llava = None
        self.rag = None
        
        # Modules
        self.planner = None
        self.zoomer = ImageZoomer()
        self.validator = None
        
        logger.info("HybridReActOrchestrator initialized")
    
    def load_tools(
        self,
        load_biomedclip: bool = True,
        load_dino: bool = True,
        load_medsam: bool = True,
        load_llava: bool = True,
        load_rag: bool = True,
        llava_quantize: bool = True
    ) -> None:
        """Load all tools and initialize modules."""
        from ..tools import (
            BiomedCLIPTool, GroundingDINOTool, MedSAMTool, LLaVATool, MedicalRAG
        )
        
        if load_biomedclip:
            logger.info("Loading BiomedCLIP...")
            self.biomedclip = BiomedCLIPTool(device=self.tool_device)
        
        if load_dino:
            logger.info("Loading Grounding DINO...")
            self.grounding_dino = GroundingDINOTool(device=self.tool_device)
        
        if load_medsam:
            logger.info("Loading MedSAM...")
            self.medsam = MedSAMTool(device=self.tool_device)
        
        if load_llava:
            logger.info("Loading LLaVA...")
            self.llava = LLaVATool(
                device=self.llava_device,
                quantize_4bit=llava_quantize
            )
            self.planner = StrategicPlanner(self.llava)
            self.validator = DetectionValidator(
                self.llava if self.enable_verification else None
            )
        
        if load_rag and self.enable_rag:
            logger.info("Loading RAG...")
            self.rag = MedicalRAG()
        
        logger.info("✓ All tools loaded")
    
    def process(
        self,
        image: Union[str, Image.Image],
        query: str,
        force_segmentation: bool = False
    ) -> PipelineResult:
        """
        Process image with Hybrid ReAct loop.
        
        Args:
            image: PIL Image or base64 string
            query: User query
            force_segmentation: Force segmentation even if not needed
            
        Returns:
            PipelineResult with all outputs
        """
        start_time = time.time()
        result = PipelineResult()
        
        # Convert image
        if isinstance(image, str):
            pil_image = base64_to_image(image)
        else:
            pil_image = image
        
        if pil_image.mode != "RGB":
            pil_image = pil_image.convert("RGB")
        
        original_size = pil_image.size
        
        try:
            # =========== STEP 1: TRIAGE ===========
            result.steps_executed.append("triage")
            if self.biomedclip:
                triage = self.biomedclip.triage(pil_image)
                result.triage_modality = triage["modality"]
                result.triage_confidence = triage["modality_confidence"]
            
            # =========== STEP 2: PLANNING ===========
            result.steps_executed.append("planning")
            if self.planner:
                plan = self.planner.create_plan(
                    pil_image, query, result.triage_modality
                )
            else:
                plan = StrategicPlan()
            result.strategic_plan = plan
            
            # =========== STEP 3: ReAct LOOP ===========
            state = AgentState(max_iterations=self.max_iterations, plan=plan)
            
            while state.iteration < state.max_iterations and not state.success:
                state.iteration += 1
                result.steps_executed.append(f"react_iter_{state.iteration}")
                
                # THINK
                thought = self._think(state, pil_image)
                state.thought_history.append(thought)
                
                # ACT
                detection = self._act(state, pil_image, original_size)
                state.detections.append(detection)
                
                # OBSERVE (Validate)
                if detection.boxes:
                    valid_boxes, valid_scores, rejections = self.validator.validate_boxes(
                        detection.boxes,
                        detection.scores,
                        original_size,
                        plan.target_size,
                        plan.confidence_threshold
                    )
                    
                    if valid_boxes:
                        # Semantic verification
                        if self.enable_verification and self.llava:
                            verifications = self.validator.semantic_verify(
                                pil_image, valid_boxes, plan.target_object
                            )
                            
                            # Filter by verification
                            final_boxes = []
                            final_scores = []
                            for box, score, ver in zip(valid_boxes, valid_scores, verifications):
                                if ver.is_valid or ver.confidence in ["high", "medium"]:
                                    final_boxes.append(box)
                                    final_scores.append(score)
                            
                            state.verified_boxes = final_boxes
                            result.detection_scores = final_scores
                        else:
                            state.verified_boxes = valid_boxes
                            result.detection_scores = valid_scores
                        
                        if state.verified_boxes:
                            state.success = True
                            state.action_history.append("accept")
                    else:
                        state.action_history.append("reject_size")
                else:
                    state.action_history.append("no_detection")
                
                # Decide next action if not successful
                if not state.success:
                    action = self.validator.suggest_action(
                        bool(detection.boxes),
                        bool(valid_boxes) if detection.boxes else False,
                        plan.target_size,
                        state.iteration
                    )
                    state.action_history.append(action)
                    
                    if action == "give_up":
                        break
                    elif action == "retry_with_zoom":
                        if plan.fallback_regions:
                            plan.anatomical_location = plan.fallback_regions.pop(0)
                            plan.strategy = Strategy.ZOOM_IN
                    elif action == "retry_stricter":
                        plan.confidence_threshold *= 1.5
            
            result.agent_iterations = state.iteration
            result.verified_boxes = state.verified_boxes
            result.raw_boxes = [b for d in state.detections for b in d.boxes]
            
            # =========== STEP 4: SEGMENTATION ===========
            if self.medsam and (state.verified_boxes or force_segmentation):
                result.steps_executed.append("segmentation")
                if state.verified_boxes:
                    result.masks = self.medsam.segment(
                        pil_image,
                        boxes=state.verified_boxes
                    )
            
            # =========== STEP 5: SYNTHESIS ===========
            result.steps_executed.append("synthesis")
            if self.llava:
                result.llava_analysis = self._synthesize_analysis(
                    pil_image, query, result
                )
                result.final_report = self._generate_report(pil_image, result)
            
            # =========== STEP 6: RAG ===========
            if self.rag and self.enable_rag and self._needs_rag(query):
                result.steps_executed.append("rag")
                rag_result = self.rag.query(query)
                result.rag_response = rag_result.answer
            
            # =========== STEP 7: VISUALIZATION ===========
            if state.verified_boxes or result.masks:
                annotated = draw_boxes_on_image(pil_image, state.verified_boxes)
                if result.masks:
                    annotated = draw_masks_on_image(annotated, result.masks)
                result.annotated_image = annotated
            
            result.success = True
            
        except Exception as e:
            logger.error(f"Pipeline error: {e}")
            result.errors.append(str(e))
            result.success = False
        
        result.execution_time = time.time() - start_time
        return result
    
    def _think(self, state: AgentState, image: Image.Image) -> str:
        """Generate thought for current iteration."""
        plan = state.plan
        
        if state.iteration == 1:
            thought = (
                f"Starting detection for '{plan.target_object}' "
                f"(size: {plan.target_size.value}, strategy: {plan.strategy.value})"
            )
            if plan.anatomical_location:
                thought += f" in region '{plan.anatomical_location}'"
        else:
            last_action = state.action_history[-1] if state.action_history else "none"
            thought = f"Iteration {state.iteration}: Previous action was '{last_action}'. "
            
            if last_action == "reject_size":
                thought += "Boxes were too large. Trying with zoom or stricter threshold."
            elif last_action == "no_detection":
                thought += "No detections found. Trying different region or threshold."
        
        logger.info(f"THINK: {thought}")
        return thought
    
    def _act(
        self,
        state: AgentState,
        image: Image.Image,
        original_size: Tuple[int, int]
    ) -> DetectionResult:
        """Execute detection action based on strategy."""
        plan = state.plan
        result = DetectionResult()
        
        if not self.grounding_dino:
            result.is_valid = False
            result.rejection_reason = "DINO not loaded"
            return result
        
        if plan.strategy == Strategy.ZOOM_IN and plan.anatomical_location:
            # Crop to region first
            cropped, crop_region = self.zoomer.crop_to_region(
                image, plan.anatomical_location
            )
            result.source_region = plan.anatomical_location
            result.crop_offset = (crop_region[0], crop_region[1])
            
            logger.info(f"ACT: Zooming to '{plan.anatomical_location}', "
                       f"crop size: {cropped.size}")
            
            # Detect on cropped image
            detection = self.grounding_dino.detect(
                cropped,
                plan.detection_prompt,
                box_threshold=plan.confidence_threshold,
                return_phrases=True
            )
            
            # Map boxes back to original coordinates
            if detection.get("boxes"):
                result.boxes = self.zoomer.map_boxes_to_original(
                    detection["boxes"],
                    crop_region,
                    original_size
                )
                result.scores = detection.get("scores", [0.5] * len(result.boxes))
                result.phrases = detection.get("phrases", [])
        
        elif plan.strategy == Strategy.MULTI_SCALE:
            # Try full image first, then grid
            all_boxes = []
            all_scores = []
            
            # Full image
            detection = self.grounding_dino.detect(
                image,
                plan.detection_prompt,
                box_threshold=plan.confidence_threshold,
                return_phrases=True
            )
            if detection.get("boxes"):
                all_boxes.extend(detection["boxes"])
                all_scores.extend(detection.get("scores", [0.5] * len(detection["boxes"])))
            
            # Grid crops if no good results
            if not all_boxes or state.iteration > 1:
                grid_crops = self.zoomer.create_grid_crops(image, (2, 2), overlap=0.15)
                for cropped, crop_region in grid_crops:
                    det = self.grounding_dino.detect(
                        cropped,
                        plan.detection_prompt,
                        box_threshold=plan.confidence_threshold,
                        return_phrases=True
                    )
                    if det.get("boxes"):
                        mapped = self.zoomer.map_boxes_to_original(
                            det["boxes"], crop_region, original_size
                        )
                        all_boxes.extend(mapped)
                        all_scores.extend(det.get("scores", [0.5] * len(mapped)))
            
            # Deduplicate overlapping boxes
            result.boxes, result.scores = self._nms_boxes(all_boxes, all_scores)
            result.source_region = "multi_scale"
        
        else:  # GLOBAL_SCAN
            detection = self.grounding_dino.detect(
                image,
                plan.detection_prompt,
                box_threshold=plan.confidence_threshold,
                return_phrases=True
            )
            result.boxes = detection.get("boxes", [])
            result.scores = detection.get("scores", [0.5] * len(result.boxes))
            result.phrases = detection.get("phrases", [])
            result.source_region = "full"
        
        logger.info(f"ACT: Detected {len(result.boxes)} boxes from '{result.source_region}'")
        return result
    
    def _nms_boxes(
        self,
        boxes: List[List[float]],
        scores: List[float],
        iou_threshold: float = 0.5
    ) -> Tuple[List[List[float]], List[float]]:
        """Apply Non-Maximum Suppression to remove overlapping boxes."""
        if not boxes:
            return [], []
        
        indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        
        keep = []
        while indices:
            current = indices.pop(0)
            keep.append(current)
            
            remaining = []
            for idx in indices:
                iou = self._compute_iou(boxes[current], boxes[idx])
                if iou < iou_threshold:
                    remaining.append(idx)
            indices = remaining
        
        return [boxes[i] for i in keep], [scores[i] for i in keep]
    
    def _compute_iou(self, box1: List[float], box2: List[float]) -> float:
        """Compute Intersection over Union."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def _synthesize_analysis(
        self,
        image: Image.Image,
        query: str,
        result: PipelineResult
    ) -> str:
        """Generate analysis summary using LLaVA."""
        if not self.llava:
            return ""
        
        context = f"""Based on the analysis:
- Image modality: {result.triage_modality} ({result.triage_confidence:.0%} confidence)
- Detection strategy: {result.strategic_plan.strategy.value if result.strategic_plan else 'N/A'}
- Regions detected: {len(result.verified_boxes)}
- Target: {result.strategic_plan.target_object if result.strategic_plan else 'abnormality'}

User query: {query}

Provide a concise medical analysis of the findings."""
        
        return self.llava.query(image, context, max_new_tokens=300)
    
    def _generate_report(
        self,
        image: Image.Image,
        result: PipelineResult
    ) -> str:
        """Generate structured medical report."""
        if not self.llava:
            return ""
        
        prompt = f"""Generate a brief structured report:

FINDINGS:
- Modality: {result.triage_modality}
- Detected regions: {len(result.verified_boxes)}
- Analysis: (summarize key findings)

IMPRESSION:
(one-line clinical impression)"""
        
        return self.llava.query(image, prompt, max_new_tokens=200)
    
    def _needs_rag(self, query: str) -> bool:
        """Check if query needs RAG knowledge retrieval."""
        rag_keywords = [
            "treatment", "therapy", "prognosis", "management",
            "symptoms", "causes", "guidelines", "recommend"
        ]
        return any(kw in query.lower() for kw in rag_keywords)
    
    def unload_all(self) -> None:
        """Unload all tools."""
        for tool in [self.biomedclip, self.grounding_dino, self.medsam, self.llava]:
            if tool:
                tool.unload()
        
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("All tools unloaded")
    
    def __repr__(self) -> str:
        tools = []
        if self.biomedclip: tools.append("BiomedCLIP")
        if self.grounding_dino: tools.append("DINO")
        if self.medsam: tools.append("MedSAM")
        if self.llava: tools.append("LLaVA")
        if self.rag: tools.append("RAG")
        return f"HybridReActOrchestrator(tools=[{', '.join(tools) or 'none'}], max_iter={self.max_iterations})"


# Backward compatibility alias
TriMedOrchestrator = HybridReActOrchestrator
