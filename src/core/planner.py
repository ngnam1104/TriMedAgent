"""
Strategic Planner - LLaVA-based Planning
========================================

Uses LLaVA to analyze image and generate strategic detection plan.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from PIL import Image

from ..types.models import Strategy, TargetSize, StrategicPlan
from ..utils.constants import ANATOMICAL_REGIONS, FALLBACK_REGIONS

logger = logging.getLogger(__name__)


class StrategicPlanner:
    """
    LLaVA-based Strategic Planner.
    
    Analyzes image and query to generate a strategic plan
    with target identification, location, and detection strategy.
    """
    
    PLANNING_PROMPT = """You are a medical imaging expert AI. Analyze this image and the user's query to create a detection strategy.

USER QUERY: {query}

Analyze the image and provide a strategic plan in the following JSON format:
```json
{{
    "target_object": "<what to detect, e.g., 'nodule', 'tumor', 'fracture'>",
    "target_size": "<tiny|small|medium|large>",
    "anatomical_location": "<specific region, e.g., 'right_upper_lobe', 'liver', 'left_lung'>",
    "strategy": "<global_scan|zoom_in|multi_scale>",
    "detection_prompt": "<specific prompt for object detection, e.g., 'small round nodule. white spot. lesion.'>",
    "confidence_threshold": <0.1-0.5>,
    "reasoning": "<your analysis reasoning>"
}}
```

SIZE GUIDELINES:
- tiny: Very small spots like calcifications, small nodules (< 5% of image)
- small: Lesions, small tumors, small fractures (5-15% of image)  
- medium: Organs, larger tumors (15-30% of image)
- large: Full organs, diffuse patterns (> 30% of image)

STRATEGY GUIDELINES:
- global_scan: For large/medium objects or when location is unknown
- zoom_in: For tiny/small objects when you can identify the region
- multi_scale: When uncertain about size

IMPORTANT: For small pathologies like nodules, you MUST specify zoom_in strategy with the anatomical_location.

Provide ONLY the JSON, no other text."""

    def __init__(self, llava_tool):
        self.llava = llava_tool
    
    def create_plan(
        self,
        image: Image.Image,
        query: str,
        modality: str = ""
    ) -> StrategicPlan:
        """
        Generate strategic plan from image and query.
        
        Args:
            image: Input image
            query: User query
            modality: Optional modality hint from triage
            
        Returns:
            StrategicPlan with detection strategy
        """
        prompt = self.PLANNING_PROMPT.format(query=query)
        
        if modality:
            prompt += f"\n\nHINT: This appears to be a {modality} image."
        
        try:
            response = self.llava.query(
                image,
                prompt,
                temperature=0.1,
                max_new_tokens=512
            )
            
            plan = self._parse_plan_response(response, query)
            logger.info(f"Strategic Plan: target={plan.target_object}, "
                       f"size={plan.target_size.value}, strategy={plan.strategy.value}")
            return plan
            
        except Exception as e:
            logger.warning(f"Planning failed: {e}, using default plan")
            return self._create_default_plan(query)
    
    def _parse_plan_response(self, response: str, query: str) -> StrategicPlan:
        """Parse LLaVA response into StrategicPlan."""
        plan = StrategicPlan()
        
        # Extract JSON from response
        json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                
                plan.target_object = data.get("target_object", "abnormality")
                plan.detection_prompt = data.get("detection_prompt", f"{plan.target_object}. lesion. finding.")
                plan.anatomical_location = data.get("anatomical_location", "")
                plan.confidence_threshold = float(data.get("confidence_threshold", 0.25))
                plan.reasoning = data.get("reasoning", "")
                
                # Parse target size
                size_str = data.get("target_size", "medium").lower()
                plan.target_size = {
                    "tiny": TargetSize.TINY,
                    "small": TargetSize.SMALL,
                    "medium": TargetSize.MEDIUM,
                    "large": TargetSize.LARGE
                }.get(size_str, TargetSize.MEDIUM)
                
                # Parse strategy
                strategy_str = data.get("strategy", "global_scan").lower()
                plan.strategy = {
                    "global_scan": Strategy.GLOBAL_SCAN,
                    "zoom_in": Strategy.ZOOM_IN,
                    "multi_scale": Strategy.MULTI_SCALE,
                    "sliding_window": Strategy.SLIDING_WINDOW
                }.get(strategy_str, Strategy.GLOBAL_SCAN)
                
                # Generate fallback regions
                plan.fallback_regions = self._get_fallback_regions(plan.anatomical_location)
                
                return plan
                
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON plan")
        
        return self._create_default_plan(query)
    
    def _create_default_plan(self, query: str) -> StrategicPlan:
        """Create default plan from query keywords."""
        query_lower = query.lower()
        plan = StrategicPlan()
        
        # Detect target from keywords
        if any(kw in query_lower for kw in ["nodule", "spot", "calcification"]):
            plan.target_object = "nodule"
            plan.target_size = TargetSize.TINY
            plan.strategy = Strategy.ZOOM_IN
            plan.detection_prompt = "small nodule. white spot. round lesion. calcification."
            plan.confidence_threshold = 0.15
        elif any(kw in query_lower for kw in ["tumor", "mass", "lesion"]):
            plan.target_object = "tumor"
            plan.target_size = TargetSize.SMALL
            plan.strategy = Strategy.MULTI_SCALE
            plan.detection_prompt = "tumor. mass. lesion. growth. abnormality."
            plan.confidence_threshold = 0.20
        elif any(kw in query_lower for kw in ["fracture", "break"]):
            plan.target_object = "fracture"
            plan.target_size = TargetSize.SMALL
            plan.strategy = Strategy.GLOBAL_SCAN
            plan.detection_prompt = "fracture. break. crack. bone injury."
            plan.confidence_threshold = 0.20
        else:
            plan.target_object = "abnormality"
            plan.target_size = TargetSize.MEDIUM
            plan.strategy = Strategy.GLOBAL_SCAN
            plan.detection_prompt = "abnormality. lesion. finding. pathology."
            plan.confidence_threshold = 0.25
        
        # Detect anatomical location
        for region in ANATOMICAL_REGIONS.keys():
            region_words = region.replace("_", " ")
            if region_words in query_lower:
                plan.anatomical_location = region
                if plan.target_size in [TargetSize.TINY, TargetSize.SMALL]:
                    plan.strategy = Strategy.ZOOM_IN
                break
        
        return plan
    
    def _get_fallback_regions(self, primary_region: str) -> list:
        """Get fallback regions to try if primary fails."""
        return FALLBACK_REGIONS.get(primary_region, ["center"])
