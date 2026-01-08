"""
LLaVA Brain Module
==================

Handles:
1. SFT/LoRA Model Loading (supports PeftModel)
2. JSON Plan Parsing
3. Visual Reasoning

V2 Updates:
- Support for loading LoRA adapters from local path or HuggingFace Hub
- Improved JSON parsing with repair heuristics
"""

import json
import logging
import re
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from transformers import (
    LlavaForConditionalGeneration,
    AutoProcessor,
    BitsAndBytesConfig
)

# Import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

from ...utils.image import base64_to_image, crop_image_by_box

logger = logging.getLogger(__name__)

class LLaVABrain:
    """
    The 'Brain' of TriMed-Agent.
    Responsible for high-level reasoning and planning.
    
    Supports:
    - Base model loading (quantized or full)
    - LoRA adapter loading from local path or HuggingFace Hub
    
    Example:
        # Base model only
        brain = LLaVABrain()
        
        # With LoRA adapter
        brain = LLaVABrain(lora_adapter="ngnam1104/TriMedAgent-V2/adapters/sft-v1")
    """
    
    SYSTEM_PROMPT = """You are TriMed-Agent, an expert medical AI.
Your goal is to analyze medical images and create execution plans.
Output your thought process in JSON format.
Example:
{
    "thought": "I see a chest X-ray. I need to check for nodules.",
    "action": "detect",
    "target": "lung nodule",
    "strategy": "zoom_in"
}
"""

    def __init__(
        self,
        model_name: str = "chaoyinshe/llava-med-v1.5-mistral-7b-hf",
        lora_adapter: Optional[str] = None,  # NEW: LoRA adapter path
        quantize_4bit: bool = True,
        device: str = "cuda:0",
        load_on_init: bool = False
    ):
        self.model_name = model_name
        self.lora_adapter = lora_adapter
        self.quantize_4bit = quantize_4bit
        self.device = device
        self.model = None
        self.processor = None
        
        if load_on_init:
            self.load_model()
            
    def load_model(self):
        """Load base model and optionally apply LoRA adapter"""
        if self.model is not None:
            return
            
        try:
            logger.info(f"Loading LLaVA Brain on {self.device}...")
            
            # Simple max_memory strategy for T4
            max_memory = {0: "12GiB", "cpu": "24GiB"}
            
            bnb_config = None
            if self.quantize_4bit:
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            
            # Load base model
            self.model = LlavaForConditionalGeneration.from_pretrained(
                self.model_name,
                quantization_config=bnb_config,
                device_map={'': 0},  # Force to device 0
                max_memory=max_memory,
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
            
            # Load LoRA adapter if specified
            if self.lora_adapter:
                self._load_lora_adapter()
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            logger.info("✅ LLaVA Brain Loaded" + 
                       (f" with LoRA: {self.lora_adapter}" if self.lora_adapter else ""))
            
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise
            
    def _load_lora_adapter(self):
        """
        Load LoRA adapter from local path or HuggingFace Hub.
        
        Supports:
        - Local path: "outputs/sft-v1/final"
        - HuggingFace: "ngnam1104/TriMedAgent-V2/adapters/sft-v1"
        """
        if not PEFT_AVAILABLE:
            logger.warning("PEFT not installed. Cannot load LoRA adapter. "
                          "Install with: pip install peft")
            return
            
        try:
            logger.info(f"Loading LoRA adapter: {self.lora_adapter}")
            
            # PeftModel.from_pretrained handles both local and HF paths
            self.model = PeftModel.from_pretrained(
                self.model,
                self.lora_adapter,
                is_trainable=False  # Inference mode
            )
            
            # Merge adapter for faster inference (optional)
            # self.model = self.model.merge_and_unload()
            
            logger.info("✅ LoRA adapter loaded successfully")
            
        except Exception as e:
            logger.warning(f"Failed to load LoRA adapter: {e}. Using base model.")
            
    def set_adapter(self, adapter_path: str):
        """
        Dynamically load a different LoRA adapter.
        Useful for switching between SFT and RL adapters.
        """
        if not PEFT_AVAILABLE:
            logger.error("PEFT not available")
            return False
            
        try:
            # If model is already a PeftModel, we need to reload base
            if hasattr(self.model, 'base_model'):
                # Unload current adapter
                self.model = self.model.unload()
                
            # Load new adapter
            self.model = PeftModel.from_pretrained(
                self.model,
                adapter_path,
                is_trainable=False
            )
            self.lora_adapter = adapter_path
            logger.info(f"✅ Switched to adapter: {adapter_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to set adapter: {e}")
            return False

    def query(self, image: Image.Image, prompt: str) -> str:
        self.load_model()
        
        full_prompt = f"<image>\n{self.SYSTEM_PROMPT}\nUser: {prompt}\nAssistant:"
        
        inputs = self.processor(text=full_prompt, images=image, return_tensors="pt").to(self.model.device)
        
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.2
            )
            
        return self.processor.decode(
            output_ids[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()

    def plan(self, image: Image.Image, task: str) -> Dict[str, Any]:
        """
        Generate a JSON plan for the given task.
        """
        prompt = f"""Task: {task}
Create a JSON execution plan.
Format:
{{
    "thought": "reasoning here",
    "action": "detect" or "zoom",
    "target": "object name"
}}
"""
        response = self.query(image, prompt)
        return self.parse_plan(response)
        
    def parse_plan(self, response: str) -> Dict[str, Any]:
        """
        Robustly parses a JSON plan from the LLM response.
        Handles markdown, missing brackets, and validation.
        """
        default_plan = {
            "action": "global_scan",
            "target": "abnormality",
            "thought": "Failed to parse plan, defaulting to global scan."
        }
        
        try:
            # 1. CLEANUP
            # Remove markdown code blocks
            clean_text = response.replace("```json", "").replace("```", "").strip()
            
            # 2. EXTRACT JSON
            # Regex to find the first outer-most brace pair
            # This regex looks for { followed by any char (lazy) until the last }
            match = re.search(r"\{.*\}", clean_text, re.DOTALL)
            
            if not match:
                logger.warning(f"No JSON found in response: {response[:100]}...")
                return default_plan
                
            json_str = match.group(0)
            
            # 3. REPAIR (Simple Heuristics)
            # If it doesn't end with '}', add it (LLM cut off)
            if not json_str.strip().endswith("}"):
                json_str += "}"
                
            # 4. PARSE
            try:
                plan = json.loads(json_str)
            except json.JSONDecodeError:
                # Try one more aggressive repair: look for last " using rfind
                # Sometimes LLM outputs: { ... "key": "val 
                # Very hard to fix generically without a library, but let's try strict mode false
                # Or just Fail safe.
                logger.error("JSON Decode failed even after cleanup.")
                return default_plan
                
            # 5. VALIDATE
            required_keys = ["action", "target"]
            for k in required_keys:
                if k not in plan:
                    logger.warning(f"Missing required key '{k}' in plan.")
                    return default_plan
                    
            # Normalize action (handle case sensitivity)
            if "zoom" in str(plan["action"]).lower():
                plan["action"] = "zoom"
            else:
                plan["action"] = "detect" # Default to detect/global
                
            return plan

        except Exception as e:
            logger.error(f"Critical error in parse_plan: {e}")
            return default_plan
