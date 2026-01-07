"""
LLaVA Tool - Visual Reasoning and Q&A
=====================================

Medical visual question answering using LLaVA model.
Supports 4-bit quantization for efficient GPU memory usage.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Union

import torch
from PIL import Image

from ..utils.image import base64_to_image, crop_image_by_box

logger = logging.getLogger(__name__)


class LLaVATool:
    """
    Visual reasoning using LLaVA model.
    
    Features:
    - Medical visual question answering
    - 4-bit quantization support (for T4/low-memory GPUs)
    - Batch processing
    - Streaming generation
    - Gatekeeper verification for detection results
    
    Example:
        tool = LLaVATool(quantize_4bit=True)
        response = tool.query(image, "What abnormalities do you see?")
        print(response)
    """
    
    # Default model for medical VQA
    DEFAULT_MODEL = "chaoyinshe/llava-med-v1.5-mistral-7b-hf"
    
    # System prompts
    SYSTEM_PROMPTS = {
        "medical": "You are an expert medical imaging assistant. Analyze the image carefully and provide accurate, clinically relevant information.",
        "general": "You are a helpful vision assistant. Describe what you see in the image.",
        "gatekeeper": "You are a medical imaging expert. Verify if the detected regions actually contain the specified finding. Answer with YES or NO followed by explanation.",
        "planner": "You are a medical imaging AI that creates detection strategies.",
    }
    
    def __init__(
        self,
        model_name: str = None,
        quantize_4bit: bool = True,
        device: Optional[str] = None,
        max_new_tokens: int = 512,
        load_on_init: bool = True
    ):
        """
        Initialize LLaVA tool.
        
        Args:
            model_name: HuggingFace model name (uses DEFAULT_MODEL if None)
            quantize_4bit: Use 4-bit quantization (recommended for T4 GPUs)
            device: Device to use (auto-detect if None)
            max_new_tokens: Maximum tokens to generate
            load_on_init: Whether to load model immediately
        """
        self.model_name = model_name or self.DEFAULT_MODEL
        self.quantize_4bit = quantize_4bit
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_new_tokens = max_new_tokens
        
        self.model = None
        self.processor = None
        
        if load_on_init:
            self.load_model()
    
    def load_model(self) -> None:
        """Load LLaVA model with optional quantization."""
        if self.model is not None:
            return
        
        try:
            from transformers import (
                LlavaForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig
            )
            
            logger.info(f"Loading LLaVA from {self.model_name}...")
            
            if self.quantize_4bit and self.device == "cuda":
                logger.info("Using 4-bit quantization for memory efficiency")
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
                
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True
                )
            else:
                self.model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto" if self.device == "cuda" else None,
                    low_cpu_mem_usage=True
                )
            
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            # Set pad token if needed
            if self.processor.tokenizer.pad_token is None:
                self.processor.tokenizer.pad_token = self.processor.tokenizer.eos_token
            
            logger.info(f"✓ LLaVA loaded on {self.device}")
            
        except ImportError as e:
            logger.error(f"Required packages not installed: {e}")
            logger.error("Run: pip install transformers bitsandbytes accelerate")
            raise
        except Exception as e:
            logger.error(f"Failed to load LLaVA: {e}")
            raise
    
    def query(
        self,
        image: Union[str, Image.Image],
        question: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: Optional[int] = None,
        temperature: float = 0.2,
        do_sample: bool = True
    ) -> str:
        """
        Query LLaVA with image and question.
        
        Args:
            image: PIL Image or base64 string
            question: Question to ask about the image
            system_prompt: Optional system prompt (uses medical default if None)
            max_new_tokens: Override default max tokens
            temperature: Sampling temperature (lower = more focused)
            do_sample: Whether to use sampling (vs greedy)
            
        Returns:
            Model response string
        """
        self.load_model()
        
        if isinstance(image, str):
            image = base64_to_image(image)
        
        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Build prompt
        sys_prompt = system_prompt or self.SYSTEM_PROMPTS["medical"]
        
        # Format for LLaVA
        prompt = f"<image>\n{sys_prompt}\n\nUser: {question}\nAssistant:"
        
        # Process inputs
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or self.max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.processor.tokenizer.pad_token_id
            )
        
        # Decode
        response = self.processor.decode(
            output_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True
        ).strip()
        
        return response
    
    def verify_detection(
        self,
        image: Union[str, Image.Image],
        detection_prompt: str,
        box: List[float] = None
    ) -> Dict[str, Any]:
        """
        Gatekeeper: Verify if detected region contains the finding.
        
        Args:
            image: PIL Image or base64 string
            detection_prompt: What was being detected (e.g., "tumor")
            box: Optional bounding box [x1, y1, x2, y2] to crop
            
        Returns:
            Dictionary with verified (bool), confidence, explanation
        """
        if isinstance(image, str):
            image = base64_to_image(image)
        
        # Crop if box provided
        if box is not None:
            image = crop_image_by_box(image, box, padding=10)
        
        question = f"""Examine this image region carefully.
        
Question: Does this region actually show a {detection_prompt}?

Respond in this exact format:
VERDICT: YES or NO
CONFIDENCE: HIGH, MEDIUM, or LOW
EXPLANATION: Brief explanation of what you see."""
        
        response = self.query(
            image,
            question,
            system_prompt=self.SYSTEM_PROMPTS["gatekeeper"],
            temperature=0.1
        )
        
        # Parse response
        result = {
            "verified": False,
            "confidence": "low",
            "explanation": response,
            "raw_response": response
        }
        
        response_upper = response.upper()
        
        if "VERDICT: YES" in response_upper or response_upper.startswith("YES"):
            result["verified"] = True
        
        if "CONFIDENCE: HIGH" in response_upper:
            result["confidence"] = "high"
        elif "CONFIDENCE: MEDIUM" in response_upper:
            result["confidence"] = "medium"
        
        return result
    
    def analyze(
        self,
        image: Union[str, Image.Image],
        analysis_type: str = "comprehensive"
    ) -> Dict[str, str]:
        """
        Perform structured analysis of medical image.
        
        Args:
            image: PIL Image or base64 string
            analysis_type: Type of analysis (comprehensive, findings, impression)
            
        Returns:
            Dictionary with analysis results
        """
        if analysis_type == "comprehensive":
            questions = {
                "modality": "What imaging modality is this?",
                "anatomy": "What anatomical region is shown?",
                "findings": "What abnormalities or notable findings are present?",
                "impression": "What is your overall impression of this image?"
            }
        elif analysis_type == "findings":
            questions = {
                "findings": "List all abnormal findings in this image.",
                "severity": "Rate the severity of any abnormalities (mild, moderate, severe)."
            }
        else:  # impression
            questions = {
                "impression": "Provide a concise clinical impression of this image."
            }
        
        results = {}
        for key, question in questions.items():
            results[key] = self.query(image, question)
        
        return results
    
    def generate_report(
        self,
        image: Union[str, Image.Image],
        patient_context: str = ""
    ) -> str:
        """
        Generate a structured medical report.
        
        Args:
            image: PIL Image or base64 string
            patient_context: Optional clinical context
            
        Returns:
            Formatted report string
        """
        context_str = f"Clinical context: {patient_context}\n\n" if patient_context else ""
        
        prompt = f"""{context_str}Generate a structured medical imaging report for this image.

Include the following sections:
1. TECHNIQUE: Imaging modality and parameters
2. COMPARISON: Note if prior studies available
3. FINDINGS: Detailed description of all findings
4. IMPRESSION: Summary and clinical recommendations

Be thorough but concise."""
        
        return self.query(
            image,
            prompt,
            system_prompt="You are an expert radiologist generating a formal medical imaging report.",
            max_new_tokens=1024
        )
    
    def unload(self) -> None:
        """Unload model to free memory."""
        if self.model is not None:
            del self.model
            del self.processor
            self.model = None
            self.processor = None
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            logger.info("LLaVA unloaded")
    
    def __repr__(self) -> str:
        status = "loaded" if self.model is not None else "not loaded"
        quant = "4-bit" if self.quantize_4bit else "full"
        return f"LLaVATool(model={self.model_name}, quantization={quant}, status={status})"
