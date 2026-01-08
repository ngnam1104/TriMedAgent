"""
Logic Router - Intent Classification Module
============================================

Phân loại intent của user query:
- THEORY: Câu hỏi lý thuyết → Route to RAG
- DIAGNOSIS: Yêu cầu chẩn đoán trên ảnh → Route to Planner

Architecture:
    Query → DistilBERT Classifier → P(diagnosis|query) → Route Decision
    
Threshold: P(diagnosis) > 0.7 → DIAGNOSIS, else THEORY
"""

import torch
import torch.nn as nn
from transformers import DistilBertTokenizer, DistilBertModel
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class Intent(Enum):
    """Query intent types"""
    THEORY = "theory"       # Theoretical questions → RAG
    DIAGNOSIS = "diagnosis" # Image diagnosis → Planner
    HYBRID = "hybrid"       # Both needed


@dataclass
class RouterResult:
    """Result from Logic Router"""
    intent: Intent
    confidence: float
    raw_scores: Dict[str, float]
    reasoning: str


class LogicRouter:
    """
    Intent classifier using DistilBERT
    
    Determines whether a query needs:
    1. RAG (medical knowledge) for theory questions
    2. Planner (vision tools) for diagnosis questions
    3. Both for hybrid questions
    
    Example:
        router = LogicRouter()
        result = router.classify("Tim có to không?", has_image=True)
        # result.intent == Intent.DIAGNOSIS
    """
    
    # Keywords that strongly indicate diagnosis intent
    DIAGNOSIS_KEYWORDS = [
        # Vietnamese
        "tổn thương", "bất thường", "có gì", "nhận diện", "xác định",
        "vị trí", "chỉ ra", "khoanh", "segment", "phát hiện",
        "to không", "nhỏ không", "bình thường không", "kích thước",
        "đo", "tính", "bao nhiêu", "mấy",
        # English  
        "detect", "find", "locate", "identify", "segment",
        "abnormal", "lesion", "tumor", "nodule", "mass",
        "measure", "size", "position", "where"
    ]
    
    # Keywords that strongly indicate theory intent
    THEORY_KEYWORDS = [
        # Vietnamese
        "là gì", "định nghĩa", "triệu chứng", "nguyên nhân",
        "điều trị", "phòng ngừa", "tiên lượng", "phân loại",
        "giải thích", "so sánh", "khác nhau", "tại sao",
        # English
        "what is", "define", "explain", "symptoms", "causes",
        "treatment", "prevention", "prognosis", "types",
        "difference", "compare", "why", "how does"
    ]
    
    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        diagnosis_threshold: float = 0.7,
        use_pretrained_classifier: bool = False,
        classifier_path: Optional[str] = None
    ):
        """
        Initialize Logic Router
        
        Args:
            model_name: Base DistilBERT model
            device: cuda or cpu
            diagnosis_threshold: P(diagnosis) > threshold → route to Planner
            use_pretrained_classifier: Load trained classifier head
            classifier_path: Path to trained classifier weights
        """
        self.device = device
        self.diagnosis_threshold = diagnosis_threshold
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        
        # Load base model
        self.encoder = DistilBertModel.from_pretrained(model_name)
        self.encoder.to(device)
        self.encoder.eval()
        
        # Classifier head: [CLS] embedding → 2 classes
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 2)  # [theory, diagnosis]
        ).to(device)
        
        # Load trained weights if available
        if use_pretrained_classifier and classifier_path:
            self._load_classifier(classifier_path)
        
        logger.info(f"LogicRouter initialized on {device}")
        
    def _load_classifier(self, path: str):
        """Load trained classifier weights"""
        try:
            state_dict = torch.load(path, map_location=self.device)
            self.classifier.load_state_dict(state_dict)
            logger.info(f"Loaded classifier from {path}")
        except Exception as e:
            logger.warning(f"Failed to load classifier: {e}. Using random init.")
            
    def _rule_based_score(self, query: str) -> Tuple[float, float]:
        """
        Rule-based scoring using keywords
        
        Returns:
            (theory_score, diagnosis_score) in [0, 1]
        """
        query_lower = query.lower()
        
        # Count keyword matches
        diagnosis_count = sum(1 for kw in self.DIAGNOSIS_KEYWORDS if kw in query_lower)
        theory_count = sum(1 for kw in self.THEORY_KEYWORDS if kw in query_lower)
        
        # Normalize
        total = diagnosis_count + theory_count + 1  # +1 to avoid div by 0
        
        diagnosis_score = diagnosis_count / total
        theory_score = theory_count / total
        
        return theory_score, diagnosis_score
    
    def _neural_score(self, query: str) -> Tuple[float, float]:
        """
        Neural scoring using DistilBERT
        
        Returns:
            (theory_prob, diagnosis_prob) from softmax
        """
        # Tokenize
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        # Get [CLS] embedding
        with torch.no_grad():
            outputs = self.encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]  # [1, 768]
            
            # Classify
            logits = self.classifier(cls_embedding)  # [1, 2]
            probs = torch.softmax(logits, dim=-1).squeeze()  # [2]
            
        theory_prob = probs[0].item()
        diagnosis_prob = probs[1].item()
        
        return theory_prob, diagnosis_prob
    
    def classify(
        self,
        query: str,
        has_image: bool = False,
        use_neural: bool = True,
        use_rules: bool = True,
        neural_weight: float = 0.6
    ) -> RouterResult:
        """
        Classify query intent
        
        Args:
            query: User's question
            has_image: Whether an image is attached
            use_neural: Use neural classifier
            use_rules: Use rule-based keywords
            neural_weight: Weight for neural score (1-neural_weight for rules)
            
        Returns:
            RouterResult with intent and confidence
        """
        # Special case: No image → Can only be theory
        if not has_image:
            return RouterResult(
                intent=Intent.THEORY,
                confidence=1.0,
                raw_scores={"theory": 1.0, "diagnosis": 0.0},
                reasoning="No image provided → Theory only"
            )
        
        # Get scores
        scores = {"theory": 0.0, "diagnosis": 0.0}
        
        if use_neural:
            neural_theory, neural_diag = self._neural_score(query)
            scores["theory"] += neural_weight * neural_theory
            scores["diagnosis"] += neural_weight * neural_diag
            
        if use_rules:
            rule_theory, rule_diag = self._rule_based_score(query)
            rule_weight = 1.0 - neural_weight if use_neural else 1.0
            scores["theory"] += rule_weight * rule_theory
            scores["diagnosis"] += rule_weight * rule_diag
            
        # Normalize
        total = scores["theory"] + scores["diagnosis"] + 1e-8
        scores["theory"] /= total
        scores["diagnosis"] /= total
        
        # Apply image boost: Having image increases diagnosis probability
        IMAGE_BOOST = 0.2
        scores["diagnosis"] = min(1.0, scores["diagnosis"] + IMAGE_BOOST)
        
        # Re-normalize
        total = scores["theory"] + scores["diagnosis"]
        scores["theory"] /= total
        scores["diagnosis"] /= total
        
        # Determine intent
        if scores["diagnosis"] > self.diagnosis_threshold:
            intent = Intent.DIAGNOSIS
            confidence = scores["diagnosis"]
            reasoning = f"P(diagnosis)={confidence:.2f} > {self.diagnosis_threshold} → Planner"
        elif scores["theory"] > self.diagnosis_threshold:
            intent = Intent.THEORY
            confidence = scores["theory"]
            reasoning = f"P(theory)={confidence:.2f} > {self.diagnosis_threshold} → RAG"
        else:
            # Ambiguous → Hybrid
            intent = Intent.HYBRID
            confidence = max(scores.values())
            reasoning = f"Ambiguous (theory={scores['theory']:.2f}, diagnosis={scores['diagnosis']:.2f}) → Both"
            
        return RouterResult(
            intent=intent,
            confidence=confidence,
            raw_scores=scores,
            reasoning=reasoning
        )
    
    def route(
        self,
        query: str,
        has_image: bool = False
    ) -> Dict[str, Any]:
        """
        Route query to appropriate module
        
        Returns:
            Dict with routing decision:
            {
                "use_rag": bool,
                "use_planner": bool,
                "intent": str,
                "confidence": float,
                "reasoning": str
            }
        """
        result = self.classify(query, has_image)
        
        return {
            "use_rag": result.intent in [Intent.THEORY, Intent.HYBRID],
            "use_planner": result.intent in [Intent.DIAGNOSIS, Intent.HYBRID],
            "intent": result.intent.value,
            "confidence": result.confidence,
            "reasoning": result.reasoning
        }


class LogicRouterTrainer:
    """
    Trainer for Logic Router classifier
    
    Fine-tune DistilBERT classifier on medical intent dataset
    """
    
    def __init__(
        self,
        router: LogicRouter,
        learning_rate: float = 2e-5,
        warmup_steps: int = 100
    ):
        self.router = router
        self.optimizer = torch.optim.AdamW(
            router.classifier.parameters(),
            lr=learning_rate
        )
        self.criterion = nn.CrossEntropyLoss()
        self.warmup_steps = warmup_steps
        self.step = 0
        
    def train_step(
        self,
        queries: list,
        labels: list  # 0=theory, 1=diagnosis
    ) -> float:
        """Single training step"""
        self.router.classifier.train()
        self.router.encoder.eval()  # Freeze encoder
        
        # Tokenize batch
        inputs = self.router.tokenizer(
            queries,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.router.device)
        
        labels_tensor = torch.tensor(labels).to(self.router.device)
        
        # Forward
        with torch.no_grad():
            outputs = self.router.encoder(**inputs)
            cls_embedding = outputs.last_hidden_state[:, 0, :]
            
        logits = self.router.classifier(cls_embedding)
        loss = self.criterion(logits, labels_tensor)
        
        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.step += 1
        return loss.item()
    
    def evaluate(
        self,
        queries: list,
        labels: list
    ) -> Dict[str, float]:
        """Evaluate accuracy"""
        self.router.classifier.eval()
        
        correct = 0
        total = len(queries)
        
        for query, label in zip(queries, labels):
            result = self.router.classify(query, has_image=True)
            pred = 1 if result.intent == Intent.DIAGNOSIS else 0
            if pred == label:
                correct += 1
                
        return {
            "accuracy": correct / total,
            "total": total
        }
    
    def save(self, path: str):
        """Save classifier weights"""
        torch.save(self.router.classifier.state_dict(), path)
        logger.info(f"Saved classifier to {path}")


# Convenience function for quick routing
def route_query(
    query: str,
    has_image: bool = False,
    router: Optional[LogicRouter] = None
) -> Dict[str, Any]:
    """
    Quick function to route a query
    
    Example:
        routing = route_query("Tim có to không?", has_image=True)
        if routing["use_planner"]:
            result = planner.plan(query, image)
    """
    if router is None:
        router = LogicRouter()
    return router.route(query, has_image)


if __name__ == "__main__":
    # Test
    router = LogicRouter()
    
    # Test cases
    test_cases = [
        ("Viêm phổi là gì?", False),      # Theory, no image
        ("Tim có to không?", True),        # Diagnosis, with image
        ("Có tổn thương gì trong ảnh này?", True),  # Diagnosis
        ("So sánh X-ray và CT scan", False),  # Theory
        ("Giải thích kết quả và chỉ ra vùng bất thường", True),  # Hybrid
    ]
    
    for query, has_image in test_cases:
        result = router.classify(query, has_image)
        print(f"\nQuery: {query}")
        print(f"Has Image: {has_image}")
        print(f"Intent: {result.intent.value}")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Reasoning: {result.reasoning}")
