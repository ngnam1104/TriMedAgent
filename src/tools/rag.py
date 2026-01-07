"""
Medical RAG Engine - Knowledge Retrieval
========================================

Retrieval-Augmented Generation using Groq API for medical knowledge.
Provides fast, accurate responses enhanced with retrieved context.
"""

from __future__ import annotations

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    # Groq settings
    groq_api_key: Optional[str] = None
    model_name: str = "llama3-70b-8192"
    temperature: float = 0.3
    max_tokens: int = 1024
    
    # Retrieval settings
    top_k: int = 3
    min_similarity: float = 0.3
    
    # Prompt templates
    system_prompt: str = """You are a medical knowledge assistant. 
Provide accurate medical information based on the retrieved context.
Always cite sources when possible. If context doesn't contain relevant information,
say so clearly. Do not make up information."""


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RetrievalResult:
    """Result from knowledge retrieval."""
    text: str
    source: str = ""
    similarity_score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RAGResponse:
    """Complete RAG response."""
    answer: str
    retrieved_contexts: List[RetrievalResult] = field(default_factory=list)
    model_used: str = ""
    tokens_used: int = 0
    success: bool = True
    error: Optional[str] = None


# =============================================================================
# Medical Knowledge Base
# =============================================================================

class MedicalKnowledgeBase:
    """In-memory medical knowledge base for RAG."""
    
    def __init__(self):
        self.documents = self._init_default_kb()
    
    def _init_default_kb(self) -> List[Dict[str, Any]]:
        """Initialize default medical knowledge base."""
        return [
            # Pulmonary conditions
            {
                "topic": "pneumonia",
                "text": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus. Chest X-rays typically show consolidation, air bronchograms, or infiltrates. Treatment depends on the cause: bacterial pneumonia requires antibiotics, while viral pneumonia may need supportive care.",
                "source": "Medical Reference"
            },
            {
                "topic": "pulmonary_nodule",
                "text": "Pulmonary nodules are small, round growths in the lungs. Most are benign (non-cancerous) and caused by old infections. Nodules smaller than 6mm typically require no follow-up. Larger nodules or those with suspicious features (spiculation, rapid growth) may need CT follow-up or biopsy.",
                "source": "Radiology Guidelines"
            },
            {
                "topic": "covid19",
                "text": "COVID-19 lung manifestations include ground-glass opacities (GGO), consolidation, and crazy-paving pattern on CT. Bilateral peripheral distribution is common. Severity correlates with extent of lung involvement. Treatment is supportive with antivirals in some cases.",
                "source": "COVID-19 Guidelines"
            },
            {
                "topic": "lung_cancer",
                "text": "Lung cancer often presents as a pulmonary nodule or mass. Concerning features include size >8mm, irregular margins, spiculation, and growth over time. PET-CT helps assess metabolic activity. Biopsy is required for definitive diagnosis.",
                "source": "Oncology Reference"
            },
            # Musculoskeletal
            {
                "topic": "fracture",
                "text": "Fractures are breaks in bone continuity. X-rays are the primary diagnostic tool showing disruption of cortex, displacement, or radiolucent lines. Types include transverse, oblique, spiral, and comminuted. Treatment ranges from immobilization to surgical fixation.",
                "source": "Orthopedic Reference"
            },
            # Cardiac
            {
                "topic": "cardiomegaly",
                "text": "Cardiomegaly is enlargement of the heart. On chest X-ray, cardiothoracic ratio >0.5 suggests cardiomegaly. Causes include heart failure, valvular disease, cardiomyopathy, and pericardial effusion. Echocardiogram provides detailed assessment.",
                "source": "Cardiology Reference"
            },
            # Abdominal
            {
                "topic": "liver_lesion",
                "text": "Liver lesions can be benign (hemangioma, cyst, focal nodular hyperplasia) or malignant (hepatocellular carcinoma, metastases). CT with contrast helps characterize lesions. Enhancement patterns and washout characteristics guide diagnosis.",
                "source": "Hepatology Reference"
            },
            # Neurological
            {
                "topic": "stroke",
                "text": "Acute ischemic stroke appears as hypodensity on CT, often subtle in first 6 hours. MRI DWI is more sensitive. IV tPA can be given within 4.5 hours of symptom onset. Mechanical thrombectomy may be performed for large vessel occlusion.",
                "source": "Neurology Guidelines"
            },
            {
                "topic": "brain_tumor",
                "text": "Brain tumors appear as mass lesions on CT/MRI with varying enhancement patterns. MRI with contrast is preferred for characterization. Surrounding edema and mass effect are common. Biopsy or surgical resection needed for diagnosis.",
                "source": "Neuro-oncology Reference"
            }
        ]
    
    def search(self, query: str, top_k: int = 3) -> List[RetrievalResult]:
        """Search knowledge base using keyword matching."""
        query_lower = query.lower()
        results = []
        
        for doc in self.documents:
            # Check topic match
            if doc["topic"] in query_lower:
                results.append(RetrievalResult(
                    text=doc["text"],
                    source=doc.get("source", ""),
                    similarity_score=1.0,
                    metadata={"topic": doc["topic"]}
                ))
                continue
            
            # Check text overlap
            doc_words = set(doc["text"].lower().split())
            query_words = set(query_lower.split())
            overlap = len(doc_words & query_words)
            
            if overlap > 2:
                results.append(RetrievalResult(
                    text=doc["text"],
                    source=doc.get("source", ""),
                    similarity_score=overlap / len(query_words),
                    metadata={"topic": doc["topic"]}
                ))
        
        results.sort(key=lambda x: x.similarity_score, reverse=True)
        return results[:top_k]
    
    def add_document(self, text: str, topic: str, source: str = "") -> None:
        """Add a document to the knowledge base."""
        self.documents.append({
            "topic": topic,
            "text": text,
            "source": source
        })


# =============================================================================
# Medical RAG Engine
# =============================================================================

class MedicalRAG:
    """
    Medical RAG Engine using Groq API.
    
    Features:
    - Fast inference with Groq's LPU
    - Medical knowledge retrieval
    - Context-aware generation
    - Conversation history support
    
    Example:
        rag = MedicalRAG()
        response = rag.query("What are the symptoms of pneumonia?")
        print(response.answer)
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        api_key: Optional[str] = None
    ):
        """
        Initialize Medical RAG.
        
        Args:
            config: RAG configuration
            api_key: Groq API key (uses env var if None)
        """
        self.config = config or RAGConfig()
        
        # Get API key
        self.api_key = api_key or self.config.groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize Groq client
        self.client = None
        if self.api_key:
            self._init_groq_client()
        else:
            logger.warning("No Groq API key. RAG will run in offline mode.")
        
        # Initialize knowledge base
        self.kb = MedicalKnowledgeBase()
        
        # Conversation history
        self.conversation_history: List[Dict[str, str]] = []
    
    def _init_groq_client(self) -> None:
        """Initialize Groq client."""
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key)
            logger.info("✓ Groq client initialized")
        except ImportError:
            logger.warning("groq package not installed. Run: pip install groq")
            self.client = None
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """Retrieve relevant documents."""
        k = top_k or self.config.top_k
        return self.kb.search(query, top_k=k)
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """Generate response using Groq."""
        if not self.client:
            return "[Groq API not available. Please set GROQ_API_KEY.]"
        
        messages = [
            {"role": "system", "content": system_prompt or self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            completion = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=messages,
                temperature=temperature or self.config.temperature,
                max_tokens=max_tokens or self.config.max_tokens
            )
            return completion.choices[0].message.content
        except Exception as e:
            logger.error(f"Groq generation error: {e}")
            return f"[Error: {str(e)}]"
    
    def query(
        self,
        question: str,
        include_context: bool = True,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """Query with retrieval-augmented generation."""
        retrieved = []
        context_str = ""
        
        if include_context:
            retrieved = self.retrieve(question, top_k=top_k)
            if retrieved:
                context_str = "\n\n".join([
                    f"[Source: {r.source}]\n{r.text}" for r in retrieved
                ])
        
        if context_str:
            prompt = f"""Based on the following medical reference information:

{context_str}

Please answer this question: {question}

Provide a clear, accurate response based on the context above."""
        else:
            prompt = f"""Answer this medical question to the best of your knowledge:

{question}

Note: No specific reference material was found for this query."""
        
        answer = self.generate(prompt)
        
        return RAGResponse(
            answer=answer,
            retrieved_contexts=retrieved,
            model_used=self.config.model_name,
            success=True
        )
    
    def stream(self, question: str) -> Generator[str, None, None]:
        """Stream response token by token."""
        if not self.client:
            yield "[Groq API not available]"
            return
        
        retrieved = self.retrieve(question)
        context_str = "\n".join([r.text for r in retrieved]) if retrieved else ""
        prompt = f"""Context:\n{context_str}\n\nQuestion: {question}"""
        
        try:
            stream = self.client.chat.completions.create(
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": self.config.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        except Exception as e:
            yield f"[Error: {str(e)}]"
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
    
    @property
    def available(self) -> bool:
        """Check if RAG is available (has API key)."""
        return self.client is not None
    
    def __repr__(self) -> str:
        status = "available" if self.available else "offline"
        return f"MedicalRAG(model={self.config.model_name}, status={status})"
