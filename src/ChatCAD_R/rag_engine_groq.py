"""
Medical RAG Engine with Groq API
================================

A refactored RAG (Retrieval-Augmented Generation) engine using Groq API
instead of OpenAI for fast, cost-effective medical knowledge retrieval.

Features:
- KDTree/FAISS vector search for knowledge retrieval
- Groq API for LLM generation (Llama3, Mixtral)
- Medical context enhancement
- Conversation history support
- Streaming response support

Based on ChatCAD_R architecture.

Author: TriMedAgent Team
Version: 1.0
"""

from __future__ import annotations

import os
import sys
import json
import logging
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Generator

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MedicalRAG")

# ==============================================================================
# Add paths
# ==============================================================================
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CHATCAD_ROOT = Path(__file__).resolve().parent

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(CHATCAD_ROOT) not in sys.path:
    sys.path.insert(0, str(CHATCAD_ROOT))

# ==============================================================================
# Optional imports (fail gracefully)
# ==============================================================================
try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    logger.warning("groq package not installed. Run: pip install groq")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

try:
    from scipy.spatial import KDTree
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    logger.warning("scipy not available for KDTree. Using fallback.")

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Embedding will use fallback.")


# ==============================================================================
# Configuration
# ==============================================================================
@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    # Groq settings
    groq_api_key: Optional[str] = None
    model_name: str = "llama3-70b-8192"  # Options: llama3-70b-8192, mixtral-8x7b-32768
    temperature: float = 0.3
    max_tokens: int = 1024
    
    # Retrieval settings
    top_k: int = 5
    min_similarity: float = 0.3
    
    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"  # Lightweight model for embeddings
    
    # Knowledge base paths
    knowledge_base_path: Optional[str] = None
    embeddings_cache_path: Optional[str] = None
    
    # Prompt templates
    system_prompt: str = """You are a medical knowledge assistant. 
Your role is to provide accurate medical information based on the retrieved context.
Always cite the source when possible. If the retrieved context doesn't contain 
relevant information, say so clearly. Do not make up information."""


# ==============================================================================
# Data Classes
# ==============================================================================
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


# ==============================================================================
# Simple Vector Store (Fallback)
# ==============================================================================
class SimpleVectorStore:
    """
    Simple in-memory vector store using cosine similarity.
    Fallback when FAISS/KDTree not available.
    """
    
    def __init__(self):
        self.texts: List[str] = []
        self.embeddings: List[List[float]] = []
        self.metadata: List[Dict] = []
    
    def add(self, text: str, embedding: List[float], metadata: Dict = None):
        self.texts.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
    
    def search(self, query_embedding: List[float], top_k: int = 5) -> List[Tuple[int, float]]:
        """Search for similar vectors using cosine similarity."""
        if not self.embeddings:
            return []
        
        if NUMPY_AVAILABLE:
            query = np.array(query_embedding)
            embeddings = np.array(self.embeddings)
            
            # Cosine similarity
            query_norm = query / (np.linalg.norm(query) + 1e-8)
            emb_norms = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8)
            similarities = np.dot(emb_norms, query_norm)
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[::-1][:top_k]
            return [(idx, similarities[idx]) for idx in top_indices]
        else:
            # Pure Python fallback
            def cosine_sim(a, b):
                dot = sum(x*y for x, y in zip(a, b))
                norm_a = sum(x*x for x in a) ** 0.5
                norm_b = sum(x*x for x in b) ** 0.5
                return dot / (norm_a * norm_b + 1e-8)
            
            scores = [(i, cosine_sim(query_embedding, emb)) for i, emb in enumerate(self.embeddings)]
            scores.sort(key=lambda x: x[1], reverse=True)
            return scores[:top_k]
    
    def save(self, path: str):
        """Save to pickle file."""
        data = {
            "texts": self.texts,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
    
    def load(self, path: str):
        """Load from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.embeddings = data["embeddings"]
        self.metadata = data.get("metadata", [{} for _ in self.texts])


# ==============================================================================
# KDTree Vector Store
# ==============================================================================
class KDTreeVectorStore:
    """
    Vector store using SciPy KDTree for fast similarity search.
    Based on ChatCAD_R's search engine.
    """
    
    def __init__(self):
        self.texts: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        self.metadata: List[Dict] = []
        self.tree: Optional[KDTree] = None
    
    def build(self, texts: List[str], embeddings: np.ndarray, metadata: List[Dict] = None):
        """Build the KDTree from embeddings."""
        self.texts = texts
        self.embeddings = embeddings
        self.metadata = metadata or [{} for _ in texts]
        self.tree = KDTree(embeddings)
        logger.info(f"Built KDTree with {len(texts)} documents")
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[int, float]]:
        """Search using KDTree."""
        if self.tree is None:
            return []
        
        distances, indices = self.tree.query(query_embedding.reshape(1, -1), k=min(top_k, len(self.texts)))
        
        # Convert distance to similarity (smaller distance = higher similarity)
        # Using inverse distance normalization
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            similarity = 1 / (1 + dist)  # Convert to similarity score
            results.append((idx, similarity))
        
        return results
    
    def save(self, path: str):
        """Save to pickle file."""
        data = {
            "texts": self.texts,
            "embeddings": self.embeddings,
            "metadata": self.metadata
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved KDTree store to {path}")
    
    def load(self, path: str):
        """Load from pickle file."""
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.embeddings = data["embeddings"]
        self.metadata = data.get("metadata", [{} for _ in self.texts])
        if self.embeddings is not None:
            self.tree = KDTree(self.embeddings)
        logger.info(f"Loaded KDTree store from {path} ({len(self.texts)} documents)")


# ==============================================================================
# Embedding Manager
# ==============================================================================
class EmbeddingManager:
    """
    Manages text embeddings using SentenceTransformers or fallback.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.model = None
        self._init_model()
    
    def _init_model(self):
        """Initialize embedding model."""
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                self.model = SentenceTransformer(self.model_name)
                logger.info(f"Loaded SentenceTransformer: {self.model_name}")
            except Exception as e:
                logger.warning(f"Failed to load SentenceTransformer: {e}")
                self.model = None
        else:
            logger.warning("SentenceTransformer not available, using hash-based fallback")
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for texts."""
        if isinstance(texts, str):
            texts = [texts]
        
        if self.model is not None:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings
        else:
            # Fallback: simple hash-based embeddings (not semantically meaningful!)
            return self._fallback_embed(texts)
    
    def _fallback_embed(self, texts: List[str], dim: int = 384) -> np.ndarray:
        """
        Fallback embedding using hash.
        WARNING: This is NOT semantically meaningful, only for testing!
        """
        import hashlib
        
        embeddings = []
        for text in texts:
            # Create deterministic "embedding" from text hash
            h = hashlib.sha256(text.encode()).hexdigest()
            vec = [int(h[i:i+2], 16) / 255.0 for i in range(0, min(len(h), dim*2), 2)]
            vec = vec + [0.0] * (dim - len(vec))  # Pad to dimension
            embeddings.append(vec)
        
        return np.array(embeddings) if NUMPY_AVAILABLE else embeddings


# ==============================================================================
# Medical Prompts
# ==============================================================================
MEDICAL_RAG_PROMPT = """Based on the following medical knowledge context, answer the user's question.

### Retrieved Medical Knowledge:
{context}

### User Question:
{question}

### Instructions:
1. Answer based on the provided context
2. If the context doesn't contain relevant information, clearly state that
3. Be concise but accurate
4. Use medical terminology appropriately
5. Do not make assumptions beyond the provided information

### Answer:"""


MERGE_FINDINGS_PROMPT = """You are a medical AI assistant. The user has received visual analysis findings 
and now has a follow-up question. Use the medical knowledge context to enhance your response.

### Visual Analysis Findings:
{visual_findings}

### Retrieved Medical Knowledge:
{context}

### User Question:
{question}

### Instructions:
Provide a comprehensive answer that:
1. Relates the visual findings to the medical knowledge
2. Explains any relevant treatment options, prognosis, or considerations
3. Is clinically accurate and evidence-based

### Answer:"""


# ==============================================================================
# Main MedicalRAG Class
# ==============================================================================
class MedicalRAG:
    """
    Medical Retrieval-Augmented Generation engine using Groq API.
    
    This class provides:
    - Knowledge retrieval from medical databases
    - Groq-powered generation with retrieved context
    - Integration with visual analysis findings
    
    Example:
        ```python
        rag = MedicalRAG(groq_api_key="your-api-key")
        rag.load_knowledge_base("path/to/medical_kb.pkl")
        
        response = rag.query("What is the treatment for pneumonia?")
        print(response.answer)
        ```
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        groq_api_key: Optional[str] = None
    ):
        """
        Initialize MedicalRAG.
        
        Args:
            config: RAG configuration
            groq_api_key: Groq API key (can also be set via GROQ_API_KEY env var)
        """
        self.config = config or RAGConfig()
        
        # Set API key (priority: param > config > env)
        self.api_key = groq_api_key or self.config.groq_api_key or os.getenv("GROQ_API_KEY")
        
        # Initialize components
        self.groq_client = None
        self.embedding_manager = None
        self.vector_store = None
        
        self._init_components()
    
    def _init_components(self):
        """Initialize all components."""
        # Groq client
        if GROQ_AVAILABLE and self.api_key:
            self.groq_client = Groq(api_key=self.api_key)
            logger.info("Groq client initialized")
        elif not self.api_key:
            logger.warning("No Groq API key provided. Set GROQ_API_KEY environment variable.")
        
        # Embedding manager
        self.embedding_manager = EmbeddingManager(self.config.embedding_model)
        
        # Vector store
        if SCIPY_AVAILABLE and NUMPY_AVAILABLE:
            self.vector_store = KDTreeVectorStore()
        else:
            self.vector_store = SimpleVectorStore()
    
    # ==========================================================================
    # Knowledge Base Management
    # ==========================================================================
    
    def load_knowledge_base(self, path: Union[str, Path]):
        """
        Load pre-built knowledge base from file.
        
        Args:
            path: Path to knowledge base pickle file
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {path}")
        
        if isinstance(self.vector_store, KDTreeVectorStore):
            self.vector_store.load(str(path))
        else:
            self.vector_store.load(str(path))
        
        logger.info(f"Loaded knowledge base with {len(self.vector_store.texts)} documents")
    
    def build_knowledge_base(
        self,
        texts: List[str],
        metadata: List[Dict] = None,
        save_path: Optional[str] = None
    ):
        """
        Build knowledge base from texts.
        
        Args:
            texts: List of text documents
            metadata: Optional metadata for each document
            save_path: Optional path to save the built knowledge base
        """
        logger.info(f"Building knowledge base from {len(texts)} documents...")
        
        # Generate embeddings
        embeddings = self.embedding_manager.embed(texts)
        
        # Build vector store
        if isinstance(self.vector_store, KDTreeVectorStore):
            self.vector_store.build(texts, embeddings, metadata)
        else:
            for i, (text, emb) in enumerate(zip(texts, embeddings)):
                meta = metadata[i] if metadata else {}
                self.vector_store.add(text, emb.tolist() if hasattr(emb, 'tolist') else emb, meta)
        
        # Save if path provided
        if save_path:
            self.vector_store.save(save_path)
        
        logger.info("Knowledge base built successfully")
    
    def load_from_chatcad_db(self, db_path: Union[str, Path]):
        """
        Load knowledge base from ChatCAD_R format.
        
        Args:
            db_path: Path to ChatCAD_R database directory
        """
        db_path = Path(db_path)
        
        # Try to load existing ChatCAD_R database format
        pkl_files = list(db_path.glob("*.pkl"))
        if pkl_files:
            self.load_knowledge_base(pkl_files[0])
            return
        
        # Try to load JSON format
        json_files = list(db_path.glob("*.json"))
        if json_files:
            texts = []
            metadata = []
            for jf in json_files:
                with open(jf, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    for item in data:
                        if isinstance(item, dict) and "text" in item:
                            texts.append(item["text"])
                            metadata.append(item.get("metadata", {}))
                        elif isinstance(item, str):
                            texts.append(item)
                            metadata.append({})
            
            if texts:
                self.build_knowledge_base(texts, metadata)
    
    # ==========================================================================
    # Retrieval
    # ==========================================================================
    
    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalResult]:
        """
        Retrieve relevant documents for a query.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of RetrievalResult
        """
        top_k = top_k or self.config.top_k
        
        if self.vector_store is None or len(self.vector_store.texts) == 0:
            logger.warning("No knowledge base loaded")
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_manager.embed(query)
        if len(query_embedding.shape) > 1:
            query_embedding = query_embedding[0]
        
        # Search
        results = self.vector_store.search(query_embedding, top_k=top_k)
        
        # Build result objects
        retrieval_results = []
        for idx, score in results:
            if score >= self.config.min_similarity:
                retrieval_results.append(RetrievalResult(
                    text=self.vector_store.texts[idx],
                    similarity_score=float(score),
                    metadata=self.vector_store.metadata[idx] if self.vector_store.metadata else {},
                    source=self.vector_store.metadata[idx].get("source", "Medical KB") if self.vector_store.metadata else "Medical KB"
                ))
        
        return retrieval_results
    
    # ==========================================================================
    # Generation
    # ==========================================================================
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate response using Groq.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt override
            temperature: Generation temperature
            max_tokens: Maximum tokens
            stream: Whether to stream response
            
        Returns:
            Generated text or generator for streaming
        """
        if not self.groq_client:
            raise RuntimeError("Groq client not initialized. Please provide API key.")
        
        messages = [
            {"role": "system", "content": system_prompt or self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if stream:
                return self._stream_generate(messages, temperature, max_tokens)
            else:
                response = self.groq_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=messages,
                    temperature=temperature or self.config.temperature,
                    max_tokens=max_tokens or self.config.max_tokens
                )
                return response.choices[0].message.content
                
        except Exception as e:
            logger.error(f"Generation error: {e}")
            raise
    
    def _stream_generate(
        self,
        messages: List[Dict],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> Generator[str, None, None]:
        """Stream generation."""
        stream = self.groq_client.chat.completions.create(
            model=self.config.model_name,
            messages=messages,
            temperature=temperature or self.config.temperature,
            max_tokens=max_tokens or self.config.max_tokens,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    
    # ==========================================================================
    # Query (Retrieve + Generate)
    # ==========================================================================
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        stream: bool = False
    ) -> RAGResponse:
        """
        Full RAG query: retrieve context and generate answer.
        
        Args:
            question: User question
            top_k: Number of contexts to retrieve
            stream: Whether to stream response
            
        Returns:
            RAGResponse with answer and retrieved contexts
        """
        try:
            # Retrieve
            contexts = self.retrieve(question, top_k=top_k)
            
            # Build context string
            context_str = "\n\n".join([
                f"[Source: {ctx.source}]\n{ctx.text}" 
                for ctx in contexts
            ]) if contexts else "No relevant context found."
            
            # Build prompt
            prompt = MEDICAL_RAG_PROMPT.format(
                context=context_str,
                question=question
            )
            
            # Generate
            if stream:
                # For streaming, return generator
                answer_gen = self.generate(prompt, stream=True)
                return RAGResponse(
                    answer="".join(answer_gen),  # Collect streamed response
                    retrieved_contexts=contexts,
                    model_used=self.config.model_name,
                    success=True
                )
            else:
                answer = self.generate(prompt)
                return RAGResponse(
                    answer=answer,
                    retrieved_contexts=contexts,
                    model_used=self.config.model_name,
                    success=True
                )
                
        except Exception as e:
            logger.error(f"RAG query failed: {e}")
            return RAGResponse(
                answer=f"Error: {str(e)}",
                success=False,
                error=str(e)
            )
    
    def enhance_visual_findings(
        self,
        visual_findings: str,
        user_question: str,
        top_k: Optional[int] = None
    ) -> RAGResponse:
        """
        Enhance visual analysis findings with RAG knowledge.
        
        Args:
            visual_findings: Findings from visual analysis (LLaVA output)
            user_question: User's follow-up question
            top_k: Number of contexts to retrieve
            
        Returns:
            RAGResponse with enhanced answer
        """
        try:
            # Build combined query for retrieval
            combined_query = f"{visual_findings}\n{user_question}"
            
            # Retrieve
            contexts = self.retrieve(combined_query, top_k=top_k)
            
            # Build context string
            context_str = "\n\n".join([
                f"[Source: {ctx.source}]\n{ctx.text}" 
                for ctx in contexts
            ]) if contexts else "No additional medical context available."
            
            # Build prompt
            prompt = MERGE_FINDINGS_PROMPT.format(
                visual_findings=visual_findings,
                context=context_str,
                question=user_question
            )
            
            # Generate
            answer = self.generate(prompt)
            
            return RAGResponse(
                answer=answer,
                retrieved_contexts=contexts,
                model_used=self.config.model_name,
                success=True
            )
            
        except Exception as e:
            logger.error(f"Enhancement failed: {e}")
            return RAGResponse(
                answer=visual_findings,  # Return original findings on error
                success=False,
                error=str(e)
            )
    
    # ==========================================================================
    # Convenience Method for Orchestrator Integration
    # ==========================================================================
    
    def retrieve_and_generate(self, query: str) -> str:
        """
        Simple interface for orchestrator integration.
        
        Args:
            query: Query string (may include visual findings)
            
        Returns:
            Generated answer string
        """
        response = self.query(query)
        return response.answer if response.success else ""


# ==============================================================================
# Factory Functions
# ==============================================================================
def create_medical_rag(
    api_key: Optional[str] = None,
    knowledge_base_path: Optional[str] = None,
    model_name: str = "llama3-70b-8192"
) -> MedicalRAG:
    """
    Factory function to create MedicalRAG instance.
    
    Args:
        api_key: Groq API key
        knowledge_base_path: Path to pre-built knowledge base
        model_name: Groq model to use
        
    Returns:
        Initialized MedicalRAG
    """
    config = RAGConfig(
        groq_api_key=api_key,
        model_name=model_name
    )
    
    rag = MedicalRAG(config)
    
    if knowledge_base_path and Path(knowledge_base_path).exists():
        rag.load_knowledge_base(knowledge_base_path)
    
    return rag


# ==============================================================================
# Testing
# ==============================================================================
if __name__ == "__main__":
    print("Medical RAG Engine with Groq")
    print("=" * 50)
    
    # Check dependencies
    print(f"\nDependencies:")
    print(f"  - Groq: {'✓' if GROQ_AVAILABLE else '✗'}")
    print(f"  - NumPy: {'✓' if NUMPY_AVAILABLE else '✗'}")
    print(f"  - SciPy: {'✓' if SCIPY_AVAILABLE else '✗'}")
    print(f"  - SentenceTransformers: {'✓' if SENTENCE_TRANSFORMERS_AVAILABLE else '✗'}")
    
    # Create instance
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print(f"\n✓ Groq API key found")
        rag = create_medical_rag(api_key=api_key)
        
        # Test with dummy knowledge base
        test_texts = [
            "Pneumonia is an infection of the lungs caused by bacteria, viruses, or fungi.",
            "Treatment for bacterial pneumonia typically includes antibiotics.",
            "Chest X-rays show consolidation in pneumonia cases.",
            "COVID-19 can cause ground-glass opacities in CT scans.",
        ]
        
        rag.build_knowledge_base(test_texts)
        
        # Test query
        print("\nTesting query...")
        response = rag.query("What is the treatment for pneumonia?")
        print(f"\nAnswer: {response.answer[:500]}...")
        print(f"\nRetrieved {len(response.retrieved_contexts)} contexts")
    else:
        print("\n✗ No GROQ_API_KEY found. Set it to test generation.")
        
        # Test retrieval only
        rag = MedicalRAG()
        test_texts = ["Test document 1", "Test document 2"]
        rag.build_knowledge_base(test_texts)
        
        results = rag.retrieve("test")
        print(f"\nRetrieval test: Found {len(results)} results")
