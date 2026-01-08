"""
Medical RAG Engine - Knowledge Retrieval V2
============================================

Retrieval-Augmented Generation for medical knowledge using:
- PubMedBERT for semantic embeddings (medical domain)
- FAISS for efficient vector similarity search
- Groq API for fast LLM generation

Architecture:
    Query → PubMedBERT → FAISS → Top-K Documents → LLM → Answer
    
Improvements over V1:
- Semantic search instead of keyword matching
- Medical domain embeddings (PubMedBERT)
- Scalable vector DB (FAISS)
- Hybrid retrieval (dense + sparse)
"""

from __future__ import annotations

import os
import json
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Any, Dict, Generator, List, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class RAGConfig:
    """Configuration for RAG engine."""
    # Embedding model (PubMedBERT for medical domain)
    embedding_model: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    embedding_dim: int = 768
    
    # Groq settings for generation
    groq_api_key: Optional[str] = None
    model_name: str = "llama3-70b-8192"
    temperature: float = 0.3
    max_tokens: int = 1024
    
    # Retrieval settings
    top_k: int = 5
    min_similarity: float = 0.3
    use_hybrid: bool = True  # Combine dense + sparse
    dense_weight: float = 0.7
    
    # Index settings
    index_path: Optional[str] = None
    use_gpu_faiss: bool = False
    
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
# Medical Knowledge Base with FAISS
# =============================================================================

class PubMedBERTEncoder:
    """
    PubMedBERT-based encoder for medical text embeddings.
    
    Uses microsoft/BiomedNLP-BiomedBERT which is pre-trained on:
    - PubMed abstracts
    - PMC full-text articles
    
    Superior for medical domain vs general BERT.
    """
    
    def __init__(
        self,
        model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
        device: str = "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu",
        max_length: int = 512
    ):
        self.device = device
        self.max_length = max_length
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        
    def load(self):
        """Lazy load model (heavy, only load when needed)"""
        if self.model is not None:
            return
            
        try:
            from transformers import AutoTokenizer, AutoModel
            import torch
            
            logger.info(f"Loading PubMedBERT encoder: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info(f"✓ PubMedBERT loaded on {self.device}")
        except Exception as e:
            logger.error(f"Failed to load PubMedBERT: {e}")
            raise
            
    def encode(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True,
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Encode texts to embeddings using mean pooling.
        
        Args:
            texts: Single text or list of texts
            normalize: L2 normalize embeddings
            batch_size: Batch size for encoding
            
        Returns:
            np.ndarray of shape (n_texts, 768)
        """
        import torch
        
        self.load()
        
        if isinstance(texts, str):
            texts = [texts]
            
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            # Tokenize
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            ).to(self.device)
            
            # Encode
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling over non-padding tokens
                attention_mask = inputs["attention_mask"]
                hidden_states = outputs.last_hidden_state
                
                # Expand mask for broadcasting
                mask_expanded = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
                sum_embeddings = torch.sum(hidden_states * mask_expanded, dim=1)
                sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
                embeddings = sum_embeddings / sum_mask
                
            all_embeddings.append(embeddings.cpu().numpy())
            
        embeddings = np.vstack(all_embeddings)
        
        if normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.maximum(norms, 1e-9)
            
        return embeddings


class FAISSIndex:
    """
    FAISS vector index for efficient similarity search.
    
    Supports:
    - Flat index (exact search, best for <100K docs)
    - IVF index (approximate, faster for >100K docs)
    - GPU acceleration (optional)
    """
    
    def __init__(
        self,
        dim: int = 768,
        use_gpu: bool = False,
        index_type: str = "flat"  # "flat" or "ivf"
    ):
        self.dim = dim
        self.use_gpu = use_gpu
        self.index_type = index_type
        self.index = None
        self.documents: List[Dict[str, Any]] = []
        
        self._init_index()
        
    def _init_index(self):
        """Initialize FAISS index"""
        try:
            import faiss
            
            if self.index_type == "flat":
                # Exact search with inner product (cosine sim for normalized vecs)
                self.index = faiss.IndexFlatIP(self.dim)
            else:
                # IVF for larger datasets
                quantizer = faiss.IndexFlatIP(self.dim)
                self.index = faiss.IndexIVFFlat(quantizer, self.dim, 100)
                
            if self.use_gpu:
                try:
                    res = faiss.StandardGpuResources()
                    self.index = faiss.index_cpu_to_gpu(res, 0, self.index)
                    logger.info("Using GPU FAISS")
                except Exception as e:
                    logger.warning(f"GPU FAISS failed, using CPU: {e}")
                    
            logger.info(f"✓ FAISS index initialized ({self.index_type})")
        except ImportError:
            logger.error("faiss not installed. Run: pip install faiss-cpu")
            raise
            
    def add(
        self,
        embeddings: np.ndarray,
        documents: List[Dict[str, Any]]
    ):
        """
        Add documents to index.
        
        Args:
            embeddings: (n_docs, dim) array
            documents: List of document dicts with 'text', 'source', etc.
        """
        import faiss
        
        if len(embeddings) != len(documents):
            raise ValueError(f"Mismatch: {len(embeddings)} embeddings vs {len(documents)} documents")
            
        # Train IVF if needed
        if self.index_type == "ivf" and not self.index.is_trained:
            self.index.train(embeddings.astype(np.float32))
            
        self.index.add(embeddings.astype(np.float32))
        self.documents.extend(documents)
        
        logger.info(f"Added {len(documents)} documents. Total: {len(self.documents)}")
        
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: (1, dim) or (dim,) array
            top_k: Number of results
            min_score: Minimum similarity threshold
            
        Returns:
            List of dicts with 'document', 'score', 'index'
        """
        if len(self.documents) == 0:
            return []
            
        # Reshape if needed
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        # Search
        scores, indices = self.index.search(
            query_embedding.astype(np.float32),
            min(top_k, len(self.documents))
        )
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= min_score:
                results.append({
                    "document": self.documents[idx],
                    "score": float(score),
                    "index": int(idx)
                })
                
        return results
        
    def save(self, path: str):
        """Save index and documents to disk"""
        import faiss
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, str(path / "index.faiss"))
        
        # Save documents
        with open(path / "documents.pkl", "wb") as f:
            pickle.dump(self.documents, f)
            
        logger.info(f"Saved index to {path}")
        
    def load(self, path: str):
        """Load index and documents from disk"""
        import faiss
        
        path = Path(path)
        
        # Load FAISS index
        self.index = faiss.read_index(str(path / "index.faiss"))
        
        # Load documents
        with open(path / "documents.pkl", "rb") as f:
            self.documents = pickle.load(f)
            
        logger.info(f"Loaded index with {len(self.documents)} documents")


class MedicalKnowledgeBase:
    """
    Medical knowledge base with PubMedBERT embeddings + FAISS.
    
    Supports:
    - Semantic search (dense retrieval)
    - Keyword search (sparse retrieval)
    - Hybrid search (dense + sparse)
    """
    
    def __init__(
        self,
        encoder: Optional[PubMedBERTEncoder] = None,
        index: Optional[FAISSIndex] = None,
        use_hybrid: bool = True,
        dense_weight: float = 0.7
    ):
        self.encoder = encoder or PubMedBERTEncoder()
        self.index = index or FAISSIndex()
        self.use_hybrid = use_hybrid
        self.dense_weight = dense_weight
        
        # Keyword index for sparse retrieval
        self.keyword_index: Dict[str, List[int]] = {}
        
        # Initialize with default knowledge
        self._init_default_kb()
    
    def _init_default_kb(self) -> None:
        """Initialize default medical knowledge base."""
        default_docs = [
            # Pulmonary conditions
            {
                "topic": "pneumonia",
                "text": "Pneumonia is an infection that inflames the air sacs in one or both lungs. The air sacs may fill with fluid or pus. Chest X-rays typically show consolidation, air bronchograms, or infiltrates. Treatment depends on the cause: bacterial pneumonia requires antibiotics, while viral pneumonia may need supportive care.",
                "source": "Medical Reference",
                "keywords": ["pneumonia", "lung infection", "consolidation", "infiltrate"]
            },
            {
                "topic": "pulmonary_nodule",
                "text": "Pulmonary nodules are small, round growths in the lungs. Most are benign (non-cancerous) and caused by old infections. Nodules smaller than 6mm typically require no follow-up. Larger nodules or those with suspicious features (spiculation, rapid growth) may need CT follow-up or biopsy.",
                "source": "Radiology Guidelines",
                "keywords": ["nodule", "lung nodule", "benign", "malignant", "spiculation"]
            },
            {
                "topic": "covid19",
                "text": "COVID-19 lung manifestations include ground-glass opacities (GGO), consolidation, and crazy-paving pattern on CT. Bilateral peripheral distribution is common. Severity correlates with extent of lung involvement. Treatment is supportive with antivirals in some cases.",
                "source": "COVID-19 Guidelines",
                "keywords": ["covid", "ground glass", "GGO", "coronavirus", "SARS-CoV-2"]
            },
            {
                "topic": "lung_cancer",
                "text": "Lung cancer often presents as a pulmonary nodule or mass. Concerning features include size >8mm, irregular margins, spiculation, and growth over time. PET-CT helps assess metabolic activity. Biopsy is required for definitive diagnosis.",
                "source": "Oncology Reference",
                "keywords": ["lung cancer", "tumor", "mass", "malignancy", "biopsy"]
            },
            # Cardiac
            {
                "topic": "cardiomegaly",
                "text": "Cardiomegaly is enlargement of the heart. On chest X-ray, cardiothoracic ratio >0.5 suggests cardiomegaly. Causes include heart failure, valvular disease, cardiomyopathy, and pericardial effusion. Echocardiogram provides detailed assessment.",
                "source": "Cardiology Reference",
                "keywords": ["cardiomegaly", "enlarged heart", "CTR", "cardiothoracic ratio", "heart failure"]
            },
            # Fractures
            {
                "topic": "fracture",
                "text": "Fractures are breaks in bone continuity. X-rays are the primary diagnostic tool showing disruption of cortex, displacement, or radiolucent lines. Types include transverse, oblique, spiral, and comminuted. Treatment ranges from immobilization to surgical fixation.",
                "source": "Orthopedic Reference",
                "keywords": ["fracture", "bone", "break", "displacement", "cortex"]
            },
            # Liver
            {
                "topic": "liver_lesion",
                "text": "Liver lesions can be benign (hemangioma, cyst, focal nodular hyperplasia) or malignant (hepatocellular carcinoma, metastases). CT with contrast helps characterize lesions. Enhancement patterns and washout characteristics guide diagnosis.",
                "source": "Hepatology Reference",
                "keywords": ["liver", "hepatic", "lesion", "HCC", "hemangioma", "metastasis"]
            },
            # Neurological
            {
                "topic": "stroke",
                "text": "Acute ischemic stroke appears as hypodensity on CT, often subtle in first 6 hours. MRI DWI is more sensitive. IV tPA can be given within 4.5 hours of symptom onset. Mechanical thrombectomy may be performed for large vessel occlusion.",
                "source": "Neurology Guidelines",
                "keywords": ["stroke", "ischemia", "infarct", "tPA", "thrombectomy"]
            },
            {
                "topic": "brain_tumor",
                "text": "Brain tumors appear as mass lesions on CT/MRI with varying enhancement patterns. MRI with contrast is preferred for characterization. Surrounding edema and mass effect are common. Biopsy or surgical resection needed for diagnosis.",
                "source": "Neuro-oncology Reference",
                "keywords": ["brain tumor", "glioma", "meningioma", "mass effect", "edema"]
            }
        ]
        
        # Store documents (encoding happens lazily on first search)
        for i, doc in enumerate(default_docs):
            self.index.documents.append(doc)
            # Build keyword index
            for kw in doc.get("keywords", []):
                if kw.lower() not in self.keyword_index:
                    self.keyword_index[kw.lower()] = []
                self.keyword_index[kw.lower()].append(i)
                
        self._embeddings_built = False
    
    def _build_embeddings(self):
        """Build embeddings for all documents (lazy initialization)"""
        if self._embeddings_built or len(self.index.documents) == 0:
            return
            
        logger.info("Building document embeddings...")
        texts = [doc["text"] for doc in self.index.documents]
        embeddings = self.encoder.encode(texts)
        
        # Reinitialize index and add embeddings
        docs = self.index.documents.copy()
        self.index = FAISSIndex(dim=embeddings.shape[1])
        self.index.add(embeddings, docs)
        
        self._embeddings_built = True
        logger.info(f"✓ Built embeddings for {len(docs)} documents")
    
    def _sparse_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Keyword-based sparse search"""
        query_words = query.lower().split()
        doc_scores: Dict[int, float] = {}
        
        for word in query_words:
            for kw, indices in self.keyword_index.items():
                if word in kw or kw in word:
                    for idx in indices:
                        doc_scores[idx] = doc_scores.get(idx, 0) + 1.0
                        
        # Normalize and sort
        if doc_scores:
            max_score = max(doc_scores.values())
            results = [
                {
                    "document": self.index.documents[idx],
                    "score": score / max_score,
                    "index": idx
                }
                for idx, score in sorted(doc_scores.items(), key=lambda x: -x[1])[:top_k]
            ]
            return results
        return []
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[RetrievalResult]:
        """
        Search knowledge base using hybrid retrieval.
        
        Args:
            query: Search query
            top_k: Number of results
            min_score: Minimum similarity threshold
            
        Returns:
            List of RetrievalResult
        """
        results_map: Dict[int, Dict[str, Any]] = {}
        
        # Dense search
        if self.use_hybrid or not self.keyword_index:
            self._build_embeddings()
            query_embedding = self.encoder.encode(query)
            dense_results = self.index.search(query_embedding, top_k * 2, min_score)
            
            for r in dense_results:
                idx = r["index"]
                results_map[idx] = {
                    "document": r["document"],
                    "dense_score": r["score"],
                    "sparse_score": 0.0
                }
                
        # Sparse search
        if self.use_hybrid and self.keyword_index:
            sparse_results = self._sparse_search(query, top_k * 2)
            
            for r in sparse_results:
                idx = r["index"]
                if idx in results_map:
                    results_map[idx]["sparse_score"] = r["score"]
                else:
                    results_map[idx] = {
                        "document": r["document"],
                        "dense_score": 0.0,
                        "sparse_score": r["score"]
                    }
                    
        # Combine scores
        final_results = []
        for idx, data in results_map.items():
            combined_score = (
                self.dense_weight * data.get("dense_score", 0) +
                (1 - self.dense_weight) * data.get("sparse_score", 0)
            )
            
            if combined_score >= min_score:
                doc = data["document"]
                final_results.append(RetrievalResult(
                    text=doc["text"],
                    source=doc.get("source", ""),
                    similarity_score=combined_score,
                    metadata={
                        "topic": doc.get("topic", ""),
                        "dense_score": data.get("dense_score", 0),
                        "sparse_score": data.get("sparse_score", 0)
                    }
                ))
                
        # Sort by combined score
        final_results.sort(key=lambda x: -x.similarity_score)
        return final_results[:top_k]
    
    def add_document(
        self,
        text: str,
        topic: str,
        source: str = "",
        keywords: Optional[List[str]] = None
    ) -> None:
        """Add a document to the knowledge base."""
        doc = {
            "topic": topic,
            "text": text,
            "source": source,
            "keywords": keywords or []
        }
        
        # Add to index
        if self._embeddings_built:
            # Need to re-encode
            embedding = self.encoder.encode(text)
            self.index.add(embedding, [doc])
        else:
            self.index.documents.append(doc)
            
        # Update keyword index
        idx = len(self.index.documents) - 1
        for kw in doc.get("keywords", []):
            if kw.lower() not in self.keyword_index:
                self.keyword_index[kw.lower()] = []
            self.keyword_index[kw.lower()].append(idx)
            
    def save(self, path: str):
        """Save knowledge base to disk"""
        self._build_embeddings()
        self.index.save(path)
        
        # Save keyword index
        with open(Path(path) / "keywords.json", "w") as f:
            json.dump(self.keyword_index, f)
            
    def load(self, path: str):
        """Load knowledge base from disk"""
        self.index.load(path)
        
        # Load keyword index
        kw_path = Path(path) / "keywords.json"
        if kw_path.exists():
            with open(kw_path, "r") as f:
                self.keyword_index = json.load(f)
                
        self._embeddings_built = True


# =============================================================================
# Medical RAG Engine
# =============================================================================

class MedicalRAG:
    """
    Medical RAG Engine V2 with PubMedBERT + FAISS.
    
    Features:
    - Semantic search with medical domain embeddings
    - Hybrid retrieval (dense + sparse)
    - Groq API for fast generation
    - Conversation history support
    
    Example:
        rag = MedicalRAG()
        response = rag.query("What are the symptoms of pneumonia?")
        print(response.answer)
    """
    
    def __init__(
        self,
        config: Optional[RAGConfig] = None,
        api_key: Optional[str] = None,
        knowledge_base: Optional[MedicalKnowledgeBase] = None
    ):
        """
        Initialize Medical RAG.
        
        Args:
            config: RAG configuration
            api_key: Groq API key (uses env var if None)
            knowledge_base: Pre-built knowledge base (creates new if None)
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
        
        # Initialize knowledge base with PubMedBERT + FAISS
        if knowledge_base:
            self.kb = knowledge_base
        else:
            encoder = PubMedBERTEncoder(model_name=self.config.embedding_model)
            index = FAISSIndex(
                dim=self.config.embedding_dim,
                use_gpu=self.config.use_gpu_faiss
            )
            self.kb = MedicalKnowledgeBase(
                encoder=encoder,
                index=index,
                use_hybrid=self.config.use_hybrid,
                dense_weight=self.config.dense_weight
            )
        
        # Load pre-built index if specified
        if self.config.index_path:
            try:
                self.kb.load(self.config.index_path)
            except Exception as e:
                logger.warning(f"Failed to load index: {e}. Using default KB.")
        
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
