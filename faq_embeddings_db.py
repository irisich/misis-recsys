"""
FAQ Embeddings Database
=======================

Semantic search for FAQ using sentence-transformers and FAISS.

Requirements:
    pip install sentence-transformers faiss-cpu numpy

Usage:
    from faq_embeddings_db import FAQEmbeddingsDB
    
    db = FAQEmbeddingsDB("data/faq.json")
    db.build_index()
    db.save("data/faq_index")
    
    results = db.search("когда сессия?", top_k=3)
"""

import json
import pickle
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import pymorphy2


_morph_analyzer = None


def get_morph_analyzer():
    """Lazy initialization of pymorphy2 analyzer"""
    global _morph_analyzer
    if _morph_analyzer is None:
        _morph_analyzer = pymorphy2.MorphAnalyzer()
    return _morph_analyzer


def normalize_text(text: str) -> str:
    """
    Normalize Russian text for better embedding matching.

    Process:
    1. Lowercase
    2. Lemmatization (normalize word forms to base forms)
    3. Strip extra whitespace

    Examples:
        "Часы работы СтО?" -> "час работа сто"
        "Когда начинается сессия?" -> "когда начинаться сессия"
        "работают" -> "работать"

    Args:
        text: Input text in Russian

    Returns:
        Normalized text
    """
    if not text:
        return ""

    morph = get_morph_analyzer()

    text = text.lower()

    words = re.findall(r'[а-яё\w]+', text)

    lemmas = []
    for word in words:
        parsed = morph.parse(word)
        if parsed:
            lemma = parsed[0].normal_form
            lemmas.append(lemma)
        else:
            lemmas.append(word)

    normalized = " ".join(lemmas)
    return normalized.strip()


@dataclass
class FAQItem:
    """Single FAQ item"""
    id: str
    block: str
    subblock: str
    question: str
    answer: str
    tags: List[str]
    
    @classmethod
    def from_dict(cls, data: dict) -> "FAQItem":
        return cls(
            id=data.get("id", ""),
            block=data.get("block", ""),
            subblock=data.get("subblock", ""),
            question=data.get("question", ""),
            answer=data.get("answer", ""),
            tags=data.get("tags", [])
        )
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "block": self.block,
            "subblock": self.subblock,
            "question": self.question,
            "answer": self.answer,
            "tags": self.tags
        }
    
    def get_embedding_text(self) -> str:
        """Text used for embedding generation (normalized)"""
        tags_text = " ".join(self.tags) if self.tags else ""
        raw_text = f"{self.question} {tags_text}".strip()
        return normalize_text(raw_text)


@dataclass
class SearchResult:
    """Search result with score"""
    score: float
    item: FAQItem
    
    def to_dict(self) -> dict:
        return {
            "score": self.score,
            **self.item.to_dict()
        }


class FAQEmbeddingsDB:
    """
    FAQ Database with semantic search using embeddings.

    Recommended models for Russian:
    - "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2" (fast)
    - "cointegrated/rubert-tiny2" (Russian-specific, very fast)
    """

    DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

    def __init__(self, faq_json_path: str, model_name: Optional[str] = None):
        """
        Initialize FAQ database.

        Args:
            faq_json_path: Path to faq.json
            model_name: Sentence transformer model
        """
        self.faq_json_path = Path(faq_json_path)
        self.model_name = model_name or self.DEFAULT_MODEL

        self.items: List[FAQItem] = []
        self.embeddings: Optional[np.ndarray] = None
        self.index: Optional[faiss.Index] = None
        self.model: Optional[SentenceTransformer] = None
        self.embedding_dim: int = 384

        logging.info("Preloading pymorphy2 dictionaries...")
        get_morph_analyzer()

        self._load_faq_data()
        logging.info(f"Loaded {len(self.items)} FAQ items")
    
    def _load_faq_data(self) -> None:
        """Load FAQ from JSON"""
        with open(self.faq_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.items = [FAQItem.from_dict(item) for item in data.get("dataset", [])]
    
    def _init_model(self) -> None:
        """Load sentence transformer model"""
        if self.model is None:
            logging.info(f"Loading model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
    
    def build_index(self) -> None:
        """Build embeddings and FAISS index"""
        self._init_model()
        
        texts = [item.get_embedding_text() for item in self.items]
        
        logging.info(f"Generating embeddings for {len(texts)} items...")
        self.embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True
        )
        
        logging.info("Building FAISS index...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings.astype(np.float32))
        
        logging.info(f"Index built with {self.index.ntotal} vectors")
    
    def save(self, path: str) -> None:
        """Save index to disk"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        faiss.write_index(self.index, str(path.with_suffix(".index")))
        
        metadata = {
            "model_name": self.model_name,
            "embedding_dim": self.embedding_dim,
            "embeddings": self.embeddings,
            "items": [item.to_dict() for item in self.items]
        }
        with open(path.with_suffix(".pkl"), "wb") as f:
            pickle.dump(metadata, f)
        
        logging.info(f"Saved index to {path}")
    
    def load(self, path: str) -> None:
        """Load index from disk"""
        path = Path(path)

        self.index = faiss.read_index(str(path.with_suffix(".index")))

        with open(path.with_suffix(".pkl"), "rb") as f:
            metadata = pickle.load(f)

        self.model_name = metadata["model_name"]
        self.embedding_dim = metadata["embedding_dim"]
        self.embeddings = metadata["embeddings"]
        self.items = [FAQItem.from_dict(item) for item in metadata["items"]]

        logging.info(f"Loaded index with {self.index.ntotal} vectors")

        logging.info("Preloading sentence transformer model...")
        self._init_model()
        logging.info("Model ready")
    
    def search(self, query: str, top_k: int = 5, score_threshold: float = 0.0) -> List[SearchResult]:
        """
        Search for similar FAQ items.

        Args:
            query: Search query
            top_k: Number of results
            score_threshold: Minimum similarity (0-1)

        Returns:
            List of SearchResult
        """
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        self._init_model()

        normalized_query = normalize_text(query)

        query_embedding = self.model.encode(
            [normalized_query],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype(np.float32)
        
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and score >= score_threshold:
                results.append(SearchResult(score=float(score), item=self.items[idx]))
        
        return results
    
    def find_answer(self, question: str, threshold: float = 0.5) -> Optional[str]:
        """Find answer for a question"""
        results = self.search(question, top_k=1)
        if results and results[0].score >= threshold:
            return results[0].item.answer
        return None
    
    def get_all_questions(self) -> List[Dict]:
        """Get all FAQ items as dicts"""
        return [item.to_dict() for item in self.items]
    
    def get_blocks(self) -> List[str]:
        """Get unique block names"""
        return list(set(item.block for item in self.items))
    
    def get_by_id(self, faq_id: str) -> Optional[FAQItem]:
        """Get item by ID"""
        for item in self.items:
            if item.id == faq_id:
                return item
        return None


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")

    print("Building FAQ embeddings index...")

    db = FAQEmbeddingsDB("faq.json")
    db.build_index()
    db.save("data/faq_index")
    
    print("\nTesting search...")
    test_queries = ["когда сессия?", "как получить студенческий", "общежитие"]
    
    for query in test_queries:
        print(f"\n🔍 {query}")
        for r in db.search(query, top_k=2):
            print(f"  [{r.score:.2f}] {r.item.question}")
