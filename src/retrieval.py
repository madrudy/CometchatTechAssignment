from dataclasses import dataclass
from typing import Iterable
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from .knowledge import load_chunks
from .models import Chunk

@dataclass
class RetrievalResult:
    chunks: list[Chunk]
    query: str

class Retriever:
    
    def __init__(self, kb_dir: str, top_k: int = 6):
        self.chunks = load_chunks(kb_dir)
        self.top_k = top_k
        texts = [
            f"{c.metadata.get('title','')} {c.heading} {c.text}"
            for c in self.chunks
        ]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.matrix = self.vectorizer.fit_transform(texts)

    @staticmethod
    def _authority_bonus(c: Chunk) -> float:
        status = str(c.metadata.get("status", "")).lower()
        authority = str(c.metadata.get("policy_authority", "")).lower()
        audience = str(c.metadata.get("audience", "")).lower()
        if status == "active" and authority == "official" and audience == "customer":
            return 0.30
        if status == "active" and authority == "official":
            return 0.18
        if status == "superseded":
            return -0.30
        if status == "draft":
            return -0.45
        return 0.0

    @staticmethod
    def _safe_for_customer(c: Chunk) -> bool:
        md = c.metadata
        if str(md.get("customer_answering", "true")).lower() == "false":
            return False
        if str(md.get("status", "")).lower() in {"draft", "superseded"}:
            return False
        return True

    def search(self, query: str, top_k: int | None = None) -> RetrievalResult:
        top_k = top_k or self.top_k
        qv = self.vectorizer.transform([query])
        sims = cosine_similarity(qv, self.matrix)[0]
        scored: list[Chunk] = []
        for i, c in enumerate(self.chunks):
            if not self._safe_for_customer(c):
                continue
            raw = float(sims[i])
            if raw <= 0.06:
                continue
            score = raw + self._authority_bonus(c)
            scored.append(Chunk(c.chunk_id, c.filename, c.heading, c.text, c.metadata, score))
        scored.sort(key=lambda c: c.score, reverse=True)
        return RetrievalResult(scored[:top_k], query)

    def search_conflict_candidates(self, query: str) -> list[Chunk]:
        """Return high-scoring active official chunks across different docs for conflict checking."""
        result = self.search(query, top_k=min(12, len(self.chunks)))
        active = [
            c for c in result.chunks
            if str(c.metadata.get("status","")).lower() == "active"
            and str(c.metadata.get("policy_authority","")).lower() == "official"
        ]
        return active
