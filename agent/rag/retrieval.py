"""TF-IDF based retriever for local markdown docs.

This module implements a very small, dependency-light retriever using
scikit-learn's TfidfVectorizer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class DocChunk:
    """Represents a single retrievable chunk from a markdown document."""

    id: str
    source: str
    text: str


class TfidfRetriever:
    """Simple TF-IDF retriever over markdown docs/.

    This class is intentionally lightweight and local-only. It loads all
    markdown files in `docs_dir`, chunks them on blank lines, and builds a
    TF-IDF index. Retrieval is cosine-similarity over the TF-IDF vectors.
    """

    def __init__(self, docs_dir: str | Path):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocChunk] = []
        self.vectorizer: Optional[TfidfVectorizer] = None
        self.tfidf_matrix: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build_index(self) -> None:
        """Load documents from docs_dir, chunk them, and build TF-IDF index.

        Call this once at startup (e.g., from run_agent_hybrid.py) so that
        subsequent calls to `retrieve_topk` are fast.
        """

        self.chunks = self._load_and_chunk_docs(self.docs_dir)
        texts = [c.text for c in self.chunks]

        if not texts:
            raise RuntimeError(f"No markdown documents found in {self.docs_dir}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)

    def retrieve_topk(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve top-k chunks for a natural-language query."""

        if self.vectorizer is None or self.tfidf_matrix is None:
            raise RuntimeError("TF-IDF index has not been built. Call build_index() first.")

        if not query.strip():
            return []

        q_vec = self.vectorizer.transform([query])
        # cosine similarity for L2-normalized tf-idf
        scores = (q_vec @ self.tfidf_matrix.T).toarray()[0]

        # Get indices of top-k scores
        k = min(k, len(self.chunks))
        topk_idx = np.argsort(scores)[::-1][:k]

        results: List[Dict[str, Any]] = []
        for idx in topk_idx:
            chunk = self.chunks[int(idx)]
            score = float(scores[int(idx)])
            results.append(
                {
                    "id": chunk.id,
                    "source": chunk.source,
                    "text": chunk.text,
                    "score": score,
                }
            )
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _load_and_chunk_docs(docs_dir: Path) -> List[DocChunk]:
        """Load all .md files from docs_dir and chunk them on blank lines.

        Each paragraph (separated by one or more blank lines) becomes a chunk.
        """

        chunks: List[DocChunk] = []

        for md_path in sorted(docs_dir.glob("*.md")):
            text = md_path.read_text(encoding="utf-8")
            source = md_path.name

            # Split on blank lines, keep only non-empty chunks
            raw_chunks = _split_on_blank_lines(text)
            for i, chunk_text in enumerate(raw_chunks):
                chunk_text = chunk_text.strip()
                if not chunk_text:
                    continue
                chunk_id = f"{source}::chunk{i}"
                chunks.append(DocChunk(id=chunk_id, source=source, text=chunk_text))

        return chunks


def _split_on_blank_lines(text: str) -> List[str]:
    """Split text into blocks separated by one or more blank lines."""

    blocks: List[str] = []
    current_lines: List[str] = []

    for line in text.splitlines():
        if line.strip() == "":
            if current_lines:
                blocks.append("\n".join(current_lines))
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        blocks.append("\n".join(current_lines))

    return blocks


# Convenience function used by LangGraph nodes

_retriever_instance: Optional[TfidfRetriever] = None


def get_retriever(docs_dir: str | Path = "./docs") -> TfidfRetriever:
    """Get a global TfidfRetriever instance, building the index on first use."""

    global _retriever_instance
    if _retriever_instance is None:
        retriever = TfidfRetriever(docs_dir)
        retriever.build_index()
        _retriever_instance = retriever
    return _retriever_instance
