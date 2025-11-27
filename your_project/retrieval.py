"""
Document retrieval system using TF-IDF for semantic search.
"""
import os
import re
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class DocumentChunk:
    """Represents a chunk of a document with metadata."""
    
    def __init__(self, content: str, source: str, chunk_id: str):
        self.content = content
        self.source = source  # filename without extension
        self.chunk_id = chunk_id  # e.g., "chunk0", "chunk1"
        self.full_id = f"{source}::{chunk_id}"
    
    def __repr__(self):
        return f"<Chunk {self.full_id}: {self.content[:50]}...>"


class DocumentRetriever:
    """TF-IDF based document retriever."""
    
    def __init__(self, docs_dir: str = "docs"):
        self.docs_dir = Path(docs_dir)
        self.chunks: List[DocumentChunk] = []
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            max_features=500,
            ngram_range=(1, 2)
        )
        self.chunk_vectors = None
        self._load_and_index_documents()
    
    def _load_and_index_documents(self):
        """Load all markdown files and create chunks."""
        if not self.docs_dir.exists():
            raise FileNotFoundError(f"Docs directory not found: {self.docs_dir}")
        
        md_files = list(self.docs_dir.glob("*.md"))
        if not md_files:
            raise ValueError(f"No markdown files found in {self.docs_dir}")
        
        print(f"📚 Loading {len(md_files)} documents...")
        
        for md_file in md_files:
            self._process_file(md_file)
        
        if not self.chunks:
            raise ValueError("No chunks created from documents")
        
        # Create TF-IDF vectors
        chunk_texts = [chunk.content for chunk in self.chunks]
        self.chunk_vectors = self.vectorizer.fit_transform(chunk_texts)
        
        print(f"✅ Indexed {len(self.chunks)} chunks from {len(md_files)} documents")
    
    def _process_file(self, file_path: Path):
        """Process a single markdown file into chunks."""
        source = file_path.stem  # filename without extension
        content = file_path.read_text(encoding='utf-8')
        
        # Split by paragraphs (double newline or headers)
        raw_chunks = re.split(r'\n\s*\n+', content)
        
        chunk_idx = 0
        for raw_chunk in raw_chunks:
            cleaned = raw_chunk.strip()
            if len(cleaned) > 20:  # Filter out very short chunks
                chunk = DocumentChunk(
                    content=cleaned,
                    source=source,
                    chunk_id=f"chunk{chunk_idx}"
                )
                self.chunks.append(chunk)
                chunk_idx += 1
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[DocumentChunk, float]]:
        """
        Search for relevant document chunks.
        
        Returns:
            List of (chunk, score) tuples, sorted by relevance
        """
        if not query.strip():
            return []
        
        # Vectorize query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_vector, self.chunk_vectors)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        # Filter out very low scores
        results = []
        for idx in top_indices:
            score = similarities[idx]
            if score > 0.01:  # Minimum relevance threshold
                results.append((self.chunks[idx], float(score)))
        
        return results
    
    def get_all_chunks(self) -> List[DocumentChunk]:
        """Return all chunks (useful for debugging)."""
        return self.chunks


# Singleton instance
_retriever_instance = None


def get_retriever(docs_dir: str = "docs") -> DocumentRetriever:
    """Get or create the global retriever instance."""
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = DocumentRetriever(docs_dir)
    return _retriever_instance


if __name__ == "__main__":
    # Test the retriever
    retriever = get_retriever()
    
    test_queries = [
        "return policy for beverages",
        "Summer Beverages campaign dates",
        "how to calculate AOV"
    ]
    
    for query in test_queries:
        print(f"\n🔍 Query: {query}")
        results = retriever.search(query, top_k=3)
        for chunk, score in results:
            print(f"  Score: {score:.3f} | {chunk.full_id}")
            print(f"  {chunk.content[:100]}...")