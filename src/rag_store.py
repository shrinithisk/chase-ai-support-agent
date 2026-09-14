"""
Vector RAG Retrieval Engine for @AmazonHelp Support Agent.
Indexes clean historical customer support resolutions using sentence embeddings and FAISS index
to retrieve top-3 exemplars for grounding reply generation.
"""

import os
import pandas as pd
import numpy as np
import faiss
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer


class RAGVectorStore:
    """
    FAISS-backed Vector RAG Store indexing @AmazonHelp historical resolution pairs.
    Provides fast top-k retrieval of similar past customer queries and brand responses.
    """

    def __init__(self, data_path: str = "data/processed_subset.csv"):
        self.data_path = data_path
        self.df = None
        self.vectorizer = None
        self.index = None
        self.is_initialized = False

    def initialize(self):
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Processed dataset not found at {self.data_path}")

        self.df = pd.read_csv(self.data_path)
        queries = self.df['customer_query'].astype(str).tolist()

        # Build TF-IDF vector representations (fast, deterministic, zero network dependency)
        self.vectorizer = TfidfVectorizer(max_features=1024, stop_words='english', ngram_range=(1, 2))
        X = self.vectorizer.fit_transform(queries).toarray().astype('float32')

        # Normalize vectors for Cosine Similarity search in FAISS IndexFlatIP
        faiss.normalize_L2(X)
        self.index = faiss.IndexFlatIP(X.shape[1])
        self.index.add(X)
        self.is_initialized = True

    def retrieve_exemplars(self, query: str, top_k: int = 3) -> List[Tuple[str, str, str]]:
        """
        Retrieves top-k semantically similar historical resolution pairs.
        Returns List of tuples: (historical_query, historical_response, intent).
        """
        if not self.is_initialized:
            self.initialize()

        q_vec = self.vectorizer.transform([query]).toarray().astype('float32')
        faiss.normalize_L2(q_vec)
        
        distances, indices = self.index.search(q_vec, top_k)
        
        results = []
        for idx in indices[0]:
            if idx < len(self.df):
                row = self.df.iloc[idx]
                results.append((
                    str(row['customer_query']),
                    str(row['brand_response']),
                    str(row['intent'])
                ))
        return results
