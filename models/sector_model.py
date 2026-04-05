"""
models/sector_model.py
───────────────────────
Industry / Sector Compatibility Model
Uses: NetworkX graph  +  Node2Vec random walks  +  TF-IDF  +  SVD embeddings
Output: sector similarity scores 0.0 – 1.0

Note: Install node2vec via `pip install node2vec`
      Falls back to SVD-only if node2vec is unavailable.
"""

import numpy as np
import pandas as pd
import networkx as nx
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    from node2vec import Node2Vec
    NODE2VEC_AVAILABLE = True
except ImportError:
    NODE2VEC_AVAILABLE = False

SECTORS = [
    "FinTech", "HealthTech", "AI/ML", "CleanTech", "EdTech",
    "SaaS", "Cybersecurity", "Web3", "BioTech", "SpaceTech",
]


class SectorCompatibilityModel:
    """
    Builds a sector co-investment graph from investor preferences,
    then learns dense embeddings via Node2Vec (or SVD fallback).

    Relationships learned
    ─────────────────────
    Sectors frequently co-preferred by investors → similar embeddings
    Sectors rarely paired → dissimilar embeddings
    """

    def __init__(self, embedding_dim: int = 16, walk_length: int = 20,
                 num_walks: int = 50, workers: int = 1):
        self.embedding_dim = embedding_dim
        self.walk_length   = walk_length
        self.num_walks     = num_walks
        self.workers       = workers
        self.sector_idx    = {s: i for i, s in enumerate(SECTORS)}
        self.graph         = None
        self.embeddings    = None   # shape: (n_sectors, embedding_dim)
        self.tfidf         = TfidfVectorizer(max_features=50, stop_words="english")
        self.is_trained    = False

    # ── Build co-investment graph ──────────────────────────────────────────
    def _build_graph(self, investors_df: pd.DataFrame) -> nx.Graph:
        G = nx.Graph()
        G.add_nodes_from(SECTORS)

        for _, row in investors_df.iterrows():
            sectors = [s for s in row["preferred_sectors"] if s in self.sector_idx]
            for i, s1 in enumerate(sectors):
                for s2 in sectors[i + 1:]:
                    if G.has_edge(s1, s2):
                        G[s1][s2]["weight"] += 1
                    else:
                        G.add_edge(s1, s2, weight=1)
        return G

    # ── SVD fallback embeddings (no node2vec package needed) ──────────────
    def _svd_embeddings(self, G: nx.Graph) -> np.ndarray:
        n   = len(SECTORS)
        adj = np.zeros((n, n))
        for s1, s2, data in G.edges(data=True):
            i, j = self.sector_idx[s1], self.sector_idx[s2]
            adj[i][j] = data.get("weight", 1)
            adj[j][i] = data.get("weight", 1)
        np.fill_diagonal(adj, adj.max(axis=1) + 1)
        log_adj = np.log1p(adj)
        svd     = TruncatedSVD(n_components=self.embedding_dim, random_state=42)
        return svd.fit_transform(log_adj)

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, investors_df: pd.DataFrame) -> dict:
        self.graph = self._build_graph(investors_df)

        if NODE2VEC_AVAILABLE:
            node2vec = Node2Vec(
                self.graph,
                dimensions=self.embedding_dim,
                walk_length=self.walk_length,
                num_walks=self.num_walks,
                workers=self.workers,
                quiet=True,
            )
            model = node2vec.fit(window=5, min_count=1, batch_words=4)
            self.embeddings = np.array([model.wv[s] for s in SECTORS])
            method = "Node2Vec"
        else:
            self.embeddings = self._svd_embeddings(self.graph)
            method = "SVD (node2vec not installed)"

        self.is_trained = True
        return {
            "model":         "SectorCompatibilityModel",
            "method":        method,
            "embedding_dim": self.embedding_dim,
            "graph_edges":   self.graph.number_of_edges(),
        }

    # ── Inference ──────────────────────────────────────────────────────────
    def similarity(self, sector_a: str, sector_b: str) -> float:
        a = self.embeddings[self.sector_idx[sector_a]].reshape(1, -1)
        b = self.embeddings[self.sector_idx[sector_b]].reshape(1, -1)
        return round(float(cosine_similarity(a, b)[0][0]), 4)

    def top_similar(self, sector: str, k: int = 3) -> list:
        sims = {s: self.similarity(sector, s) for s in SECTORS if s != sector}
        return sorted(sims.items(), key=lambda x: -x[1])[:k]

    def similarity_matrix(self) -> pd.DataFrame:
        sim = cosine_similarity(self.embeddings)
        return pd.DataFrame(sim, index=SECTORS, columns=SECTORS).round(3)

    def investor_sector_fit(self, investor_sectors: list, target_sector: str) -> float:
        """Average similarity between investor's sectors and a target sector."""
        scores = [self.similarity(s, target_sector) for s in investor_sectors
                  if s in self.sector_idx and s != target_sector]
        return round(float(np.mean(scores)), 4) if scores else 0.0
