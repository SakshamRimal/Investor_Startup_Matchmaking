"""
models/suggestion_engine.py
────────────────────────────
Novel Opportunity Suggestion Engine
Uses: TF-IDF  +  Keras Autoencoder  +  KMeans clustering
Output: ranked startup suggestions per investor (similarity + novelty blend)
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

SECTORS = [
    "FinTech", "HealthTech", "AI/ML", "CleanTech", "EdTech",
    "SaaS", "Cybersecurity", "Web3", "BioTech", "SpaceTech",
]
STAGES = ["Pre-Seed", "Seed", "Series A", "Series B", "Series C"]
TEXT_DIM = 20
NUM_DIM  = 8


class SuggestionEngine:
    """
    Discovers diverse, novel investment opportunities.

    Steps
    ─────
    1. TF-IDF  — encode investor thesis / startup pitch as text vectors
    2. Autoencoder — compress all profiles into a shared latent space
    3. KMeans  — cluster latent vectors into affinity groups
    4. Rank    — blend cosine similarity + cross-cluster novelty bonus
    """

    def __init__(self, latent_dim: int = 16, n_clusters: int = 6):
        self.latent_dim  = latent_dim
        self.n_clusters  = n_clusters
        self.scaler      = StandardScaler()
        self.kmeans      = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        self.tfidf_inv   = TfidfVectorizer(max_features=TEXT_DIM, stop_words="english")
        self.tfidf_stp   = TfidfVectorizer(max_features=TEXT_DIM, stop_words="english")
        self.encoder: Model     = None
        self.autoencoder: Model = None
        self.inv_latent   = None
        self.stp_latent   = None
        self.inv_clusters = None
        self.stp_clusters = None
        self._investors: pd.DataFrame = None
        self._startups:  pd.DataFrame = None
        self.is_trained   = False

    # ── Feature builders ───────────────────────────────────────────────────
    def _inv_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        sec = np.zeros((len(df), len(SECTORS)))
        stg = np.zeros((len(df), len(STAGES)))
        for i, (_, row) in enumerate(df.iterrows()):
            for s in row["preferred_sectors"]:
                if s in SECTORS: sec[i, SECTORS.index(s)] = 1
            for s in row["preferred_stages"]:
                if s in STAGES:  stg[i, STAGES.index(s)] = 1

        fn  = self.tfidf_inv.fit_transform if fit else self.tfidf_inv.transform
        txt = fn(df["thesis"]).toarray()
        tf  = np.zeros((len(df), TEXT_DIM))
        tf[:, :txt.shape[1]] = txt[:, :TEXT_DIM]

        num = df[["risk_appetite", "portfolio_size", "follow_on_rate",
                  "years_active", "num_exits"]].values
        if num.shape[1] < NUM_DIM:
            pad = np.zeros((len(df), NUM_DIM - num.shape[1]))
            num = np.hstack([num, pad])
        return np.hstack([sec, stg, tf, num])

    def _stp_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
        sec = np.zeros((len(df), len(SECTORS)))
        stg = np.zeros((len(df), len(STAGES)))
        for i, (_, row) in enumerate(df.iterrows()):
            if row["sector"] in SECTORS: sec[i, SECTORS.index(row["sector"])] = 1
            if row["stage"]  in STAGES:  stg[i, STAGES.index(row["stage"])]  = 1

        fn  = self.tfidf_stp.fit_transform if fit else self.tfidf_stp.transform
        txt = fn(df["pitch"]).toarray()
        tf  = np.zeros((len(df), TEXT_DIM))
        tf[:, :txt.shape[1]] = txt[:, :TEXT_DIM]

        num = df[["mrr", "growth_rate", "burn_rate", "runway_months",
              "nps_score", "churn_rate", "team_size", "prior_exits"]].values
        return np.hstack([sec, stg, tf, num])

    # ── Keras Autoencoder ─────────────────────────────────────────────────
    def _build_autoencoder(self, input_dim: int) -> tuple:
        inp = Input(shape=(input_dim,), name="entity_features")

        # Encoder
        x      = layers.Dense(64, activation="relu")(inp)
        x      = layers.Dense(32, activation="relu")(x)
        latent = layers.Dense(self.latent_dim, activation="relu", name="latent")(x)

        # Decoder
        x   = layers.Dense(32, activation="relu")(latent)
        x   = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(input_dim, name="reconstruction")(x)

        autoencoder = Model(inp, out,    name="Autoencoder")
        encoder     = Model(inp, latent, name="Encoder")
        autoencoder.compile(optimizer="adam", loss="mse")
        return autoencoder, encoder

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, investors_df: pd.DataFrame, startups_df: pd.DataFrame,
              epochs: int = 50) -> dict:
        self._investors = investors_df.reset_index(drop=True)
        self._startups  = startups_df.reset_index(drop=True)

        inv_feats  = self._inv_features(investors_df, fit=True)
        stp_feats  = self._stp_features(startups_df,  fit=True)
        all_feats  = np.vstack([inv_feats, stp_feats])
        all_scaled = self.scaler.fit_transform(all_feats)

        self.autoencoder, self.encoder = self._build_autoencoder(all_scaled.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=0
        )
        self.autoencoder.fit(
            all_scaled, all_scaled,
            epochs=epochs, batch_size=32,
            validation_split=0.1,
            callbacks=[early_stop],
            verbose=0,
        )

        # Encode all entities
        all_latent        = self.encoder.predict(all_scaled, verbose=0)
        n_inv             = len(investors_df)
        self.inv_latent   = all_latent[:n_inv]
        self.stp_latent   = all_latent[n_inv:]

        # KMeans clustering
        all_clusters      = self.kmeans.fit_predict(all_latent)
        self.inv_clusters = all_clusters[:n_inv]
        self.stp_clusters = all_clusters[n_inv:]
        self.is_trained   = True

        return {
            "model":       "SuggestionEngine",
            "latent_dim":  self.latent_dim,
            "n_clusters":  self.n_clusters,
        }

    # ── Inference ──────────────────────────────────────────────────────────
    def suggest(self, investor_id: str, top_k: int = 5,
                novelty_weight: float = 0.3) -> list:
        """
        Return top-k startup suggestions for a given investor.

        novelty_weight : float
            0.0 = pure similarity  |  1.0 = maximize cross-cluster novelty
        """
        idx = self._investors[self._investors["investor_id"] == investor_id].index
        if len(idx) == 0:
            raise ValueError(f"Investor '{investor_id}' not found")
        idx = idx[0]

        inv_vec     = self.inv_latent[idx].reshape(1, -1)
        inv_cluster = self.inv_clusters[idx]

        sims          = cosine_similarity(inv_vec, self.stp_latent)[0]
        novelty_bonus = (self.stp_clusters != inv_cluster).astype(float) * 0.15
        combined      = (1 - novelty_weight) * sims + novelty_weight * (sims + novelty_bonus)

        top_idxs = np.argsort(combined)[::-1][:top_k]
        results  = []
        for i in top_idxs:
            stp = self._startups.iloc[i]
            results.append({
                "startup_id":       stp["startup_id"],
                "name":             stp["name"],
                "sector":           stp["sector"],
                "stage":            stp["stage"],
                "similarity_score": round(float(sims[i]), 4),
                "novelty_score":    round(float(novelty_bonus[i]), 4),
                "combined_score":   round(float(combined[i]), 4),
            })
        return results

    def cluster_report(self) -> pd.DataFrame:
        rows = []
        for c in range(self.n_clusters):
            rows.append({
                "cluster":   c,
                "investors": int((self.inv_clusters == c).sum()),
                "startups":  int((self.stp_clusters == c).sum()),
            })
        return pd.DataFrame(rows)
