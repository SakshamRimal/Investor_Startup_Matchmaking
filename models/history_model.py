

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from sklearn.model_selection import train_test_split

SECTORS = [
    "FinTech", "HealthTech", "AI/ML", "CleanTech", "EdTech",
    "SaaS", "Cybersecurity", "Web3", "BioTech", "SpaceTech",
]


class InvestmentHistoryModel:
    """
    Bidirectional LSTM that learns sequential investment patterns and
    predicts which sectors an investor is likely to target next.

    Input  : padded sequence of past sector indices
    Output : softmax over all sectors  (next-sector probabilities)
    """

    def __init__(self, max_seq_len: int = 12, lstm_units: int = 64):
        self.max_seq_len = max_seq_len
        self.lstm_units  = lstm_units
        self.n_sectors   = len(SECTORS)
        self.sector_idx  = {s: i for i, s in enumerate(SECTORS)}
        self.model: Model = None
        self.is_trained   = False

    # ── Simulate sector sequences from investor profiles ───────────────────
    def _simulate_histories(self, investors_df: pd.DataFrame) -> list:
        histories = []
        for _, row in investors_df.iterrows():
            preferred = row["preferred_sectors"]
            n         = max(6, row["portfolio_size"] // 4)
            weights   = [3.0 if s in preferred else 1.0 for s in SECTORS]
            probs     = np.array(weights) / sum(weights)
            seq       = list(np.random.choice(self.n_sectors, size=n, p=probs))
            histories.append(seq)
        return histories

    # ── Build (input_seq, next_label) pairs ───────────────────────────────
    def _make_pairs(self, histories: list):
        X, y = [], []
        for hist in histories:
            for i in range(1, len(hist)):
                window = hist[max(0, i - self.max_seq_len): i]
                pad    = [0] * (self.max_seq_len - len(window)) + window
                X.append(pad)
                y.append(hist[i])
        return np.array(X), np.array(y)

    # ── Keras model ────────────────────────────────────────────────────────
    def _build(self) -> Model:
        inp = Input(shape=(self.max_seq_len,), name="sector_sequence")
        x   = layers.Embedding(self.n_sectors + 1, 32, mask_zero=True)(inp)
        x   = layers.Bidirectional(layers.LSTM(self.lstm_units))(x)
        x   = layers.Dropout(0.3)(x)
        x   = layers.Dense(64, activation="relu")(x)
        out = layers.Dense(self.n_sectors, activation="softmax", name="next_sector")(x)

        model = Model(inp, out, name="InvestmentHistoryLSTM")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, investors_df: pd.DataFrame, epochs: int = 40) -> dict:
        histories = self._simulate_histories(investors_df)
        X, y      = self._make_pairs(histories)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        self.model = self._build()
        early_stop = keras.callbacks.EarlyStopping(
            patience=5, restore_best_weights=True, verbose=0
        )

        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=[early_stop],
            verbose=0,
        )

        _, acc = self.model.evaluate(X_val, y_val, verbose=0)
        self.is_trained = True

        return {
            "model":        "InvestmentHistoryLSTM",
            "val_accuracy": round(float(acc), 4),
            "sequences":    len(X),
        }

    # ── Inference ──────────────────────────────────────────────────────────
    def predict_next(self, sector_history: list, top_k: int = 3) -> list:
        """
        Given a list of sector names, predict the top-k most likely next sectors.

        Example
        -------
        model.predict_next(["FinTech", "AI/ML", "SaaS"], top_k=3)
        → [("Cybersecurity", 0.21), ("SaaS", 0.18), ("FinTech", 0.14)]
        """
        seq    = [self.sector_idx.get(s, 0) for s in sector_history]
        window = seq[-self.max_seq_len:]
        padded = [0] * (self.max_seq_len - len(window)) + window
        probs  = self.model.predict(np.array([padded]), verbose=0)[0]
        ranked = sorted(zip(SECTORS, probs), key=lambda x: -x[1])[:top_k]
        return [(s, round(float(p), 4)) for s, p in ranked]
