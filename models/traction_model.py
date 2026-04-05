"""
models/traction_model.py
─────────────────────────
Startup Traction Scorer
Uses: Keras regression network
Output: traction score 0 – 100
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class TractionModel:
    """
    Regression network that scores startup traction on a 0–100 scale.

    Features analysed
    -----------------
    Growth rate · Burn multiple · Capital efficiency
    Churn rate  · NPS score     · Runway
    Founder strength · Team density
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self.model: Model = None
        self.is_trained = False

    # ── Feature engineering ────────────────────────────────────────────────
    @staticmethod
    def _engineer(df: pd.DataFrame):
        X = pd.DataFrame()
        X["burn_multiple"]       = df["burn_rate"] / (df["mrr"] + 1)
        X["capital_efficiency"]  = df["mrr"] / (df["total_raised"] / 12 + 1)
        X["growth_score"]        = df["growth_rate"].clip(-0.1, 0.5) * 200
        X["retention"]           = (1 - df["churn_rate"]) * 100
        X["runway_score"]        = df["runway_months"].clip(0, 18) / 18 * 100
        X["founder_strength"]    = df["prior_exits"] * 15 + df["patent_count"] * 5
        X["team_density"]        = df["active_users"] / (df["team_size"] + 1)
        X["nps"]                 = df["nps_score"]
        X["mrr_log"]             = np.log1p(df["mrr"])
        X["years"]               = df["years_operating"]

        # Composite traction label (used as training target)
        y = (
            X["growth_score"]                              * 0.30 +
            X["retention"]                                 * 0.20 +
            X["runway_score"]                              * 0.15 +
            X["capital_efficiency"].clip(0, 10) / 10 * 100 * 0.15 +
            X["nps"].clip(0, 80)    / 80         * 100 * 0.10 +
            X["founder_strength"].clip(0, 50) / 50 * 100 * 0.10
        ).clip(0, 100)

        return X.values, y.values

    # ── Keras model ────────────────────────────────────────────────────────
    @staticmethod
    def _build(input_dim: int) -> Model:
        inp = Input(shape=(input_dim,), name="startup_features")
        x   = layers.Dense(64)(inp)
        x   = layers.LeakyReLU(negative_slope=0.1)(x)
        x   = layers.Dropout(0.2)(x)
        x   = layers.Dense(32)(x)
        x   = layers.LeakyReLU(negative_slope=0.1)(x)
        x   = layers.Dense(16, activation="relu")(x)
        out = layers.Dense(1, name="traction_score")(x)

        model = Model(inp, out, name="TractionModel")
        model.compile(
            optimizer=keras.optimizers.Adam(5e-4),
            loss=keras.losses.Huber(),
            metrics=["mae"],
        )
        return model

    # ── Training ───────────────────────────────────────────────────────────
    def train(self, startups_df: pd.DataFrame, epochs: int = 60) -> dict:
        X, y     = self._engineer(startups_df)
        X_scaled = self.scaler.fit_transform(X)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = self._build(X_tr.shape[1])
        early_stop = keras.callbacks.EarlyStopping(
            patience=8, restore_best_weights=True, verbose=0
        )

        self.model.fit(
            X_tr, y_tr,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=16,
            callbacks=[early_stop],
            verbose=0,
        )

        y_pred = self.model.predict(X_val, verbose=0).flatten()
        mse    = mean_squared_error(y_val, y_pred)
        self.is_trained = True

        return {
            "model": "TractionModel",
            "mse":   round(mse, 3),
            "rmse":  round(float(np.sqrt(mse)), 3),
        }

    # ── Inference ──────────────────────────────────────────────────────────
    def predict(self, startups_df: pd.DataFrame) -> np.ndarray:
        X, _ = self._engineer(startups_df)
        X    = self.scaler.transform(X)
        return self.model.predict(X, verbose=0).flatten().clip(0, 100)

    def predict_single(self, startup_dict: dict) -> dict:
        df    = pd.DataFrame([startup_dict])
        score = float(self.predict(df)[0])
        X, _  = self._engineer(df)
        feats = X[0]
        return {
            "traction_score":    round(score, 2),
            "growth_score":      round(float(feats[2]), 2),
            "retention_score":   round(float(feats[3]), 2),
            "runway_score":      round(float(feats[4]), 2),
            "founder_strength":  round(float(feats[5]), 2),
        }
