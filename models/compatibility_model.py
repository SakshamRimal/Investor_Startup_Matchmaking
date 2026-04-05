
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, Input
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


NUMERIC_FEATURES = [
    "sector_match", "stage_match", "check_fit", "growth_good", "runway_ok",
    "risk_appetite", "portfolio_size", "follow_on_rate",
    "mrr", "growth_rate", "burn_rate", "runway_months",
    "nps_score", "churn_rate", "team_size", "prior_exits",
]


class CompatibilityModel:


    def __init__(self):
        self.scaler = StandardScaler()
        self.model: Model = None
        self.is_trained = False
        
    @staticmethod
    def _engineer(df: pd.DataFrame) -> pd.DataFrame:
        X = df[NUMERIC_FEATURES].copy()
        X["mrr_log"]        = np.log1p(X["mrr"])
        X["burn_efficiency"] = X["mrr"] / (X["burn_rate"] + 1)
        X["risk_x_growth"]  = X["risk_appetite"] * X["growth_rate"]
        return X
    
    @staticmethod
    def _build(input_dim: int) -> Model:
        inp = Input(shape=(input_dim,), name="match_features")
        x   = layers.Dense(128, activation="relu")(inp)
        x   = layers.BatchNormalization()(x)
        x   = layers.Dropout(0.3)(x)
        x   = layers.Dense(64, activation="relu")(x)
        x   = layers.BatchNormalization()(x)
        x   = layers.Dropout(0.2)(x)
        x   = layers.Dense(32, activation="relu")(x)
        out = layers.Dense(1, activation="sigmoid", name="match_prob")(x)

        model = Model(inp, out, name="CompatibilityModel")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
            metrics=[keras.metrics.AUC(name="auc")],
        )
        return model

    
    def train(self, match_df: pd.DataFrame, epochs: int = 40) -> dict:
        X = self._engineer(match_df)
        y = match_df["outcome"].values

        X_scaled = self.scaler.fit_transform(X)
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )

        self.model = self._build(X_tr.shape[1])
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

        y_prob = self.model.predict(X_val, verbose=0).flatten()
        auc    = roc_auc_score(y_val, y_prob)
        self.is_trained = True

        return {
            "model":     "CompatibilityModel",
            "auc_roc":   round(auc, 4),
            "n_train":   len(X_tr),
            "n_val":     len(X_val),
        }

    def predict(self, match_df: pd.DataFrame) -> np.ndarray:
        X = self.scaler.transform(self._engineer(match_df))
        return self.model.predict(X, verbose=0).flatten()

    def predict_single(self, feature_dict: dict) -> float:
        df = pd.DataFrame([feature_dict])
        return float(self.predict(df)[0])
