from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_recall_curve

class CompatibilityModel:
    def __init__(self):
        self.model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X_test, y_test):
        y_pred = self.predict(X_test)
        auc = roc_auc_score(y_test, y_pred)
        print(f"Model AUC: {auc:.3f}")
        return auc