"""Calibrated regime classifier (Phase 3).

Multinomial logistic regression on simulation-grid features, probability-
calibrated by temperature scaling fitted on the calibration grid itself.
The locked evaluation grid is never touched during fitting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

REGIME_CLASSES = [
    "SYNCHRONOUS", "NEAR_SYNCHRONOUS", "DIFFUSE_ASYNCHRONOUS",
    "CLUSTERED", "NO_TRANSITION",
]


def regime_label(truth: str, taus: np.ndarray, cluster_gap: float = 1.5) -> str:
    """Map engine truth + sampled taus to classifier classes."""
    if truth == "no_transition":
        return "NO_TRANSITION"
    if truth == "synchronous":
        return "SYNCHRONOUS"
    if truth == "near_synchronous":
        return "NEAR_SYNCHRONOUS"
    if truth in ("two_cluster", "three_cluster"):
        return "CLUSTERED"
    if truth == "diffuse":
        return "DIFFUSE_ASYNCHRONOUS"
    return truth


@dataclass
class RegimeClassifier:
    """Logistic + temperature-calibrated probabilities over REGIME_CLASSES."""

    C: float = 1.0
    max_iter: int = 2000
    model: object | None = None
    scaler_mean: np.ndarray | None = None
    scaler_sd: np.ndarray | None = None
    temperature: float = 1.0
    fitted: bool = False
    meta: dict = field(default_factory=dict)

    def _fit_logreg(self, X, y):
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import LabelEncoder

        self._le = LabelEncoder().fit(y)
        yv = self._le.transform(y)
        self.scaler_mean = X.mean(axis=0)
        self.scaler_sd = X.std(axis=0)
        self.scaler_sd[self.scaler_sd == 0] = 1.0
        Xs = (X - self.scaler_mean) / self.scaler_sd
        self.model = LogisticRegression(
            C=self.C, max_iter=self.max_iter, multi_class="multinomial",
            class_weight="balanced",
        )
        self.model.fit(Xs, yv)
        self.fitted = True
        self._classes_in_model = list(self._le.classes_)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RegimeClassifier":
        self._fit_logreg(np.asarray(X), np.asarray(y))
        # temperature scaling on the same (held-out-to-temp data should be
        # supplied separately via calibrate() when available)
        return self

    def calibrate(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fit temperature on a held-out calibration split; returns T."""
        logits = self._logits(X)
        best_t, best_loss = 1.0, np.inf
        for T_c in np.geomspace(0.2, 5.0, 60):
            p = self._softmax(logits / T_c)
            yi = self._le.transform(np.asarray(y))
            loss = -np.log(np.clip(p[np.arange(len(yi)), yi], 1e-12, 1)).mean()
            if loss < best_loss:
                best_loss, best_t = loss, T_c
        self.temperature = float(best_t)
        return self.temperature

    def _logits(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X) - self.scaler_mean) / self.scaler_sd
        return self.model.decision_function(Xs)

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / e.sum(axis=1, keepdims=True)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """(n, len(REGIME_CLASSES)) probability matrix, columns in class order."""
        p = self._softmax(self._logits(np.atleast_2d(X)) / self.temperature)
        out = np.zeros((p.shape[0], len(REGIME_CLASSES)))
        for j, cls in enumerate(self._classes_in_model):
            if cls in REGIME_CLASSES:
                out[:, REGIME_CLASSES.index(cls)] = p[:, j]
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return np.array(REGIME_CLASSES)[p.argmax(axis=1)]


def multiclass_brier(P: np.ndarray, y_idx: np.ndarray, n_classes: int) -> float:
    Y = np.zeros((len(y_idx), n_classes))
    Y[np.arange(len(y_idx)), y_idx] = 1.0
    return float(np.mean(np.sum((P - Y) ** 2, axis=1)))


def expected_calibration_error(P: np.ndarray, y_idx: np.ndarray, n_bins: int = 10) -> float:
    conf = P.max(axis=1)
    pred = P.argmax(axis=1)
    ece = 0.0
    for b in range(n_bins):
        lo, hi = b / n_bins, (b + 1) / n_bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            ece += m.mean() * abs((pred[m] == y_idx[m]).mean() - conf[m].mean())
    return float(ece)
