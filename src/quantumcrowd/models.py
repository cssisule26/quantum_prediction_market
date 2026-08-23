"""Market and classical probability baselines."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def clip_probability(values: np.ndarray | pd.Series, eps: float = 1e-5) -> np.ndarray:
    return np.clip(np.asarray(values, dtype=float), eps, 1.0 - eps)


def logit(values: np.ndarray | pd.Series) -> np.ndarray:
    p = clip_probability(values)
    return np.log(p / (1.0 - p))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype=float), -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-values))


def equal_market_weights(df: pd.DataFrame) -> np.ndarray:
    counts = df.groupby("market_id")["market_id"].transform("size").to_numpy(dtype=float)
    return 1.0 / counts


class MarketBaseline:
    """Return the market probability without fitting."""

    name = "market"

    def fit(self, df: pd.DataFrame) -> "MarketBaseline":
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return clip_probability(df["market_prob"])


@dataclass
class AnchoredLogitModel:
    """Learn a log-odds correction while fixing the market-price coefficient to one."""

    feature_names: list[str]
    l2: float = 0.1
    maxiter: int = 500
    name: str = "anchored_logit"

    def fit(self, df: pd.DataFrame) -> "AnchoredLogitModel":
        x = df[self.feature_names].to_numpy(dtype=float)
        y = df["resolved"].to_numpy(dtype=float)
        offset = logit(df["market_prob"])
        weights = equal_market_weights(df)
        weights = weights / weights.sum()

        self.mean_ = x.mean(axis=0)
        self.scale_ = x.std(axis=0)
        self.scale_[self.scale_ < 1e-10] = 1.0
        xs = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(xs)), xs])

        def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
            correction = design @ params
            pred = sigmoid(offset + correction)
            loss = -np.sum(
                weights * (y * np.log(clip_probability(pred)) + (1 - y) * np.log(clip_probability(1 - pred)))
            )
            penalty = 0.5 * self.l2 * np.dot(params[1:], params[1:])
            gradient = design.T @ (weights * (pred - y))
            gradient[1:] += self.l2 * params[1:]
            return float(loss + penalty), gradient

        result = minimize(
            objective,
            np.zeros(design.shape[1]),
            method="L-BFGS-B",
            jac=True,
            options={"maxiter": self.maxiter},
        )
        if not result.success:
            raise RuntimeError(f"Anchored model optimization failed: {result.message}")
        self.intercept_ = float(result.x[0])
        self.coef_ = result.x[1:].copy()
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = df[self.feature_names].to_numpy(dtype=float)
        xs = (x - self.mean_) / self.scale_
        correction = self.intercept_ + xs @ self.coef_
        return clip_probability(sigmoid(logit(df["market_prob"]) + correction))


@dataclass
class MLPProbabilityModel:
    """Small nonlinear classical benchmark including the market logit as a feature."""

    feature_names: list[str]
    hidden_units: int = 8
    alpha: float = 0.01
    seed: int = 7
    maxiter: int = 600
    name: str = "classical_mlp"

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [logit(df["market_prob"]), df[self.feature_names].to_numpy(dtype=float)]
        )

    def fit(self, df: pd.DataFrame) -> "MLPProbabilityModel":
        model = MLPClassifier(
            hidden_layer_sizes=(self.hidden_units,),
            activation="tanh",
            solver="lbfgs",
            alpha=self.alpha,
            max_iter=self.maxiter,
            random_state=self.seed,
        )
        self.pipeline_ = make_pipeline(StandardScaler(), model)
        self.pipeline_.fit(self._matrix(df), df["resolved"].to_numpy(dtype=int))
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        return clip_probability(self.pipeline_.predict_proba(self._matrix(df))[:, 1])

