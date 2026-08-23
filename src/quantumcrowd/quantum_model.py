"""Optional Qiskit hybrid residual model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from .models import clip_probability, equal_market_weights, logit, sigmoid


@dataclass
class QuantumResidualModel:
    """Add a shallow EstimatorQNN feature to an anchored probability model.

    Qiskit imports are lazy so the classical research pipeline remains usable without
    quantum extras. The default estimator is exact and deterministic; hardware testing
    should occur only after the model and parameters are frozen.
    """

    feature_names: list[str]
    n_qubits: int = 4
    reps: int = 1
    l2: float = 0.02
    maxiter: int = 60
    seed: int = 7
    name: str = "qiskit_residual"

    def _prepare_training_features(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[self.feature_names].to_numpy(dtype=float)
        self.standardizer_ = StandardScaler().fit(raw)
        standardized = self.standardizer_.transform(raw)
        n_components = min(self.n_qubits, standardized.shape[1], standardized.shape[0] - 1)
        if n_components < 2:
            raise ValueError("QuantumResidualModel requires at least two components")
        self.actual_qubits_ = n_components
        self.pca_ = PCA(n_components=n_components, random_state=self.seed).fit(standardized)
        reduced = self.pca_.transform(standardized)
        self.angle_scaler_ = MinMaxScaler(feature_range=(-np.pi, np.pi)).fit(reduced)
        return self.angle_scaler_.transform(reduced)

    def _transform_features(self, df: pd.DataFrame) -> np.ndarray:
        raw = df[self.feature_names].to_numpy(dtype=float)
        reduced = self.pca_.transform(self.standardizer_.transform(raw))
        return np.clip(self.angle_scaler_.transform(reduced), -np.pi, np.pi)

    def fit(self, df: pd.DataFrame) -> "QuantumResidualModel":
        try:
            from qiskit import QuantumCircuit
            from qiskit.circuit.library import real_amplitudes, zz_feature_map
            from qiskit.primitives import StatevectorEstimator
            from qiskit_machine_learning.neural_networks import EstimatorQNN
        except ImportError as exc:
            raise ImportError(
                "Install the optional quantum dependencies with: pip install -e '.[quantum]'"
            ) from exc

        x = self._prepare_training_features(df)
        y = df["resolved"].to_numpy(dtype=float)
        offset = logit(df["market_prob"])
        sample_weight = equal_market_weights(df)
        sample_weight = sample_weight / sample_weight.sum()

        feature_map = zz_feature_map(feature_dimension=self.actual_qubits_, reps=1)
        ansatz = real_amplitudes(num_qubits=self.actual_qubits_, reps=self.reps)
        circuit = QuantumCircuit(self.actual_qubits_)
        circuit.compose(feature_map, inplace=True)
        circuit.compose(ansatz, inplace=True)
        self.circuit_ = circuit
        self.qnn_ = EstimatorQNN(
            circuit=circuit,
            input_params=feature_map.parameters,
            weight_params=ansatz.parameters,
            estimator=StatevectorEstimator(seed=self.seed),
            default_precision=0.0,
        )

        rng = np.random.default_rng(self.seed)
        n_qweights = self.qnn_.num_weights
        initial = np.concatenate(
            [
                rng.normal(scale=0.08, size=n_qweights),
                np.zeros(self.actual_qubits_),  # classical residual coefficients
                np.array([0.1, 0.0]),  # quantum scale and intercept
            ]
        )

        def unpack(params: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
            q_weights = params[:n_qweights]
            beta = params[n_qweights : n_qweights + self.actual_qubits_]
            alpha = float(params[-2])
            intercept = float(params[-1])
            return q_weights, beta, alpha, intercept

        def objective(params: np.ndarray) -> float:
            q_weights, beta, alpha, intercept = unpack(params)
            quantum_feature = np.asarray(self.qnn_.forward(x, q_weights)).reshape(-1)
            correction = intercept + (x / np.pi) @ beta + alpha * quantum_feature
            pred = clip_probability(sigmoid(offset + correction))
            loss = -np.sum(sample_weight * (y * np.log(pred) + (1 - y) * np.log(1 - pred)))
            penalty = 0.5 * self.l2 * np.dot(params, params)
            return float(loss + penalty)

        result = minimize(
            objective,
            initial,
            method="COBYLA",
            options={"maxiter": self.maxiter, "rhobeg": 0.2, "catol": 1e-6},
        )
        self.optimization_result_ = result
        self.q_weights_, self.beta_, self.alpha_, self.intercept_ = unpack(result.x)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        x = self._transform_features(df)
        quantum_feature = np.asarray(self.qnn_.forward(x, self.q_weights_)).reshape(-1)
        correction = self.intercept_ + (x / np.pi) @ self.beta_ + self.alpha_ * quantum_feature
        return clip_probability(sigmoid(logit(df["market_prob"]) + correction))

