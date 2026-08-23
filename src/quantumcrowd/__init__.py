"""QuantumCrowd: hybrid probability calibration for prediction markets."""

from .features import FEATURE_COLUMNS, build_features, select_forecast_horizon
from .models import AnchoredLogitModel, MarketBaseline, MLPProbabilityModel

__all__ = [
    "FEATURE_COLUMNS",
    "AnchoredLogitModel",
    "MarketBaseline",
    "MLPProbabilityModel",
    "build_features",
    "select_forecast_horizon",
]

__version__ = "0.2.0"
