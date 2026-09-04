"""CoxPH risk, survival-curve, and minimal concordance evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from pycox.models import CoxPH

from .train import validate_survival_dataset


@dataclass(frozen=True)
class RiskPrediction:
    sample_ids: tuple[str, ...]
    log_risk: np.ndarray
    relative_risk: np.ndarray
    interpretation: str = "RELATIVE_RISK_NOT_A_PROBABILITY"


@dataclass(frozen=True)
class SurvivalPrediction:
    sample_ids: tuple[str, ...]
    times: np.ndarray
    survival_probabilities: np.ndarray
    interpretation: str = "MODEL_BASED_UNCALIBRATED_SURVIVAL_ESTIMATES"


@dataclass(frozen=True)
class ConcordanceResult:
    value: float
    comparable_pairs: int
    method: str = "HARRELL_C_INDEX_NO_IPCW"
    interpretation: str = "DISCRIMINATION_ONLY"


@dataclass(frozen=True)
class CoxPHEvaluationResult:
    risk: RiskPrediction
    survival: SurvivalPrediction
    concordance: ConcordanceResult


def _sample_ids(sample_ids: Sequence[str] | None, count: int) -> tuple[str, ...]:
    resolved = tuple(str(value) for value in sample_ids) if sample_ids is not None else tuple(
        str(index) for index in range(count)
    )
    if len(resolved) != count or len(set(resolved)) != count:
        raise ValueError("sample_ids must be unique and match the sample count.")
    return resolved


def predict_log_risk(
    model: CoxPH, features: Any, *, sample_ids: Sequence[str] | None = None
) -> RiskPrediction:
    shape = getattr(features, "shape", ())
    dummy_count = int(shape[0]) if len(shape) > 0 else 0
    validated = validate_survival_dataset(
        features,
        np.zeros(dummy_count, dtype=np.float32),
        np.zeros(dummy_count, dtype=np.float32),
        partition="prediction",
        require_event=False,
    )
    identifiers = _sample_ids(sample_ids, validated.features.shape[0])
    log_risk = np.asarray(model.predict(validated.features), dtype=np.float64).reshape(-1)
    if log_risk.shape[0] != len(identifiers) or not np.isfinite(log_risk).all():
        raise RuntimeError("Model returned invalid log-risk predictions.")
    relative_risk = np.exp(log_risk)
    if not np.isfinite(relative_risk).all():
        raise RuntimeError("Exponentiated relative-risk predictions are non-finite.")
    return RiskPrediction(identifiers, log_risk, relative_risk)


def predict_survival(
    model: CoxPH, features: Any, *, sample_ids: Sequence[str] | None = None
) -> SurvivalPrediction:
    if getattr(model, "baseline_hazards_", None) is None:
        raise RuntimeError("Baseline hazards must be estimated from training outcomes first.")
    shape = getattr(features, "shape", ())
    dummy_count = int(shape[0]) if len(shape) > 0 else 0
    validated = validate_survival_dataset(
        features,
        np.zeros(dummy_count, dtype=np.float32),
        np.zeros(dummy_count, dtype=np.float32),
        partition="prediction",
        require_event=False,
    )
    risk = predict_log_risk(model, validated.features, sample_ids=sample_ids)
    frame = model.predict_surv_df(validated.features)
    values = frame.to_numpy(dtype=np.float64).T
    times = frame.index.to_numpy(dtype=np.float64)
    if values.shape != (len(risk.sample_ids), len(times)):
        raise RuntimeError("Unexpected survival-prediction dimensions.")
    if not np.isfinite(values).all() or np.any(values < 0.0) or np.any(values > 1.0):
        raise RuntimeError("Survival predictions must be finite values in [0, 1].")
    return SurvivalPrediction(risk.sample_ids, times, values)


def concordance_index(durations: Any, events: Any, log_risk: Any) -> ConcordanceResult:
    risk = np.asarray(log_risk, dtype=np.float64).reshape(-1)
    features = np.zeros((risk.shape[0], 1), dtype=np.float32)
    validated = validate_survival_dataset(
        features, durations, events, partition="evaluation", require_event=True
    )
    if risk.shape[0] != validated.durations.shape[0] or not np.isfinite(risk).all():
        raise ValueError("log_risk must be finite and match the outcome sample count.")
    concordant = 0.0
    comparable = 0
    for index, (duration, event) in enumerate(
        zip(validated.durations, validated.events, strict=True)
    ):
        if event != 1.0:
            continue
        later = np.flatnonzero(validated.durations > duration)
        for other in later:
            comparable += 1
            if risk[index] > risk[other]:
                concordant += 1.0
            elif risk[index] == risk[other]:
                concordant += 0.5
    if comparable == 0:
        raise ValueError("Concordance is undefined because there are no comparable pairs.")
    return ConcordanceResult(concordant / comparable, comparable)


def evaluate_coxph(
    model: CoxPH,
    features: Any,
    durations: Any,
    events: Any,
    *,
    sample_ids: Sequence[str] | None = None,
) -> CoxPHEvaluationResult:
    """Evaluate a held-out partition after training/model selection is complete."""

    validated = validate_survival_dataset(
        features, durations, events, partition="evaluation", require_event=True
    )
    risk = predict_log_risk(model, validated.features, sample_ids=sample_ids)
    survival = predict_survival(model, validated.features, sample_ids=risk.sample_ids)
    concordance = concordance_index(validated.durations, validated.events, risk.log_risk)
    return CoxPHEvaluationResult(risk, survival, concordance)
