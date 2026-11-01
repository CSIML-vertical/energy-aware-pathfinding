"""Baseline and machine-learning estimators for road-quality estimation.

All estimators predict normalized simulated road roughness on the [0, 1]
scale. This module contains simple baselines, a supervised Random Forest,
and an iterative pseudo-labeling Random Forest used by the integrated runner.

Important: true roughness may be passed only as y_train (the labeled training
partition). It must never be included in X feature columns.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def predict_fixed_road_class(
    road_classes: Sequence[Any],
    road_class_settings: Mapping[str, Mapping[str, Any]],
    default: float = 0.5,
) -> np.ndarray:
    """Return the configured class-level initial value for each road class.

    The class values come from Raghavam's ROAD_CLASSES settings. Unknown
    classes use ``default``.
    """
    fixed_values = {
        name: float(settings["initial_estimate"])
        for name, settings in road_class_settings.items()
    }
    return np.asarray(
        [fixed_values.get(str(road_class), float(default)) for road_class in road_classes],
        dtype=float,
    )


def predict_initial_roughness(initial_roughness: Sequence[float]) -> np.ndarray:
    """Use each segment's supplied initial roughness estimate unchanged."""
    return np.asarray(initial_roughness, dtype=float).copy()


def predict_mean_observation_response(
    response_means: Sequence[float],
    initial_roughness: Sequence[float],
) -> np.ndarray:
    """Use mean calibrated vehicle response; fall back to initial roughness
    when a segment has no valid observation response.
    """
    response = np.asarray(response_means, dtype=float)
    initial = np.asarray(initial_roughness, dtype=float)
    if response.shape != initial.shape:
        raise ValueError("response_means and initial_roughness must have equal shapes")
    return np.where(np.isfinite(response), response, initial)


def predict_training_class_mean(
    training_road_classes: Sequence[Any],
    training_true_roughness: Sequence[float],
    target_road_classes: Sequence[Any],
) -> np.ndarray:
    """Predict each target segment using labeled-training mean for its class.

    The truth values passed here must come only from the labeled training
    partition. A global training-set mean is used for classes absent there.
    """
    train_classes = np.asarray(training_road_classes, dtype=object)
    train_targets = np.asarray(training_true_roughness, dtype=float)
    target_classes = np.asarray(target_road_classes, dtype=object)

    if train_classes.ndim != 1 or train_targets.ndim != 1 or target_classes.ndim != 1:
        raise ValueError("Class and target inputs must be one-dimensional")
    if len(train_classes) != len(train_targets):
        raise ValueError("training_road_classes and training_true_roughness must have equal lengths")
    if len(train_targets) == 0:
        raise ValueError("The labeled training partition cannot be empty")
    if not np.isfinite(train_targets).all():
        raise ValueError("training_true_roughness must contain only finite values")

    global_mean = float(np.mean(train_targets))
    class_means: dict[Any, float] = {}
    for road_class in dict.fromkeys(train_classes.tolist()):
        selected = train_targets[train_classes == road_class]
        if len(selected):
            class_means[road_class] = float(np.mean(selected))

    return np.asarray(
        [class_means.get(road_class, global_mean) for road_class in target_classes],
        dtype=float,
    )


def _resolve_features(
    X: pd.DataFrame,
    numeric_features: Sequence[str] | None,
    categorical_features: Sequence[str],
) -> tuple[list[str], list[str]]:
    """Resolve and validate feature column names for the RF pipelines."""
    categorical = list(categorical_features)
    numeric = (
        [column for column in X.columns if column not in categorical]
        if numeric_features is None
        else list(numeric_features)
    )
    requested = set(numeric) | set(categorical)
    missing = requested - set(X.columns)
    if missing:
        raise ValueError(f"Feature columns are missing from X: {sorted(missing)}")
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise ValueError(f"Features cannot be both numeric and categorical: {sorted(overlap)}")
    if not numeric and not categorical:
        raise ValueError("At least one numeric or categorical feature is required")
    return numeric, categorical


def make_random_forest_model(
    seed: int,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] = ("road_class",),
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
) -> Pipeline:
    """Build a preprocessing + Random Forest regression pipeline.

    Numeric columns are median-imputed. Categorical columns are most-frequent
    imputed and one-hot encoded. Because this factory does not receive a data
    frame, pass ``numeric_features`` explicitly when the model has numeric
    columns. The fit/predict functions infer numeric columns from their input
    DataFrame when ``numeric_features`` is omitted.
    """
    if n_estimators < 1:
        raise ValueError("n_estimators must be at least 1")
    if min_samples_leaf < 1:
        raise ValueError("min_samples_leaf must be at least 1")

    # Resolve column names when a sample DataFrame is supplied by callers in
    # the fit/predict functions. Here the caller supplies the feature lists.
    numeric = list(numeric_features or [])
    categorical = list(categorical_features)
    if not numeric and not categorical:
        raise ValueError("At least one numeric or categorical feature is required")

    transformers = []
    if numeric:
        numeric_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        transformers.append(("numeric", numeric_pipe, numeric))
    if categorical:
        category_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ])
        transformers.append(("categorical", category_pipe, categorical))

    preprocessor = ColumnTransformer(transformers)
    forest = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("model", forest)])


def _make_model_for_data(
    X: pd.DataFrame,
    seed: int,
    numeric_features: Sequence[str] | None,
    categorical_features: Sequence[str],
    n_estimators: int,
    min_samples_leaf: int,
) -> Pipeline:
    numeric, categorical = _resolve_features(X, numeric_features, categorical_features)
    return make_random_forest_model(
        seed=seed,
        numeric_features=numeric,
        categorical_features=categorical,
        n_estimators=n_estimators,
        min_samples_leaf=min_samples_leaf,
    )


def fit_predict_supervised_random_forest(
    X_train: pd.DataFrame,
    y_train: Sequence[float],
    X_target: pd.DataFrame,
    seed: int = 42,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] = ("road_class",),
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
) -> tuple[np.ndarray, Pipeline]:
    """Fit supervised Random Forest using labeled rows and predict target rows.

    Returns ``(predictions, fitted_model)``. Predictions are clipped to the
    valid normalized roughness range [0, 1].
    """
    if len(X_train) != len(y_train):
        raise ValueError("X_train and y_train must have equal lengths")
    if len(X_train) == 0:
        raise ValueError("X_train cannot be empty")
    model = _make_model_for_data(
        X_train, seed, numeric_features, categorical_features,
        n_estimators, min_samples_leaf,
    )
    model.fit(X_train, np.asarray(y_train, dtype=float))
    predictions = np.clip(model.predict(X_target), 0.0, 1.0)
    return predictions, model


def tree_mean_and_uncertainty(
    model: Pipeline,
    X: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return tree-mean predictions and tree-to-tree spread as heuristic uncertainty."""
    transformed = model.named_steps["preprocessor"].transform(X)
    forest = model.named_steps["model"]
    per_tree = np.vstack([tree.predict(transformed) for tree in forest.estimators_])
    return per_tree.mean(axis=0), per_tree.std(axis=0)


def iterative_fit_predict(
    X_labeled: pd.DataFrame,
    y_labeled: Sequence[float],
    X_pool: pd.DataFrame,
    X_test: pd.DataFrame,
    seed: int = 42,
    batch_size: int = 5,
    max_iterations: int = 10,
    confidence_quantile: float = 0.25,
    numeric_features: Sequence[str] | None = None,
    categorical_features: Sequence[str] = ("road_class",),
    n_estimators: int = 200,
    min_samples_leaf: int = 2,
) -> tuple[np.ndarray, pd.DataFrame, Pipeline, np.ndarray]:
    """Run generic Random Forest self-training with uncertainty-ranked pseudo-labels.

    The model is fitted on labeled data, predicts the unlabeled pool, then adds
    a batch of relatively low tree-disagreement predictions as pseudo-labels.
    This repeats for up to ``max_iterations``. Returns test predictions,
    iteration history, the final fitted model, and pool predictions.

    This is generic self-training, not an exact reproduction of the cited
    paper's vehicle-calibration method. Tree disagreement is a heuristic for
    uncertainty and is not a guarantee of correctness.
    """
    if len(X_labeled) != len(y_labeled):
        raise ValueError("X_labeled and y_labeled must have equal lengths")
    if len(X_labeled) == 0:
        raise ValueError("X_labeled cannot be empty")
    if batch_size < 1 or max_iterations < 1:
        raise ValueError("batch_size and max_iterations must be positive")
    if not 0.0 < confidence_quantile <= 1.0:
        raise ValueError("confidence_quantile must be in (0, 1]")

    X_labeled = X_labeled.reset_index(drop=True).copy()
    y_labeled = pd.Series(np.asarray(y_labeled, dtype=float)).reset_index(drop=True)
    X_pool = X_pool.reset_index(drop=True).copy()
    X_test = X_test.copy()
    remaining = list(range(len(X_pool)))
    pseudo_features: list[pd.DataFrame] = []
    pseudo_targets: list[float] = []
    history: list[dict[str, Any]] = []

    for iteration in range(1, max_iterations + 1):
        if not remaining:
            break
        X_fit = pd.concat([X_labeled] + pseudo_features, ignore_index=True)
        y_fit = pd.concat([y_labeled, pd.Series(pseudo_targets, dtype=float)], ignore_index=True)
        model = _make_model_for_data(
            X_fit, seed + iteration, numeric_features, categorical_features,
            n_estimators, min_samples_leaf,
        )
        model.fit(X_fit, y_fit)

        candidate_X = X_pool.iloc[remaining]
        predictions, uncertainty = tree_mean_and_uncertainty(model, candidate_X)
        cutoff = float(np.quantile(uncertainty, confidence_quantile))
        eligible = np.flatnonzero(uncertainty <= cutoff)
        if len(eligible) == 0:
            eligible = np.array([int(np.argmin(uncertainty))])
        ordered = eligible[np.argsort(uncertainty[eligible])]
        chosen_local = ordered[:batch_size]
        chosen_global = [remaining[int(i)] for i in chosen_local]

        for local_i, global_i in zip(chosen_local, chosen_global):
            pseudo_features.append(X_pool.iloc[[global_i]].copy())
            pseudo_targets.append(float(predictions[int(local_i)]))

        history.append({
            "iteration": iteration,
            "pool_rows_remaining_before": len(remaining),
            "pseudo_labels_added": len(chosen_global),
            "uncertainty_cutoff": cutoff,
            "mean_selected_tree_spread": float(np.mean(uncertainty[chosen_local])),
            "mean_pseudo_roughness": float(np.mean(predictions[chosen_local])),
        })
        chosen_set = set(chosen_global)
        remaining = [i for i in remaining if i not in chosen_set]

    X_fit = pd.concat([X_labeled] + pseudo_features, ignore_index=True)
    y_fit = pd.concat([y_labeled, pd.Series(pseudo_targets, dtype=float)], ignore_index=True)
    final_model = _make_model_for_data(
        X_fit, seed, numeric_features, categorical_features,
        n_estimators, min_samples_leaf,
    )
    final_model.fit(X_fit, y_fit)
    test_prediction = np.clip(final_model.predict(X_test), 0.0, 1.0)
    pool_prediction = (
        np.clip(final_model.predict(X_pool), 0.0, 1.0)
        if len(X_pool) else np.array([], dtype=float)
    )
    return test_prediction, pd.DataFrame(history), final_model, pool_prediction
