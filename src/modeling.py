from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectFromModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop one of each pair of strongly correlated columns.

    The transformer is intentionally simple and sklearn-compatible so it can be
    stored inside a Pipeline and serialized with joblib.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_features = X.shape[1]
        if n_features <= 1:
            self.keep_mask_ = np.ones(n_features, dtype=bool)
            return self
        corr = np.corrcoef(X, rowvar=False)
        corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
        upper = np.triu(np.abs(corr), k=1)
        to_drop = np.zeros(n_features, dtype=bool)
        for j in range(n_features):
            if np.any(upper[:, j] > self.threshold):
                to_drop[j] = True
        self.keep_mask_ = ~to_drop
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return X[:, self.keep_mask_]


def regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) > 1 else np.nan,
    }


def make_regression_bins(y, n_bins: int = 5) -> np.ndarray:
    """Quantile bins for stratifying a continuous regression target."""
    y_series = pd.Series(np.asarray(y, dtype=float))
    for q in range(n_bins, 1, -1):
        try:
            bins = pd.qcut(y_series, q=q, labels=False, duplicates="drop")
            bins = np.asarray(bins, dtype=int)
            if len(np.unique(bins)) >= 2:
                return bins
        except ValueError:
            continue
    return np.zeros(len(y_series), dtype=int)


def make_rf_pipeline(
    random_state: int = 42,
    corr_threshold: float = 0.98,
    feature_selection: bool = True,
    n_estimators: int = 1000,
    criterion: str = "squared_error",
    max_depth=6,
    max_features=0.7,
    min_samples_leaf: int = 4,
    min_samples_split: int = 6,
    max_samples=0.9,
    ccp_alpha: float = 0.0,
    n_jobs: int = -1,
) -> Pipeline:
    """Build the final RandomForest pipeline used in the project."""
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("variance", VarianceThreshold()),
        ("corr", CorrelationFilter(threshold=corr_threshold)),
    ]
    if feature_selection:
        selector_rf = RandomForestRegressor(
            n_estimators=max(300, n_estimators // 2),
            criterion=criterion,
            max_depth=max_depth,
            max_features=max_features,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_samples=max_samples,
            ccp_alpha=ccp_alpha,
            bootstrap=True,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        steps.append(("select", SelectFromModel(selector_rf, threshold="median")))
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_leaf=min_samples_leaf,
        min_samples_split=min_samples_split,
        max_samples=max_samples,
        ccp_alpha=ccp_alpha,
        bootstrap=True,
        random_state=random_state,
        n_jobs=n_jobs,
    )
    steps.append(("model", rf))
    return Pipeline(steps)


DEFAULT_RF_PARAM_DISTRIBUTIONS = {
    "corr__threshold": [0.90, 0.95, 0.98, 0.995],
    "model__criterion": ["squared_error", "absolute_error"],
    "model__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
    "model__max_depth": [None, 4, 6, 8, 10],
    "model__min_samples_leaf": [2, 3, 4, 5, 6],
    "model__min_samples_split": [4, 6, 8, 10, 12],
    "model__max_samples": [0.65, 0.80, 0.90, None],
    "model__ccp_alpha": [0.0, 1e-4, 1e-3],
}


def nested_param_grid_for_selected() -> dict:
    grid = dict(DEFAULT_RF_PARAM_DISTRIBUTIONS)
    grid.update({
        "select__threshold": ["median", "mean", "0.75*mean", "1.25*mean"],
        "select__estimator__max_features": ["sqrt", 0.5, 0.8],
        "select__estimator__min_samples_leaf": [1, 2, 4],
    })
    return grid
