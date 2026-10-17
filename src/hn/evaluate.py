"""Metrics with uncertainty, for a held-out test set."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def bootstrap_ci(y, p, metric=average_precision_score, n: int = 1000, seed: int = 0, alpha: float = 0.05) -> tuple[float, float, float]:
    """(point, lower, upper) by resampling test rows."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    draws = np.array([metric(y[i], p[i]) for i in idx if y[i].any()])
    return metric(y, p), *np.quantile(draws, [alpha / 2, 1 - alpha / 2])


def paired_bootstrap_diff(y, p_a, p_b, metric=average_precision_score, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """metric(b) - metric(a) with CI; same resampled rows for both models."""
    y, p_a, p_b = map(np.asarray, (y, p_a, p_b))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    draws = np.array([metric(y[i], p_b[i]) - metric(y[i], p_a[i]) for i in idx if y[i].any()])
    return metric(y, p_b) - metric(y, p_a), *np.quantile(draws, [0.025, 0.975])


def precision_at_k_daily(day: pd.Series, y, p, k: int = 30) -> float:
    """If you may only pick k stories per day, what fraction of picks are positives?"""
    df = pd.DataFrame({"day": day.values, "y": np.asarray(y), "p": np.asarray(p)})
    picks = df.sort_values("p", ascending=False).groupby("day").head(k)
    return picks["y"].mean()


def summary(y, p, day: pd.Series | None = None) -> dict:
    out = {
        "auprc": bootstrap_ci(y, p),
        "auroc": bootstrap_ci(y, p, roc_auc_score),
        # Brier only makes sense for probabilities; a raw ranking score (e.g. log score) gets NaN.
        "brier": brier_score_loss(y, p) if 0 <= np.min(p) and np.max(p) <= 1 else np.nan,
    }
    if day is not None:
        out["precision@30/day"] = precision_at_k_daily(day, y, p)
    return out


def calibration_table(y, p, bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bin"] = pd.qcut(df["p"], bins, duplicates="drop")
    return df.groupby("bin", observed=True).agg(mean_pred=("p", "mean"), frac_pos=("y", "mean"), n=("y", "size"))
