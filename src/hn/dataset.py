"""Assemble a modeling table for one observation time."""
import numpy as np
import pandas as pd

from . import db, features, labels

TEST_DAYS = 2
TARGETS = ["top30", "final_score", "log_final_score", "tier"]
STATIC = ["hour_utc", "dow_utc", "is_show_hn", "is_ask_hn", "url_missing", "domain", "title"]


def build(t_min: int, trajectory: bool = False) -> pd.DataFrame:
    """One row per labeled story on a complete day that was observed by t. Columns: features at t, targets, `day`."""
    stories, snapshots = db.read_stories(), db.read_snapshots()
    final = labels.final_metrics(snapshots)
    keep = stories[stories["day"].isin(labels.labeled_days(stories, final))]
    final = final.loc[final.index.intersection(keep.index)]

    df = features.static_features(keep)
    # Inner join drops stories the collector had not fetched by t. Those are a collection artifact
    # (late discovery), and they finish in the top 30 at ~10x the base rate, so keeping them with
    # zero-filled features would let the model learn the collector's behavior instead of the story's.
    df = df.join(features.point_features(snapshots, t_min), how="inner")
    if trajectory:
        df = df.join(features.trajectory_features(snapshots, t_min))

    df["top30"] = labels.top_k_daily(keep, final).astype(int)
    df["final_score"] = final["final_score"]
    df["log_final_score"] = np.log1p(df["final_score"])
    df["tier"] = labels.tiers(df["final_score"])
    df["day"] = keep["day"]
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c not in TARGETS + ["day"]]


def temporal_split(df: pd.DataFrame, test_days: int = TEST_DAYS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Last `test_days` complete days are the test set. Never tune on it."""
    days = sorted(df["day"].unique())
    cutoff = days[-test_days]
    return df[df["day"] < cutoff], df[df["day"] >= cutoff]
