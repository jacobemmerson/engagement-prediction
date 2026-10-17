"""Outcome labels computed from the +24h snapshot."""
import numpy as np
import pandas as pd

HORIZON_S = 24 * 3600
TOLERANCE_S = 1800
TIER_EDGES = [-np.inf, 5, 30, 100, np.inf]
TIER_NAMES = ["<5", "5-30", "30-100", "100+"]


def final_metrics(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Score/comments at the snapshot nearest +24h. Stories with no snapshot within tolerance are dropped."""
    dist = (snapshots["age_s"] - HORIZON_S).abs()
    near = snapshots[dist <= TOLERANCE_S]
    best = dist[near.index].groupby(near["story_id"]).idxmin()
    out = near.loc[best.values, ["story_id", "score", "descendants", "kids_count"]].set_index("story_id")
    return out.rename(columns=lambda c: f"final_{c}")


def labeled_days(stories: pd.DataFrame, final: pd.DataFrame) -> pd.DatetimeIndex:
    """UTC days that are fully inside the collection window and fully labeled.

    The first day is dropped because collection started mid-day, so its "top 30" is over a partial
    day. Any day with a story lacking a 24h snapshot is dropped for the same reason.
    """
    has_label = stories.index.isin(final.index)
    coverage = pd.Series(has_label, index=stories.index).groupby(stories["day"]).mean()
    complete = coverage[coverage == 1.0].index
    return complete[1:] if len(complete) and complete[0] == stories["day"].min() else complete


def top_k_daily(stories: pd.DataFrame, final: pd.DataFrame, k: int = 30) -> pd.Series:
    """True if the story is in the top-k of its UTC posting day by 24h score."""
    df = stories[["day"]].join(final["final_score"], how="inner")
    rank = df.groupby("day")["final_score"].rank(method="first", ascending=False)
    return (rank <= k).rename("top30")


def tiers(final_score: pd.Series) -> pd.Series:
    return pd.cut(final_score, TIER_EDGES, labels=range(len(TIER_NAMES)), right=False).astype(int).rename("tier")
