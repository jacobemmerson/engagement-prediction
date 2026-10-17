"""Features available at observation time `t` (minutes after posting)."""
import numpy as np
import pandas as pd

COUNT_COLS = ["score", "descendants", "kids_count"]


def state_at(snapshots: pd.DataFrame, age_s: float) -> pd.DataFrame:
    """Latest snapshot per story taken no later than `age_s` after posting. Stories not yet fetched are absent."""
    early = snapshots[snapshots["age_s"] <= age_s]
    return early.groupby("story_id").last()


def point_features(snapshots: pd.DataFrame, t_min: int) -> pd.DataFrame:
    """log1p counts at the latest snapshot <= t. Stories not yet fetched by t are absent."""
    state = state_at(snapshots, t_min * 60)
    return np.log1p(state[COUNT_COLS]).rename(columns=lambda c: f"log_{c}")


def trajectory_features(snapshots: pd.DataFrame, t_min: int) -> pd.DataFrame:
    """Shape of the score/comment curve up to t, beyond its endpoint."""
    t_s = t_min * 60
    window = snapshots[snapshots["age_s"] <= t_s]
    now = state_at(window, t_s)
    score_at = lambda age: state_at(window, age)["score"].reindex(now.index).fillna(1)  # HN stories start at 1 point

    out = pd.DataFrame(index=now.index)
    out["n_snapshots"] = window.groupby("story_id").size()
    out["obs_age_min"] = now["age_s"] / 60
    out["velocity_10m"] = now["score"] - score_at(t_s - 600)
    out["velocity_30m"] = now["score"] - score_at(t_s - 1800)
    out["log_score_half"] = np.log1p(score_at(t_s / 2))
    grouped = window.groupby("story_id")
    first_point = window[window["score"] > 1].groupby("story_id")["age_s"].min()
    first_comment = window[window["descendants"] > 0].groupby("story_id")["age_s"].min()
    # Censored at t when the event has not happened yet.
    out["min_to_first_point"] = first_point.reindex(now.index).fillna(t_s) / 60
    out["min_to_first_comment"] = first_comment.reindex(now.index).fillna(t_s) / 60
    out["comments_per_point"] = now["descendants"] / now["score"].clip(lower=1)
    return out


def static_features(stories: pd.DataFrame) -> pd.DataFrame:
    title = stories["title"].fillna("")
    return pd.DataFrame(
        {
            "hour_utc": stories["created"].dt.hour,
            "dow_utc": stories["created"].dt.dayofweek,
            "is_show_hn": title.str.startswith("Show HN").astype(int),
            "is_ask_hn": title.str.startswith("Ask HN").astype(int),
            "url_missing": stories["url"].isna().astype(int),
            "domain": stories["domain"].fillna("none"),
            "title": title,
        },
        index=stories.index,
    )
