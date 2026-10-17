"""Read-only access to the collector's SQLite database."""
from functools import lru_cache
from pathlib import Path
import sqlite3

import pandas as pd

DB_PATH = Path(__file__).resolve().parents[2] / "data" / "hn.sqlite"


def connect(path: Path = DB_PATH) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


@lru_cache(maxsize=1)
def read_stories(path: Path = DB_PATH) -> pd.DataFrame:
    with connect(path) as conn:
        df = pd.read_sql(
            "SELECT story_id, created_ts, author, title, url, domain, is_dead FROM stories",
            conn,
            index_col="story_id",
        )
    df["created"] = pd.to_datetime(df["created_ts"], unit="s", utc=True)
    df["day"] = df["created"].dt.floor("D")
    return df


@lru_cache(maxsize=1)
def read_snapshots(path: Path = DB_PATH) -> pd.DataFrame:
    """One row per (story, fetch). `age_s` is seconds since the story was posted."""
    with connect(path) as conn:
        df = pd.read_sql(
            """SELECT s.story_id, s.snapshot_ts - st.created_ts AS age_s,
                      s.score, s.descendants, s.kids_count, s.is_dead
               FROM snapshots s JOIN stories st USING (story_id)
               ORDER BY s.story_id, s.snapshot_ts""",
            conn,
        )
    # The HN API omits score/descendants for killed or flagged items; that is zero engagement, not unknown.
    df[["score", "descendants"]] = df[["score", "descendants"]].fillna(0).astype(int)
    return df
