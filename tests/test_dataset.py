import sqlite3

import numpy as np

import pytest

from hn import dataset, db, features, labels


@pytest.fixture(scope="module")
def df():
    return dataset.build(30, trajectory=True)


def test_at_most_k_positives_per_complete_day(df):
    positives = df.groupby("day")["top30"].sum()
    assert (positives <= 30).all() and (positives >= 25).all()  # a few top-30 stories were fetched late and dropped


def test_point_features_match_direct_sql(df):
    story_id = df.index[100]
    with sqlite3.connect(f"file:{db.DB_PATH}?mode=ro", uri=True) as conn:
        score = conn.execute(
            """SELECT s.score FROM snapshots s JOIN stories st USING (story_id)
               WHERE s.story_id = ? AND s.snapshot_ts - st.created_ts <= 1800
               ORDER BY s.snapshot_ts DESC LIMIT 1""",
            (int(story_id),),
        ).fetchone()[0]
    assert df.loc[story_id, "log_score"] == pytest.approx(np.log1p(score or 0))


def test_no_feature_peeks_past_t(df):
    assert df["obs_age_min"].max() <= 30
    assert df["min_to_first_comment"].max() <= 30


def test_every_row_has_a_label_and_is_on_a_complete_day(df):
    stories, snapshots = db.read_stories(), db.read_snapshots()
    final = labels.final_metrics(snapshots)
    complete = labels.labeled_days(stories, final)
    assert df["day"].isin(complete).all()
    assert len(df) <= stories["day"].isin(complete).sum()
    assert df.isna().sum().sum() == 0


def test_temporal_split_is_ordered(df):
    train, test = dataset.temporal_split(df)
    assert train["day"].max() < test["day"].min()
    assert test["day"].nunique() == dataset.TEST_DAYS
    assert len(train) + len(test) == len(df)


def test_stories_unobserved_by_t_are_excluded():
    stories, snapshots = db.read_stories(), db.read_snapshots()
    observed = features.point_features(snapshots, 10).index
    df10 = dataset.build(10)
    assert df10.index.isin(observed).all()
    assert len(df10) < len(dataset.build(360))  # more stories have been fetched by 6h than by 10min
