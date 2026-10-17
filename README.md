# Early Engagement Prediction on Hacker News

Predicts whether a new Hacker News story will be in its day's top 30 by score 24 hours after
posting, using only what is known 30 minutes after posting.

## Pipeline

1. **Collect** (`data/main.py`). An asyncio collector polls the Hacker News API
   (`/v0/newstories`) every 10 minutes, stores each new story, and schedules two snapshots per
   story: one at +30 minutes (features) and one at +24 hours (label). State is in SQLite
   (`data/hn.sqlite`) so the collector can crash and resume without losing scheduled snapshots.
2. **Warehouse** (`data/export.sh`). The `stories`, `snapshots` and `snapshot_targets` tables are
   exported to CSV and loaded into Snowflake, where the label and the training view
   (`HN_DB.PUBLIC.TRAIN_DATASET_V`) are built. The Snowflake SQL is not in this repository.
3. **Model** (`models/`). `eda.ipynb` explores the features; `modeling.ipynb` trains the model.

## Data

- 4,067 stories posted Feb 4-9, 2026 (UTC); 133 (3.3%) are positive.
- Label `Y_TOP30_DAILY`: the story ranks in the top 30 of its UTC posting day by score at 24 hours.
- Features at 30 minutes: log(1 + score), log(1 + comments), log(1 + direct replies), posting hour
  and day of week, Show HN and Ask HN flags, missing-URL flag, and TF-IDF over the title (1-2 grams,
  `min_df=5`).

## Results

XGBoost (`scale_pos_weight` set per fold for the class imbalance) with hyperparameters from a
500-trial Optuna search. The objective is mean average precision (AUPRC) over 3-fold stratified
cross-validation.

| model | 3-fold CV AUPRC | positive rate (random-ranking AUPRC) |
| --- | --- | --- |
| XGBoost + TF-IDF title features | 0.349 | 0.033 |

The CV AUPRC is the best value found by the search, so it is optimistic: the folds used to pick the
hyperparameters are the ones it is reported on. There is no held-out test set yet, and no linear
baseline has been run. The notebook's final cell scores the model on its own training data; those
numbers are not a performance estimate.

EDA findings (`models/eda.ipynb`): positive stories average 12.3 points and 3.3 comments at 30
minutes against 1.7 and 0.4 for the rest. Author identity and posting frequency do not correlate with
the label (r = -0.02). "Show HN" and "Ask HN" titles correlate slightly negatively.

## Run

Collector (Python 3.12, `aiohttp`):

```bash
cd data
pip install aiohttp
bash collect.sh      # python main.py --db hn.sqlite --poll-seconds 600 ...; see main.py --help
bash export.sh       # writes stories.csv, snapshots.csv, snapshot_targets.csv
```

Load the CSVs into Snowflake and build `TRAIN_DATASET_V`. The notebooks read Snowflake credentials
from `.env`:

```
SF_USERNAME=...
SF_PASSWORD=...
SF_ACCOUNT=...
```

Notebooks:

```bash
pip install pandas numpy scikit-learn xgboost optuna seaborn snowflake-connector-python python-dotenv jupyter
jupyter notebook models/
```

## Next steps

- Hold out the last day of stories as a test set and report AUPRC there.
- Add a logistic-regression baseline.
- Add the Snowflake SQL that builds the label and training view.
- Add earlier observation points (5, 10, 15 minutes) to measure how prediction quality grows with
  observation time.
