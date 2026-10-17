"""Shared preprocessing and model constructors (ported from models/modeling.ipynb v1)."""
import numpy as np
import optuna
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

TEXT_COL = "title"
CATEGORICAL_COLS = ["domain"]


def make_pipeline(model, feature_cols: list[str]) -> Pipeline:
    numeric = [c for c in feature_cols if c not in CATEGORICAL_COLS + [TEXT_COL]]
    transformers = [
        ("numeric", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
    ]
    if any(c in feature_cols for c in CATEGORICAL_COLS):
        transformers.append(
            ("categorical", OneHotEncoder(handle_unknown="infrequent_if_exist", max_categories=20, min_frequency=20), CATEGORICAL_COLS)
        )
    if TEXT_COL in feature_cols:
        transformers.append(
            ("text", TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=5, max_df=0.5, strip_accents="unicode"), TEXT_COL)
        )
    return Pipeline([("preprocess", ColumnTransformer(transformers)), ("model", model)])


def logistic() -> LogisticRegression:
    return LogisticRegression(class_weight="balanced", C=0.1, max_iter=2000)


def xgb_classifier(pos_weight: float = 1.0, **params) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic", eval_metric="aucpr", tree_method="hist",
        n_jobs=-1, random_state=42, scale_pos_weight=pos_weight, **params,
    )


def xgb_regressor(**params) -> XGBRegressor:
    return XGBRegressor(tree_method="hist", n_jobs=-1, random_state=42, **params)


def xgb_search_space(trial: optuna.Trial) -> dict:
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
    }


def pos_weight(y) -> float:
    return float((y == 0).sum() / max((y == 1).sum(), 1))


def tune_xgb(X, y, feature_cols: list[str], n_trials: int = 100, n_splits: int = 3, seed: int = 42) -> dict:
    """Optuna over `xgb_search_space`, scored by mean CV AUPRC on the training set only."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    def objective(trial):
        params = xgb_search_space(trial)
        scores = []
        for tr, va in cv.split(X, y):
            pipe = make_pipeline(xgb_classifier(pos_weight(y.iloc[tr]), **params), feature_cols)
            pipe.fit(X.iloc[tr], y.iloc[tr])
            scores.append(average_precision_score(y.iloc[va], pipe.predict_proba(X.iloc[va])[:, 1]))
        return float(np.mean(scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study.best_params
