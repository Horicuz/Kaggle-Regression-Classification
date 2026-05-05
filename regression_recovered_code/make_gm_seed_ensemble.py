from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_auc_score
from xgboost import XGBRegressor


RANDOM_STATE = 42
TRAIN_PATH = Path("CC_education_economy_train.csv")
TEST_PATH = Path("CC_education_economy_test.csv")
PRIVATE_PATH = Path("CC_private_test.csv")

# Current best pseudo-label source. This is the 25% CatBoost + 75% LightGBM blend
# that scored 271,404,123.79 on Kaggle public.
PSEUDO_SOURCE_PATH = Path("submission_regression_final.csv")
SUMMARY_PATH = Path("gm_seed_ensemble_summary.json")

REGRESSION_TARGET = "salary"
CLASSIFICATION_TARGET = "vacation"
EXCLUDED = [REGRESSION_TARGET, CLASSIFICATION_TARGET, "total_days_worked"]

SEEDS = [42, 123, 777, 2024, 9999]
PSEUDO_WEIGHT = 0.42
TE_SMOOTHING = 24.0

EDUCATION_ORDER = {"High School": 0, "Diploma": 1, "Bachelor": 2, "Master": 3, "PhD": 4}
COMPANY_SIZE_ORDER = {"Startup": 0, "Small": 1, "Medium": 2, "Large": 3, "Enterprise": 4}
SKILL_BRACKET_ORDER = {"low": 0, "mid": 1, "high": 2}
INTERACTION_COLS = ["edu_company", "job_company", "loc_company", "edu_job"]

LGBM_PARAMS = {
    "objective": "regression",
    "n_estimators": 3600,
    "learning_rate": 0.018,
    "num_leaves": 63,
    "max_depth": -1,
    "min_child_samples": 55,
    "subsample": 0.86,
    "subsample_freq": 1,
    "colsample_bytree": 0.90,
    "reg_alpha": 0.35,
    "reg_lambda": 2.5,
    "n_jobs": -1,
    "verbosity": -1,
    "force_col_wise": True,
}

XGB_PARAMS = {
    "objective": "reg:squarederror",
    "n_estimators": 2600,
    "learning_rate": 0.018,
    "max_depth": 4,
    "min_child_weight": 18,
    "subsample": 0.86,
    "colsample_bytree": 0.88,
    "reg_alpha": 0.25,
    "reg_lambda": 5.0,
    "tree_method": "hist",
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "verbosity": 0,
}

CAT_PARAMS = {
    "loss_function": "RMSE",
    "iterations": 2200,
    "learning_rate": 0.022,
    "depth": 6,
    "l2_leaf_reg": 5.6,
    "random_seed": RANDOM_STATE,
    "verbose": False,
    "allow_writing_files": False,
    "thread_count": -1,
}


def iqr_bounds(series: pd.Series, q1=0.25, q3=0.75, multiplier=1.5) -> tuple[float, float]:
    q_low = series.quantile(q1)
    q_high = series.quantile(q3)
    iqr = q_high - q_low
    return float(q_low - multiplier * iqr), float(q_high + multiplier * iqr)


def add_base_features(df: pd.DataFrame, agg_lower: float, agg_upper: float) -> pd.DataFrame:
    out = df.copy()
    if "aggregated_score" in out.columns:
        out["aggregated_score"] = out["aggregated_score"].clip(agg_lower, agg_upper)
    if "remote_work" in out.columns:
        out["remote_work_missing"] = out["remote_work"].isna().astype("int8")
    if "education_level" in out.columns:
        out["education_level_ord"] = out["education_level"].map(EDUCATION_ORDER).astype(float)
    if "company_size" in out.columns:
        out["company_size_ord"] = out["company_size"].map(COMPANY_SIZE_ORDER).astype(float)
    if "skill_bracket" in out.columns:
        out["skill_bracket_ord"] = out["skill_bracket"].map(SKILL_BRACKET_ORDER).astype(float)
    if {"experience_years", "skills_count"}.issubset(out.columns):
        years = out["experience_years"].replace(0, np.nan)
        out["skills_per_year"] = (
            out["skills_count"] / years
        ).replace([np.inf, -np.inf], np.nan).fillna(out["skills_count"])
        out["experience_x_skills"] = out["experience_years"] * out["skills_count"]
    if {"experience_years", "certifications"}.issubset(out.columns):
        years = out["experience_years"].replace(0, np.nan)
        out["certifications_per_year"] = (
            out["certifications"] / years
        ).replace([np.inf, -np.inf], np.nan).fillna(out["certifications"])
    return out


def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if {"education_level", "company_size"}.issubset(out.columns):
        out["edu_company"] = out["education_level"].astype(str) + "_" + out["company_size"].astype(str)
    if {"job_title", "company_size"}.issubset(out.columns):
        out["job_company"] = out["job_title"].astype(str) + "_" + out["company_size"].astype(str)
    if {"location", "company_size"}.issubset(out.columns):
        out["loc_company"] = out["location"].astype(str) + "_" + out["company_size"].astype(str)
    if {"education_level", "job_title"}.issubset(out.columns):
        out["edu_job"] = out["education_level"].astype(str) + "_" + out["job_title"].astype(str)
    return out


class KFoldTargetEncodingPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, numeric_cols, categorical_cols, n_splits=5, smoothing=30.0, random_state=42):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.n_splits = n_splits
        self.smoothing = smoothing
        self.random_state = random_state

    def _as_frame(self, X):
        return X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

    def _category_series(self, X, col):
        if col in X.columns:
            return X[col].astype("object").where(X[col].notna(), "Missing").astype(str)
        return pd.Series("Missing", index=X.index, dtype="object")

    def _mapping(self, categories, y):
        tmp = pd.DataFrame({"category": categories, "target": np.asarray(y, dtype=float)})
        stats = tmp.groupby("category")["target"].agg(["mean", "count"])
        return (stats["mean"] * stats["count"] + self.global_mean_ * self.smoothing) / (
            stats["count"] + self.smoothing
        )

    def _numeric_frame(self, X):
        out = pd.DataFrame(index=X.index)
        for col in self.numeric_cols:
            values = pd.to_numeric(X[col], errors="coerce") if col in X.columns else pd.Series(np.nan, index=X.index)
            out[col] = values.fillna(self.numeric_medians_.get(col, 0.0))
        return out

    def fit(self, X, y):
        X = self._as_frame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.global_mean_ = float(y.mean())
        self.numeric_medians_ = {}
        for col in self.numeric_cols:
            median = float(pd.to_numeric(X[col], errors="coerce").median()) if col in X.columns else 0.0
            self.numeric_medians_[col] = 0.0 if np.isnan(median) else median
        self.category_maps_ = {
            col: self._mapping(self._category_series(X, col), y)
            for col in self.categorical_cols
        }
        return self

    def fit_transform(self, X, y=None, **fit_params):
        if y is None:
            raise ValueError("Target encoding needs y.")
        X = self._as_frame(X).reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        self.fit(X, y)

        out = self._numeric_frame(X)
        inner_cv = KFold(n_splits=min(self.n_splits, len(X)), shuffle=True, random_state=self.random_state)
        for col in self.categorical_cols:
            encoded = pd.Series(self.global_mean_, index=X.index, dtype=float)
            categories = self._category_series(X, col)
            for train_idx, valid_idx in inner_cv.split(X):
                mapping = self._mapping(categories.iloc[train_idx], y.iloc[train_idx])
                encoded.iloc[valid_idx] = categories.iloc[valid_idx].map(mapping).fillna(self.global_mean_)
            out[f"{col}_target_mean"] = encoded.astype(float)
        return out.to_numpy(dtype=float)

    def transform(self, X):
        X = self._as_frame(X).reset_index(drop=True)
        out = self._numeric_frame(X)
        for col in self.categorical_cols:
            categories = self._category_series(X, col)
            out[f"{col}_target_mean"] = categories.map(self.category_maps_[col]).fillna(self.global_mean_).astype(float)
        return out.to_numpy(dtype=float)


def make_one_hot_encoder():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def compute_adversarial_weights(X_local: pd.DataFrame, X_private: pd.DataFrame, num_cols, cat_cols) -> tuple[np.ndarray, dict]:
    X_adv = pd.concat([X_local, X_private], ignore_index=True)
    y_adv = np.r_[np.zeros(len(X_local), dtype=int), np.ones(len(X_private), dtype=int)]

    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
        ("onehot", make_one_hot_encoder()),
    ])
    pre = ColumnTransformer([
        ("num", numeric, num_cols),
        ("cat", categorical, cat_cols),
    ])
    clf = Pipeline([
        ("preprocess", pre),
        ("model", RandomForestClassifier(
            n_estimators=220,
            max_depth=14,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_adv, y_adv, test_size=0.25, stratify=y_adv, random_state=RANDOM_STATE
    )
    clf.fit(X_tr, y_tr)
    auc = roc_auc_score(y_va, clf.predict_proba(X_va)[:, 1])
    p_local = np.clip(clf.predict_proba(X_local)[:, 1], 0.05, 0.95)
    weights = np.sqrt(p_local / (1.0 - p_local))
    weights = np.clip(weights, 0.35, 3.0)
    weights = weights / np.mean(weights)
    return weights.astype(float), {
        "auc": float(auc),
        "mean": float(np.mean(weights)),
        "std": float(np.std(weights)),
        "min": float(np.min(weights)),
        "max": float(np.max(weights)),
    }


def save_submission(ids: pd.Series, pred: np.ndarray, path: str, reference: np.ndarray | None = None) -> dict:
    pred = np.asarray(pred, dtype=float)
    out = pd.DataFrame({"id": ids.reset_index(drop=True), "prediction": pred})
    out.to_csv(path, index=False)
    row = {
        "file": path,
        "mean": float(out["prediction"].mean()),
        "std": float(out["prediction"].std()),
        "min": float(out["prediction"].min()),
        "max": float(out["prediction"].max()),
    }
    if reference is not None:
        diff = pred - reference
        row.update({
            "diff_vs_reference_mean": float(np.mean(diff)),
            "diff_vs_reference_abs_mean": float(np.mean(np.abs(diff))),
            "corr_with_reference": float(np.corrcoef(reference, pred)[0, 1]),
        })
    return row


def main() -> None:
    start = time.time()
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    private_raw = pd.read_csv(PRIVATE_PATH)
    pseudo = pd.read_csv(PSEUDO_SOURCE_PATH)
    if not pseudo["id"].reset_index(drop=True).equals(private_raw["id"].reset_index(drop=True)):
        raise ValueError("Pseudo-label ids are not aligned with private test ids.")

    base_features = [col for col in train_raw.columns if col not in EXCLUDED]
    agg_lower, agg_upper = iqr_bounds(train_raw["aggregated_score"])
    local_raw = pd.concat([train_raw, test_raw], ignore_index=True)

    X_local = add_interactions(add_base_features(local_raw[base_features], agg_lower, agg_upper)).reset_index(drop=True)
    y_local = local_raw[REGRESSION_TARGET].reset_index(drop=True).astype(float)
    private_features = [col for col in base_features if col in private_raw.columns]
    X_private = add_interactions(add_base_features(private_raw[private_features], agg_lower, agg_upper)).reset_index(drop=True)
    y_pseudo = pseudo["prediction"].to_numpy(dtype=float)

    cat_cols = X_local.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    cat_cols = list(dict.fromkeys(cat_cols + [c for c in INTERACTION_COLS if c in X_local.columns]))
    for frame in [X_local, X_private]:
        for col in cat_cols:
            if col in frame.columns:
                frame[col] = frame[col].astype("object").where(frame[col].notna(), "Missing").astype(str)
    num_cols = [col for col in X_local.columns if col not in cat_cols]

    local_adv_weights, adv_summary = compute_adversarial_weights(X_local, X_private, num_cols, cat_cols)

    X_master = pd.concat([X_local, X_private], ignore_index=True)
    y_master = pd.Series(np.r_[y_local.to_numpy(dtype=float), y_pseudo], name="salary_or_pseudo")
    master_weights = np.r_[local_adv_weights, np.full(len(X_private), PSEUDO_WEIGHT, dtype=float)]

    preprocessor = KFoldTargetEncodingPreprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        n_splits=5,
        smoothing=TE_SMOOTHING,
        random_state=RANDOM_STATE,
    )
    X_train = preprocessor.fit_transform(X_master, y_master)
    X_test = preprocessor.transform(X_private)
    y_train_log = np.log1p(y_master.to_numpy(dtype=float))

    min_salary = float(y_local.min())
    max_salary = float(y_local.max())
    reference = y_pseudo

    lgbm_seed_preds = []
    for seed in SEEDS:
        print(f"Training LGBM seed {seed}...")
        model = LGBMRegressor(**{**LGBM_PARAMS, "random_state": seed})
        model.fit(X_train, y_train_log, sample_weight=master_weights)
        lgbm_seed_preds.append(np.expm1(model.predict(X_test)))
    pred_lgbm_ensembled = np.mean(lgbm_seed_preds, axis=0)

    xgb_seed_preds = []
    for seed in SEEDS:
        print(f"Training XGBoost seed {seed}...")
        xgb = XGBRegressor(**{**XGB_PARAMS, "random_state": seed})
        xgb.fit(X_train, y_train_log, sample_weight=master_weights, verbose=False)
        xgb_seed_preds.append(np.expm1(xgb.predict(X_test)))
    pred_xgboost = np.mean(xgb_seed_preds, axis=0)

    cat_seed_preds = []
    for seed in SEEDS:
        print(f"Training CatBoost seed {seed}...")
        cat = CatBoostRegressor(**{**CAT_PARAMS, "random_seed": seed})
        cat.fit(X_train, y_train_log, sample_weight=master_weights)
        cat_seed_preds.append(np.expm1(cat.predict(X_test)))
    pred_catboost = np.mean(cat_seed_preds, axis=0)

    pred_lgbm_ensembled = np.clip(pred_lgbm_ensembled, min_salary, max_salary)
    pred_xgboost = np.clip(pred_xgboost, min_salary, max_salary)
    pred_catboost = np.clip(pred_catboost, min_salary, max_salary)

    blend_60_25_15 = np.clip(
        0.60 * pred_lgbm_ensembled + 0.25 * pred_catboost + 0.15 * pred_xgboost,
        min_salary,
        max_salary,
    )
    blend_75_15_10 = np.clip(
        0.75 * pred_lgbm_ensembled + 0.15 * pred_catboost + 0.10 * pred_xgboost,
        min_salary,
        max_salary,
    )
    blend_85_10_05 = np.clip(
        0.85 * pred_lgbm_ensembled + 0.10 * pred_catboost + 0.05 * pred_xgboost,
        min_salary,
        max_salary,
    )
    blend_90_05_05 = np.clip(
        0.90 * pred_lgbm_ensembled + 0.05 * pred_catboost + 0.05 * pred_xgboost,
        min_salary,
        max_salary,
    )
    blend_92_04_04 = np.clip(
        0.92 * pred_lgbm_ensembled + 0.04 * pred_catboost + 0.04 * pred_xgboost,
        min_salary,
        max_salary,
    )
    blend_95_03_02 = np.clip(
        0.95 * pred_lgbm_ensembled + 0.03 * pred_catboost + 0.02 * pred_xgboost,
        min_salary,
        max_salary,
    )
    current_50_gm_50 = np.clip(0.50 * reference + 0.50 * blend_75_15_10, min_salary, max_salary)

    outputs = [
        {"kind": "lgbm_seed_ensemble", **save_submission(private_raw["id"], pred_lgbm_ensembled, "submission_gm_lgbm_seed_ensemble.csv", reference)},
        {"kind": "xgboost", **save_submission(private_raw["id"], pred_xgboost, "submission_gm_xgboost.csv", reference)},
        {"kind": "catboost", **save_submission(private_raw["id"], pred_catboost, "submission_gm_catboost.csv", reference)},
        {"kind": "requested_blend_60_25_15", **save_submission(private_raw["id"], blend_60_25_15, "submission_gm_blend_60lgbm_25cat_15xgb.csv", reference)},
        {"kind": "lgbm_heavy_blend_75_15_10", **save_submission(private_raw["id"], blend_75_15_10, "submission_gm_blend_75lgbm_15cat_10xgb.csv", reference)},
        {"kind": "lgbm_heavy_blend_85_10_05", **save_submission(private_raw["id"], blend_85_10_05, "submission_gm_blend_85lgbm_10cat_05xgb.csv", reference)},
        {"kind": "lgbm_heavy_blend_90_05_05", **save_submission(private_raw["id"], blend_90_05_05, "submission_gm_blend_90lgbm_05cat_05xgb.csv", reference)},
        {"kind": "lgbm_heavy_blend_92_04_04", **save_submission(private_raw["id"], blend_92_04_04, "submission_gm_blend_92lgbm_04cat_04xgb.csv", reference)},
        {"kind": "lgbm_heavy_blend_95_03_02", **save_submission(private_raw["id"], blend_95_03_02, "submission_gm_blend_95lgbm_03cat_02xgb.csv", reference)},
        {"kind": "current_best_50_gm_50", **save_submission(private_raw["id"], current_50_gm_50, "submission_gm_blend_current50_gm50.csv", reference)},
    ]

    summary = {
        "pseudo_source": str(PSEUDO_SOURCE_PATH),
        "target_transform": "np.log1p(y) for training; np.expm1(pred) for prediction",
        "sample_weight": {
            "local": "sqrt adversarial odds from RandomForest private-vs-local detector, clipped and mean-normalized",
            "pseudo_private": PSEUDO_WEIGHT,
            "adversarial_summary": adv_summary,
        },
        "lgbm_params": LGBM_PARAMS,
        "xgb_params": XGB_PARAMS,
        "cat_params": CAT_PARAMS,
        "seeds": SEEDS,
        "clip": {"min_salary": min_salary, "max_salary": max_salary},
        "outputs": outputs,
        "seconds": round(time.time() - start, 2),
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
