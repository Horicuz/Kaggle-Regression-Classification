from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import KFold, train_test_split

from final_submission_tuning import (
    RANDOM_STATE,
    KFoldTargetEncodingPreprocessor,
    add_regression_base_features,
    load_all_data,
)


N_TRIALS = 30
N_SPLITS = 5

TRAIN_WEIGHT_CLIP_PERCENTILE = 99.0
TRAIN_WEIGHT_P_EPS = 1e-6

ADVERSARIAL_VALIDATION_PARAMS = {
    "iterations": 700,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
}

SUMMARY_PATH = Path("feature_engineering_sqrt_optuna_summary.json")
BASE_SUBMISSION_PATH = Path("submission_feat_eng_sqrt_base.csv")
ALPHA_015_SUBMISSION_PATH = Path("submission_feat_eng_sqrt_alpha_015.csv")
ALPHA_025_SUBMISSION_PATH = Path("submission_feat_eng_sqrt_alpha_025.csv")
ALPHA_030_SUBMISSION_PATH = Path("submission_feat_eng_sqrt_alpha_030.csv")


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adaugă 4 coloane de interacțiune categorice:
    - edu_company: education_level + "_" + company_size
    - job_company: job_title + "_" + company_size
    - loc_company: location + "_" + company_size
    - edu_job: education_level + "_" + job_title
    """
    result = df.copy()

    if "education_level" in result.columns and "company_size" in result.columns:
        result["edu_company"] = (
            result["education_level"].astype(str) + "_" + result["company_size"].astype(str)
        )

    if "job_title" in result.columns and "company_size" in result.columns:
        result["job_company"] = (
            result["job_title"].astype(str) + "_" + result["company_size"].astype(str)
        )

    if "location" in result.columns and "company_size" in result.columns:
        result["loc_company"] = (
            result["location"].astype(str) + "_" + result["company_size"].astype(str)
        )

    if "education_level" in result.columns and "job_title" in result.columns:
        result["edu_job"] = (
            result["education_level"].astype(str) + "_" + result["job_title"].astype(str)
        )

    return result


def build_adversarial_classifier() -> CatBoostClassifier:
    return CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=ADVERSARIAL_VALIDATION_PARAMS["iterations"],
        learning_rate=ADVERSARIAL_VALIDATION_PARAMS["learning_rate"],
        depth=ADVERSARIAL_VALIDATION_PARAMS["depth"],
        l2_leaf_reg=ADVERSARIAL_VALIDATION_PARAMS["l2_leaf_reg"],
        auto_class_weights="Balanced",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )


def prepare_catboost_frame(df: pd.DataFrame, cat_cols: list[str]) -> pd.DataFrame:
    """Convertește ALL coloane din cat_cols la string (inclusiv cele cu float values)."""
    out = df.copy()
    for col in cat_cols:
        if col in out.columns:
            # Converteste la string, inlocuieste NaN cu "Missing"
            out[col] = out[col].fillna("Missing").astype(str)
    return out


def compute_sqrt_odds_weights(data: dict, cat_cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """
    Calculează greutăți folosind varianta amortizată sqrt(p / (1 - p)).
    """
    train_raw = data["train_raw"]
    test_raw = data["test_raw"]
    private_test_raw = data["private_test_raw"]
    base_features = data["base_features"]
    agg_lower = data["agg_lower"]
    agg_upper = data["agg_upper"]

    adv_features = [c for c in base_features if c in private_test_raw.columns]
    local_raw = pd.concat([train_raw[adv_features], test_raw[adv_features]], ignore_index=True)
    private_raw = private_test_raw[adv_features].copy()

    # Adaugă feature engineering pe raw data
    local_raw = add_interaction_features(local_raw)
    private_raw = add_interaction_features(private_raw)

    X_adv_raw = pd.concat([local_raw, private_raw], ignore_index=True)
    y_adv = pd.Series([0] * len(local_raw) + [1] * len(private_raw), name="is_private_test")
    X_adv_base = add_regression_base_features(X_adv_raw, agg_lower, agg_upper)

    X_adv_train, X_adv_valid, y_adv_train, y_adv_valid = train_test_split(
        X_adv_base,
        y_adv,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_adv,
    )

    cat_features = [c for c in cat_cols if c in X_adv_base.columns]

    fit_cat_features = [c for c in cat_cols if c in X_adv_base.columns]

    X_adv_train = prepare_catboost_frame(X_adv_train, fit_cat_features)
    X_adv_valid = prepare_catboost_frame(X_adv_valid, fit_cat_features)

    adversarial_model = build_adversarial_classifier()
    adversarial_model.fit(
        X_adv_train,
        y_adv_train,
        cat_features=fit_cat_features,
        eval_set=(X_adv_valid, y_adv_valid),
        use_best_model=False,
    )

    valid_prob = adversarial_model.predict_proba(X_adv_valid)[:, 1]
    adversarial_auc = roc_auc_score(y_adv_valid, valid_prob)

    # Pentru predict, selectezi DOAR coloanele care sunt în X_adv_train
    adv_train_cols = X_adv_train.columns.tolist()
    X_cv_for_adv = prepare_catboost_frame(data["X_cv_base"][adv_train_cols], fit_cat_features)
    X_private_for_adv = prepare_catboost_frame(data["X_private_base"][adv_train_cols], fit_cat_features)

    local_prob = adversarial_model.predict_proba(X_cv_for_adv)[:, 1]
    private_prob = adversarial_model.predict_proba(X_private_for_adv)[:, 1]

    local_prob = np.clip(local_prob, TRAIN_WEIGHT_P_EPS, 1.0 - TRAIN_WEIGHT_P_EPS)
    # SQRT variant pentru amortizare
    raw_odds = np.sqrt(local_prob / (1.0 - local_prob))
    raw_odds_cap = float(np.quantile(raw_odds, TRAIN_WEIGHT_CLIP_PERCENTILE / 100.0))
    train_weights = np.clip(raw_odds, None, raw_odds_cap).astype(float)

    summary = {
        "adversarial_auc": float(adversarial_auc),
        "train_weight_clip_percentile": TRAIN_WEIGHT_CLIP_PERCENTILE,
        "train_weight_cap": raw_odds_cap,
        "train_weight_mean": float(np.mean(train_weights)),
        "train_weight_std": float(np.std(train_weights)),
        "train_weight_min": float(np.min(train_weights)),
        "train_weight_median": float(np.median(train_weights)),
        "train_weight_max": float(np.max(train_weights)),
        "private_probability_mean": float(np.mean(private_prob)),
        "private_probability_std": float(np.std(private_prob)),
        "private_probability_min": float(np.min(private_prob)),
        "private_probability_median": float(np.median(private_prob)),
        "private_probability_max": float(np.max(private_prob)),
        "weight_formula": "sample_weight = sqrt(p / (1 - p)), clipped at the 99th percentile",
    }
    return train_weights, private_prob.astype(float), summary


def precompute_target_encoded_folds(data: dict, train_weights: np.ndarray, n_splits: int, cat_cols: list[str]) -> list[dict]:
    X_cv_base = data["X_cv_base"].reset_index(drop=True)
    y_cv = data["y_cv"].reset_index(drop=True)
    num_cols = data["num_cols"]

    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds: list[dict] = []

    for fold_id, (train_idx, valid_idx) in enumerate(outer_cv.split(X_cv_base), start=1):
        X_fold_train = X_cv_base.iloc[train_idx].reset_index(drop=True)
        y_fold_train = y_cv.iloc[train_idx].reset_index(drop=True)
        X_fold_valid = X_cv_base.iloc[valid_idx].reset_index(drop=True)
        y_fold_valid = y_cv.iloc[valid_idx].reset_index(drop=True)

        preprocessor = KFoldTargetEncodingPreprocessor(
            numeric_cols=num_cols,
            categorical_cols=cat_cols,
            n_splits=5,
            smoothing=30.0,
            random_state=RANDOM_STATE,
        )
        X_train_encoded = preprocessor.fit_transform(X_fold_train, y_fold_train)
        X_valid_encoded = preprocessor.transform(X_fold_valid)

        folds.append(
            {
                "fold": fold_id,
                "X_train": X_train_encoded,
                "y_train": y_fold_train.to_numpy(dtype=float),
                "train_weight": train_weights[train_idx],
                "X_valid": X_valid_encoded,
                "y_valid": y_fold_valid.to_numpy(dtype=float),
                "valid_weight": train_weights[valid_idx],
            }
        )
        print(f"Fold {fold_id} precomputat: train={X_train_encoded.shape}, valid={X_valid_encoded.shape}")

    return folds


def build_regressor(params: dict) -> TransformedTargetRegressor:
    regressor = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(params["iterations"]),
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
        thread_count=-1,
    )
    return TransformedTargetRegressor(
        regressor=regressor,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def objective_factory(folds: list[dict]):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 800, 2500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "depth": trial.suggest_int("depth", 4, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }

        fold_scores = []
        for fold_data in folds:
            model = build_regressor(params)
            model.fit(
                fold_data["X_train"],
                fold_data["y_train"],
                sample_weight=fold_data["train_weight"],
            )
            valid_pred = model.predict(fold_data["X_valid"])
            fold_scores.append(
                mean_squared_error(
                    fold_data["y_valid"],
                    valid_pred,
                    sample_weight=fold_data["valid_weight"],
                )
            )

        return float(np.mean(fold_scores))

    return objective


def fit_final_model(data: dict, train_weights: np.ndarray, best_params: dict, cat_cols: list[str]) -> tuple[np.ndarray, dict]:
    X_cv_base = data["X_cv_base"].reset_index(drop=True)
    y_cv = data["y_cv"].reset_index(drop=True)
    X_private_base = data["X_private_base"].reset_index(drop=True)
    num_cols = data["num_cols"]

    full_preprocessor = KFoldTargetEncodingPreprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        n_splits=5,
        smoothing=30.0,
        random_state=RANDOM_STATE,
    )
    X_train_encoded = full_preprocessor.fit_transform(X_cv_base, y_cv)
    X_private_encoded = full_preprocessor.transform(X_private_base)

    final_model = build_regressor(best_params)
    final_model.fit(X_train_encoded, y_cv.to_numpy(dtype=float), sample_weight=train_weights)

    base_private_pred = final_model.predict(X_private_encoded)
    shapes_info = {
        "train_encoded_shape": list(X_train_encoded.shape),
        "private_encoded_shape": list(X_private_encoded.shape),
    }
    return base_private_pred, shapes_info


def make_submission(ids: np.ndarray, prediction: np.ndarray, path: Path) -> dict:
    submission = pd.DataFrame({"id": ids, "prediction": prediction})
    submission.to_csv(path, index=False)
    return {
        "file": str(path),
        "rows": int(len(submission)),
        "prediction_mean": float(submission["prediction"].mean()),
        "prediction_std": float(submission["prediction"].std()),
        "prediction_min": float(submission["prediction"].min()),
        "prediction_median": float(submission["prediction"].median()),
        "prediction_max": float(submission["prediction"].max()),
    }


def build_alpha_adjusted_predictions(base_predictions: np.ndarray, private_prob: np.ndarray, alpha: float) -> np.ndarray:
    """
    Aplică post-procesare cu multiplicatorii 'Robin Hood':
    adjustment_factor = 1 + alpha * (rank_pct(private_probability) - 0.5)
    """
    rank_pct = pd.Series(private_prob).rank(method="average", pct=True).to_numpy()
    factor = 1.0 + alpha * (rank_pct - 0.5)
    return np.asarray(base_predictions, dtype=float) * factor


def main() -> None:
    # Încarcă datele și adaugă feature engineering
    data = load_all_data()

    # Adaugă interacțiuni pe datele brute (vor fi procesate în compute_sqrt_odds_weights)
    data["X_cv_base"] = add_interaction_features(data["X_cv_base"])
    data["X_private_base"] = add_interaction_features(data["X_private_base"])

    # Actualizează cat_cols cu noile coloane
    interaction_cols = ["edu_company", "job_company", "loc_company", "edu_job"]
    cat_cols = data["cat_cols"] + [col for col in interaction_cols if col in data["X_cv_base"].columns]
    num_cols = data["num_cols"]

    # Convertește coloanele categorice la string pentru a evita erorile CatBoost
    for col in cat_cols:
        if col in data["X_cv_base"].columns:
            data["X_cv_base"][col] = data["X_cv_base"][col].astype("object").where(
                data["X_cv_base"][col].notna(), "Missing"
            ).astype(str)
        if col in data["X_private_base"].columns:
            data["X_private_base"][col] = data["X_private_base"][col].astype("object").where(
                data["X_private_base"][col].notna(), "Missing"
            ).astype(str)

    print(f"Noi cat_cols cu interacțiuni: {len(cat_cols)} coloane")
    print(f"Interacțiuni adăugate: {[col for col in interaction_cols if col in data['X_cv_base'].columns]}")

    # Calculează greutăți cu sqrt odds
    train_weights, private_prob, adv_summary = compute_sqrt_odds_weights(data, cat_cols)

    # Precomputează K-Fold cu target encoding
    folds = precompute_target_encoded_folds(data, train_weights, N_SPLITS, cat_cols)

    # Lansează Optuna tuning (30 trials)
    print(f"\nLansez Optuna cu {N_TRIALS} trials...")
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE),
        study_name="feature_engineering_sqrt_optuna",
    )
    study.optimize(objective_factory(folds), n_trials=N_TRIALS)

    best_params = {
        "iterations": int(study.best_params["iterations"]),
        "learning_rate": float(study.best_params["learning_rate"]),
        "depth": int(study.best_params["depth"]),
        "l2_leaf_reg": float(study.best_params["l2_leaf_reg"]),
    }

    print(f"\nBest value (Weighted K-Fold MSE): {study.best_value:,.2f}")
    print(f"Best params: {best_params}")

    # Antrenează modelul final
    base_private_pred, shapes_info = fit_final_model(data, train_weights, best_params, cat_cols)

    private_test_raw = data["private_test_raw"]
    private_ids = private_test_raw["id"].to_numpy() if "id" in private_test_raw.columns else np.arange(1, len(base_private_pred) + 1)

    # Generează 4 submisii: base + 3 cu alpha
    print("\nGenerez submisii finale...")
    base_summary = make_submission(private_ids, base_private_pred, BASE_SUBMISSION_PATH)

    alpha_015_pred = build_alpha_adjusted_predictions(base_private_pred, private_prob, 0.015)
    alpha_015_summary = make_submission(private_ids, alpha_015_pred, ALPHA_015_SUBMISSION_PATH)

    alpha_025_pred = build_alpha_adjusted_predictions(base_private_pred, private_prob, 0.025)
    alpha_025_summary = make_submission(private_ids, alpha_025_pred, ALPHA_025_SUBMISSION_PATH)

    alpha_030_pred = build_alpha_adjusted_predictions(base_private_pred, private_prob, 0.030)
    alpha_030_summary = make_submission(private_ids, alpha_030_pred, ALPHA_030_SUBMISSION_PATH)

    summary = {
        "pipeline": "Feature Engineering + Sqrt Odds Weighting + Optuna",
        "feature_engineering": {
            "new_interactions": ["edu_company", "job_company", "loc_company", "edu_job"],
            "total_categorical_cols": len(cat_cols),
        },
        "study": {
            "n_trials": N_TRIALS,
            "best_value": float(study.best_value),
            "best_params": best_params,
            "best_trial_number": int(study.best_trial.number),
        },
        "adversarial_validation": adv_summary,
        "final_shapes": shapes_info,
        "outputs": [
            {"kind": "base", **base_summary},
            {"kind": "alpha_0.015", "alpha": 0.015, **alpha_015_summary},
            {"kind": "alpha_0.025", "alpha": 0.025, **alpha_025_summary},
            {"kind": "alpha_0.030", "alpha": 0.030, **alpha_030_summary},
        ],
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSalvat {SUMMARY_PATH}")
    print(f"\nSubmisii generate:")
    print(f"  - {BASE_SUBMISSION_PATH}")
    print(f"  - {ALPHA_015_SUBMISSION_PATH}")
    print(f"  - {ALPHA_025_SUBMISSION_PATH}")
    print(f"  - {ALPHA_030_SUBMISSION_PATH}")


if __name__ == "__main__":
    main()
