from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor

from final_submission_tuning import KFoldTargetEncodingPreprocessor, load_all_data
from feature_engineering_sqrt_optuna import (
    build_alpha_adjusted_predictions,
    compute_sqrt_odds_weights,
    add_interaction_features,
)


SEEDS = [42, 123, 777, 999, 2024]
ALPHAS = [0.015, 0.025, 0.030, 0.035, 0.040, 0.045, 0.050, 0.055]

BEST_PARAMS_SUMMARY_PATH = Path("feature_engineering_sqrt_optuna_summary.json")
OUTPUT_SUMMARY_PATH = Path("feature_engineering_sqrt_seedavg_summary.json")
BASE_OUTPUT_PATH = Path("submission_feat_eng_sqrt_seedavg_base.csv")


def alpha_to_tag(alpha: float) -> str:
    return f"{int(round(alpha * 1000)):03d}"


def load_best_params() -> dict:
    summary = json.loads(BEST_PARAMS_SUMMARY_PATH.read_text())
    best = summary["study"]["best_params"]
    return {
        "iterations": int(best["iterations"]),
        "learning_rate": float(best["learning_rate"]),
        "depth": int(best["depth"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
    }


def prepare_data() -> tuple[dict, list[str]]:
    data = load_all_data()

    data["X_cv_base"] = add_interaction_features(data["X_cv_base"])
    data["X_private_base"] = add_interaction_features(data["X_private_base"])

    interaction_cols = ["edu_company", "job_company", "loc_company", "edu_job"]
    cat_cols = data["cat_cols"] + [col for col in interaction_cols if col in data["X_cv_base"].columns]

    for col in cat_cols:
        if col in data["X_cv_base"].columns:
            data["X_cv_base"][col] = data["X_cv_base"][col].astype("object").where(
                data["X_cv_base"][col].notna(), "Missing"
            ).astype(str)
        if col in data["X_private_base"].columns:
            data["X_private_base"][col] = data["X_private_base"][col].astype("object").where(
                data["X_private_base"][col].notna(), "Missing"
            ).astype(str)

    return data, cat_cols


def build_regressor_for_seed(params: dict, seed: int) -> TransformedTargetRegressor:
    regressor = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(params["iterations"]),
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=int(seed),
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


def main() -> None:
    best_params = load_best_params()
    print(f"Best params loaded: {best_params}")

    data, cat_cols = prepare_data()

    train_weights, private_prob, adv_summary = compute_sqrt_odds_weights(data, cat_cols)

    X_cv_base = data["X_cv_base"].reset_index(drop=True)
    y_cv = data["y_cv"].reset_index(drop=True)
    X_private_base = data["X_private_base"].reset_index(drop=True)
    num_cols = data["num_cols"]

    full_preprocessor = KFoldTargetEncodingPreprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        n_splits=5,
        smoothing=30.0,
        random_state=42,
    )
    X_train_encoded = full_preprocessor.fit_transform(X_cv_base, y_cv)
    X_private_encoded = full_preprocessor.transform(X_private_base)

    seed_predictions = []
    seed_summaries = []

    for seed in SEEDS:
        print(f"Training seed={seed}...")
        model = build_regressor_for_seed(best_params, seed)
        model.fit(X_train_encoded, y_cv.to_numpy(dtype=float), sample_weight=train_weights)
        pred = model.predict(X_private_encoded)

        seed_predictions.append(pred)
        seed_summaries.append(
            {
                "seed": int(seed),
                "prediction_mean": float(np.mean(pred)),
                "prediction_std": float(np.std(pred)),
                "prediction_min": float(np.min(pred)),
                "prediction_median": float(np.median(pred)),
                "prediction_max": float(np.max(pred)),
            }
        )

    stacked = np.vstack(seed_predictions)
    mean_pred = np.mean(stacked, axis=0)

    private_test_raw = data["private_test_raw"]
    private_ids = private_test_raw["id"].to_numpy() if "id" in private_test_raw.columns else np.arange(1, len(mean_pred) + 1)

    outputs = []
    base_summary = make_submission(private_ids, mean_pred, BASE_OUTPUT_PATH)
    outputs.append({"kind": "base", **base_summary})

    for alpha in ALPHAS:
        adjusted = build_alpha_adjusted_predictions(mean_pred, private_prob, alpha)
        out_path = Path(f"submission_feat_eng_sqrt_seedavg_alpha_{alpha_to_tag(alpha)}.csv")
        alpha_summary = make_submission(private_ids, adjusted, out_path)
        outputs.append({"kind": f"alpha_{alpha}", "alpha": float(alpha), **alpha_summary})

    summary = {
        "pipeline": "Feature Engineering + Sqrt Odds + Seed Averaging",
        "seeds": SEEDS,
        "alphas": ALPHAS,
        "base_best_params": best_params,
        "adversarial_validation": adv_summary,
        "final_shapes": {
            "train_encoded_shape": list(X_train_encoded.shape),
            "private_encoded_shape": list(X_private_encoded.shape),
        },
        "seed_predictions": seed_summaries,
        "outputs": outputs,
    }

    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"\nSaved {OUTPUT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
