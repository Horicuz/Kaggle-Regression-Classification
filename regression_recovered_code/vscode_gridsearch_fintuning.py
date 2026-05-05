from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.metrics import mean_squared_error

from final_submission_tuning import KFoldTargetEncodingPreprocessor, load_all_data
from feature_engineering_sqrt_optuna import (
    add_interaction_features,
    build_alpha_adjusted_predictions,
    compute_sqrt_odds_weights,
)


BEST_PARAMS_SUMMARY_PATH = Path("feature_engineering_sqrt_optuna_summary.json")
OUTPUT_SUMMARY_PATH = Path("gridsearch_fintuning_summary.json")
BASE_OUTPUT_PATH = Path("submission_gridsearch_base.csv")

# Grid search alphas for final submissions
SUBMISSION_ALPHAS = [0.0, 0.10, 0.20, 0.30]


def load_best_params() -> dict:
    summary = json.loads(BEST_PARAMS_SUMMARY_PATH.read_text())
    best = summary["study"]["best_params"]
    return {
        "iterations": int(best["iterations"]),
        "learning_rate": float(best["learning_rate"]),
        "depth": int(best["depth"]),
        "l2_leaf_reg": float(best["l2_leaf_reg"]),
    }


def build_param_grid(best_params: dict) -> dict:
    """
    Build 3x3x3 grid around best parameters:
    - learning_rate: [BEST * 0.9, BEST, BEST * 1.1]
    - l2_leaf_reg: [BEST - 1.0, BEST, BEST + 1.0]
    - iterations: [BEST - 100, BEST, BEST + 100]
    - depth: fixed to best value
    """
    best_lr = best_params["learning_rate"]
    best_l2 = best_params["l2_leaf_reg"]
    best_iter = best_params["iterations"]
    best_depth = best_params["depth"]

    lr_values = [best_lr * 0.9, best_lr, best_lr * 1.1]
    l2_values = [
        max(1.0, best_l2 - 1.0),
        best_l2,
        best_l2 + 1.0,
    ]
    iter_values = [best_iter - 100, best_iter, best_iter + 100]

    return {
        "learning_rate": lr_values,
        "l2_leaf_reg": l2_values,
        "iterations": iter_values,
        "depth": [best_depth],
    }


def prepare_data() -> tuple[dict, list[str]]:
    """Load and preprocess data with feature engineering."""
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


def precompute_target_encoded_folds(data: dict, train_weights: np.ndarray, n_splits: int, cat_cols: list[str]) -> list[dict]:
    """Precompute target-encoded K-Fold splits."""
    X_cv_base = data["X_cv_base"].reset_index(drop=True)
    y_cv = data["y_cv"].reset_index(drop=True)
    num_cols = data["num_cols"]

    from sklearn.model_selection import KFold
    outer_cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
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
            random_state=42,
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
    """Build CatBoost regressor with log1p/expm1 transform."""
    regressor = CatBoostRegressor(
        loss_function="RMSE",
        iterations=int(params["iterations"]),
        learning_rate=float(params["learning_rate"]),
        depth=int(params["depth"]),
        l2_leaf_reg=float(params["l2_leaf_reg"]),
        random_seed=42,
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


def evaluate_params_on_folds(params: dict, folds: list[dict]) -> tuple[float, list[float]]:
    """Evaluate parameter set on K-Fold, return mean MSE and fold MSEs."""
    fold_mses = []
    for fold_data in folds:
        model = build_regressor(params)
        model.fit(
            fold_data["X_train"],
            fold_data["y_train"],
            sample_weight=fold_data["train_weight"],
        )
        y_pred = model.predict(fold_data["X_valid"])
        fold_mse = mean_squared_error(
            fold_data["y_valid"],
            y_pred,
            sample_weight=fold_data["valid_weight"],
        )
        fold_mses.append(fold_mse)

    mean_mse = float(np.mean(fold_mses))
    return mean_mse, fold_mses


def make_submission(ids: np.ndarray, prediction: np.ndarray, path: Path) -> dict:
    """Save submission CSV and return summary stats."""
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
    print("Încalc best params din Optuna...")
    best_params = load_best_params()
    print(f"Best params: {best_params}")

    print("\nPrep data și feature engineering...")
    data, cat_cols = prepare_data()

    print("\nCompute adversarial weights...")
    train_weights, private_prob, adv_summary = compute_sqrt_odds_weights(data, cat_cols)

    print("\nPrecompute K-Fold folds...")
    folds = precompute_target_encoded_folds(data, train_weights, 5, cat_cols)

    print("\nBuild param grid (27 combinații)...")
    param_grid = build_param_grid(best_params)
    
    print(f"Learning rate range: {param_grid['learning_rate']}")
    print(f"L2 leaf reg range: {param_grid['l2_leaf_reg']}")
    print(f"Iterations range: {param_grid['iterations']}")
    print(f"Depth: {param_grid['depth']}")

    # Generate all combinations
    param_combinations = []
    for lr, l2, iters, depth in itertools.product(
        param_grid["learning_rate"],
        param_grid["l2_leaf_reg"],
        param_grid["iterations"],
        param_grid["depth"],
    ):
        param_combinations.append({
            "learning_rate": lr,
            "l2_leaf_reg": l2,
            "iterations": iters,
            "depth": depth,
        })

    print(f"\nEvaluez {len(param_combinations)} combinații...")
    
    results = []
    best_mean_mse = float("inf")
    best_combo_idx = -1

    for idx, params in enumerate(param_combinations, start=1):
        mean_mse, fold_mses = evaluate_params_on_folds(params, folds)
        results.append({
            "idx": idx,
            "params": params,
            "mean_weighted_mse": mean_mse,
            "fold_mses": fold_mses,
        })
        
        print(f"Combo {idx:2d}/{len(param_combinations)}: MSE={mean_mse:,.2f} | LR={params['learning_rate']:.6f} L2={params['l2_leaf_reg']:.3f} Iter={params['iterations']}")
        
        if mean_mse < best_mean_mse:
            best_mean_mse = mean_mse
            best_combo_idx = idx - 1

    best_result = results[best_combo_idx]
    best_result_params = best_result["params"]

    print(f"\n{'='*80}")
    print(f"BEST COMBO (#{best_result['idx']}): Weighted MSE = {best_result['mean_weighted_mse']:,.2f}")
    print(f"Params: {best_result_params}")
    print(f"Fold MSEs: {[f'{x:,.0f}' for x in best_result['fold_mses']]}")
    print(f"{'='*80}")

    # Train final model on all data using best params
    print("\nTrain final model on all data...")
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

    final_model = build_regressor(best_result_params)
    final_model.fit(X_train_encoded, y_cv.to_numpy(dtype=float), sample_weight=train_weights)

    base_private_pred = final_model.predict(X_private_encoded)

    private_test_raw = data["private_test_raw"]
    private_ids = private_test_raw["id"].to_numpy() if "id" in private_test_raw.columns else np.arange(1, len(base_private_pred) + 1)

    # Generate submissions
    print("\nGenerate submissions with alphas...")
    outputs = []
    
    base_summary = make_submission(private_ids, base_private_pred, Path("submission_gridsearch_base.csv"))
    outputs.append({"kind": "base", "alpha": 0.0, **base_summary})
    
    for alpha in [0.10, 0.20, 0.30]:
        alpha_tag = f"{int(round(alpha * 100)):03d}"
        adjusted = build_alpha_adjusted_predictions(base_private_pred, private_prob, alpha)
        out_path = Path(f"submission_gridsearch_alpha_{alpha_tag}.csv")
        alpha_summary = make_submission(private_ids, adjusted, out_path)
        outputs.append({"kind": f"alpha_{alpha}", "alpha": float(alpha), **alpha_summary})

    summary = {
        "pipeline": "Grid Search Local Fine-Tuning on Optuna Best",
        "grid_info": {
            "n_combinations": len(param_combinations),
            "learning_rate_range": param_grid["learning_rate"],
            "l2_leaf_reg_range": param_grid["l2_leaf_reg"],
            "iterations_range": param_grid["iterations"],
            "depth_fixed": param_grid["depth"][0],
        },
        "best_combo": {
            "rank": best_result["idx"],
            "weighted_mean_mse": best_result["mean_weighted_mse"],
            "fold_mses": best_result["fold_mses"],
            "params": best_result_params,
        },
        "all_results_ranked": sorted(
            [
                {
                    "rank": r["idx"],
                    "mean_mse": r["mean_weighted_mse"],
                    "params": r["params"],
                }
                for r in results
            ],
            key=lambda x: x["mean_mse"],
        ),
        "final_shapes": {
            "train_encoded_shape": list(X_train_encoded.shape),
            "private_encoded_shape": list(X_private_encoded.shape),
        },
        "adversarial_validation": adv_summary,
        "outputs": outputs,
    }

    OUTPUT_SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"\nSaved {OUTPUT_SUMMARY_PATH}")
    print(f"\nSubmissions generated:")
    for out in outputs:
        print(f"  - {out['file']}")


if __name__ == "__main__":
    main()
