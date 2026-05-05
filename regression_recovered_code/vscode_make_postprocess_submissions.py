from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from final_submission_tuning import (
    RANDOM_STATE,
    add_regression_base_features,
    load_all_data,
    make_adv_preprocessor,
)


NORMAL_SUBMISSION = Path("submission_opt_final_normal.csv")
WEIGHTED_SUBMISSION = Path("submission_opt_final_weighted.csv")
SUMMARY_PATH = Path("postprocess_submission_summary.json")


BLEND_CONFIGS = [
]

ADVERSARIAL_ALPHA_CONFIGS = [
    ("submission_weighted_advprob_centered_alpha_0025.csv", 0.025),
    ("submission_weighted_advprob_centered_alpha_0030.csv", 0.030),
    ("submission_weighted_advprob_centered_alpha_0035.csv", 0.035),

]


def validate_submission_pair(normal: pd.DataFrame, weighted: pd.DataFrame):
    if list(normal.columns) != list(weighted.columns):
        raise ValueError("Submission-urile nu au aceleasi coloane.")
    if "id" in normal.columns and not normal["id"].equals(weighted["id"]):
        raise ValueError("ID-urile din normal si weighted nu sunt aliniate.")
    if "prediction" not in normal.columns:
        raise ValueError("Lipseste coloana prediction.")


def build_private_adversarial_probability():
    data = load_all_data()
    train_raw = data["train_raw"]
    test_raw = data["test_raw"]
    private_test_raw = data["private_test_raw"]
    base_features = data["base_features"]
    agg_lower = data["agg_lower"]
    agg_upper = data["agg_upper"]
    num_cols = data["num_cols"]
    tree_cat_cols = data["tree_cat_cols"]

    adv_features = [c for c in base_features if c in private_test_raw.columns]
    adv_local_raw = pd.concat([train_raw[adv_features], test_raw[adv_features]], ignore_index=True)
    adv_private_raw = private_test_raw[adv_features].copy()

    X_adv_raw = pd.concat([adv_local_raw, adv_private_raw], ignore_index=True)
    y_adv = pd.Series([0] * len(adv_local_raw) + [1] * len(adv_private_raw), name="is_private_test")
    X_adv_base = add_regression_base_features(X_adv_raw, agg_lower, agg_upper)

    X_adv_train, X_adv_valid, y_adv_train, y_adv_valid = train_test_split(
        X_adv_base,
        y_adv,
        test_size=0.25,
        random_state=RANDOM_STATE,
        stratify=y_adv,
    )

    adversarial_model = Pipeline([
        ("preprocess", make_adv_preprocessor(num_cols, tree_cat_cols)),
        ("model", RandomForestClassifier(
            n_estimators=160,
            max_depth=12,
            min_samples_leaf=20,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])
    adversarial_model.fit(X_adv_train, y_adv_train)
    valid_prob = adversarial_model.predict_proba(X_adv_valid)[:, 1]
    auc = roc_auc_score(y_adv_valid, valid_prob)

    private_prob = adversarial_model.predict_proba(data["X_private_base"])[:, 1]
    private_rank = pd.Series(private_prob).rank(method="average", pct=True).to_numpy()
    centered_rank = private_rank - 0.5

    summary = {
        "adversarial_auc": float(auc),
        "private_probability": {
            "mean": float(np.mean(private_prob)),
            "std": float(np.std(private_prob)),
            "min": float(np.min(private_prob)),
            "q25": float(np.quantile(private_prob, 0.25)),
            "median": float(np.median(private_prob)),
            "q75": float(np.quantile(private_prob, 0.75)),
            "max": float(np.max(private_prob)),
        },
        "private_rank_centering": "factor = 1 + alpha * (rank_pct(private_probability) - 0.5)",
    }
    return private_prob, centered_rank, summary


def save_submission_like(template: pd.DataFrame, prediction: np.ndarray, path: Path):
    out = template.copy()
    out["prediction"] = prediction
    out.to_csv(path, index=False)
    return {
        "file": str(path),
        "mean": float(out["prediction"].mean()),
        "std": float(out["prediction"].std()),
        "min": float(out["prediction"].min()),
        "median": float(out["prediction"].median()),
        "max": float(out["prediction"].max()),
    }


def main():
    normal = pd.read_csv(NORMAL_SUBMISSION)
    weighted = pd.read_csv(WEIGHTED_SUBMISSION)
    validate_submission_pair(normal, weighted)

    normal_pred = normal["prediction"].to_numpy(dtype=float)
    weighted_pred = weighted["prediction"].to_numpy(dtype=float)

    outputs = []

    for filename, weighted_ratio in BLEND_CONFIGS:
        normal_ratio = 1.0 - weighted_ratio
        blended = weighted_ratio * weighted_pred + normal_ratio * normal_pred
        outputs.append({
            "kind": "blend",
            "weighted_ratio": weighted_ratio,
            "normal_ratio": normal_ratio,
            **save_submission_like(weighted, blended, Path(filename)),
        })

    private_prob, centered_rank, adversarial_summary = build_private_adversarial_probability()

    for filename, alpha in ADVERSARIAL_ALPHA_CONFIGS:
        factor = 1.0 + alpha * centered_rank
        adjusted = weighted_pred * factor
        outputs.append({
            "kind": "adversarial_probability_centered",
            "alpha": alpha,
            "factor_min": float(factor.min()),
            "factor_mean": float(factor.mean()),
            "factor_max": float(factor.max()),
            **save_submission_like(weighted, adjusted, Path(filename)),
        })

    summary = {
        "source_files": {
            "normal": str(NORMAL_SUBMISSION),
            "weighted": str(WEIGHTED_SUBMISSION),
        },
        "adversarial_summary": adversarial_summary,
        "outputs": outputs,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    print(f"Saved {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
