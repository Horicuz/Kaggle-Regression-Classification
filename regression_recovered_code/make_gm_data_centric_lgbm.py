from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

import make_gm_seed_ensemble as gm


OUT_SUMMARY = Path("gm_data_centric_lgbm_v2_summary.json")
SEEDS = [42, 123, 777, 2024, 9999]


def unique_edges(values: pd.Series, quantiles: np.ndarray) -> np.ndarray:
    edges = values.quantile(quantiles).to_numpy(dtype=float).copy()
    edges[0] = -np.inf
    edges[-1] = np.inf
    return np.unique(edges)


def cut_feature(values: pd.Series, edges: np.ndarray, prefix: str) -> pd.Series:
    codes = pd.cut(values, bins=edges, labels=False, include_lowest=True, duplicates="drop")
    return codes.astype("Int64").astype(str).radd(prefix)


def experience_band(values: pd.Series) -> pd.Series:
    exp = pd.to_numeric(values, errors="coerce")
    band = np.select(
        [
            exp.isna(),
            exp <= 3,
            exp <= 8,
            exp <= 14,
            exp >= 15,
        ],
        ["exp_missing", "Junior", "Mid", "Senior", "Expert"],
        default="exp_missing",
    )
    return pd.Series(band, index=values.index).astype(str)


def certification_bucket(values: pd.Series) -> pd.Series:
    cert = pd.to_numeric(values, errors="coerce")
    rounded = np.rint(cert.clip(0, 5)).astype("Int64").astype(str)
    bucket = "cert_" + rounded
    bucket = bucket.where(cert.notna(), "cert_missing")
    bucket = bucket.where(cert >= 0, "cert_negative")
    return bucket.astype(str)


def add_data_centric_features(
    df: pd.DataFrame,
    agg_lower: float,
    agg_upper: float,
    decile_edges: np.ndarray,
    ventile_edges: np.ndarray,
    skills_decile_edges: np.ndarray,
    skills_tail_thresholds: dict[str, float],
    tail_thresholds: dict[str, float],
    drop_numeric_agg: bool,
    drop_raw_feature_cols: tuple[str, ...] = (),
) -> pd.DataFrame:
    out = gm.add_base_features(df, agg_lower, agg_upper)
    if "aggregated_score" not in df.columns:
        return gm.add_interactions(out)

    agg_raw = pd.to_numeric(df["aggregated_score"], errors="coerce")
    agg_clipped = agg_raw.clip(agg_lower, agg_upper)

    out["aggregated_score_decile"] = cut_feature(agg_raw, decile_edges, "agg_d")
    out["aggregated_score_ventile"] = cut_feature(agg_raw, ventile_edges, "agg_v")
    out["aggregated_score_abs"] = agg_clipped.abs()

    tail = np.select(
        [
            agg_raw <= tail_thresholds["p01"],
            agg_raw <= tail_thresholds["p05"],
            agg_raw >= tail_thresholds["p99"],
            agg_raw >= tail_thresholds["p95"],
            agg_raw >= tail_thresholds["p90"],
        ],
        ["low_01", "low_05", "high_99", "high_95", "high_90"],
        default="middle",
    )
    out["aggregated_score_tail_band"] = pd.Series(tail, index=out.index).astype(str)

    if "experience_years" in out.columns:
        exp = pd.to_numeric(out["experience_years"], errors="coerce").fillna(0.0)
        out["experience_band"] = experience_band(df["experience_years"])
        out["agg_x_experience"] = agg_clipped * exp
        out["agg_per_experience_plus1"] = agg_clipped / (exp + 1.0)
    if "skills_count" in out.columns:
        skills = pd.to_numeric(out["skills_count"], errors="coerce").fillna(0.0)
        skills_raw = pd.to_numeric(df["skills_count"], errors="coerce")
        out["skills_count_decile"] = cut_feature(skills_raw, skills_decile_edges, "skills_d")
        out["skills_is_outlier_gt25"] = (skills_raw > 25).fillna(False).astype("int8")
        out["skills_tail_band"] = np.select(
            [
                skills_raw >= skills_tail_thresholds["p99"],
                skills_raw >= skills_tail_thresholds["p95"],
                skills_raw >= skills_tail_thresholds["p90"],
                skills_raw > 25,
            ],
            ["skills_high_99", "skills_high_95", "skills_high_90", "skills_gt25"],
            default="skills_normal",
        )
        out["agg_x_skills"] = agg_clipped * skills
        out["agg_per_skill_plus1"] = agg_clipped / (skills + 1.0)
    if "certifications" in out.columns:
        cert_raw = pd.to_numeric(df["certifications"], errors="coerce")
        cert = pd.to_numeric(out["certifications"], errors="coerce").fillna(0.0)
        out["certification_bucket"] = certification_bucket(df["certifications"])
        out["certifications_negative"] = (cert_raw < 0).fillna(False).astype("int8")
        out["certifications_non_integer"] = (cert_raw.notna() & (np.abs(cert_raw - np.rint(cert_raw)) > 1e-6)).astype("int8")
        out["agg_x_certifications_plus1"] = agg_clipped * (cert + 1.0)

    if "skill_bracket" in out.columns:
        out["agg_decile_skill"] = out["aggregated_score_decile"] + "_" + out["skill_bracket"].astype(str)
    if "education_level" in out.columns:
        out["agg_decile_education"] = out["aggregated_score_decile"] + "_" + out["education_level"].astype(str)
    if "job_title" in out.columns:
        out["agg_decile_job"] = out["aggregated_score_decile"] + "_" + out["job_title"].astype(str)
    if "company_size" in out.columns:
        out["agg_decile_company"] = out["aggregated_score_decile"] + "_" + out["company_size"].astype(str)
    if {"location", "job_title"}.issubset(out.columns):
        out["location_job"] = out["location"].astype(str) + "_" + out["job_title"].astype(str)
    if {"job_title", "remote_work"}.issubset(out.columns):
        remote = out["remote_work"].astype("object").where(out["remote_work"].notna(), "Missing").astype(str)
        out["job_remote"] = out["job_title"].astype(str) + "_" + remote
    if {"industry", "location"}.issubset(out.columns):
        out["industry_location"] = out["industry"].astype(str) + "_" + out["location"].astype(str)
    if {"experience_band", "job_title"}.issubset(out.columns):
        out["experience_band_job"] = out["experience_band"].astype(str) + "_" + out["job_title"].astype(str)
    if {"skills_count_decile", "job_title"}.issubset(out.columns):
        out["skills_decile_job"] = out["skills_count_decile"].astype(str) + "_" + out["job_title"].astype(str)
    if {"certification_bucket", "job_title"}.issubset(out.columns):
        out["certification_job"] = out["certification_bucket"].astype(str) + "_" + out["job_title"].astype(str)

    out = gm.add_interactions(out)
    if drop_numeric_agg and "aggregated_score" in out.columns:
        out = out.drop(columns=["aggregated_score"])
    for col in drop_raw_feature_cols:
        if col in out.columns:
            out = out.drop(columns=[col])
    return out


def prepare_frames(drop_numeric_agg: bool, drop_raw_feature_cols: tuple[str, ...] = ()):
    train_raw = pd.read_csv(gm.TRAIN_PATH)
    test_raw = pd.read_csv(gm.TEST_PATH)
    private_raw = pd.read_csv(gm.PRIVATE_PATH)
    pseudo = pd.read_csv(gm.PSEUDO_SOURCE_PATH)
    if not pseudo["id"].reset_index(drop=True).equals(private_raw["id"].reset_index(drop=True)):
        raise ValueError("Pseudo-label ids are not aligned with private test ids.")

    base_features = [col for col in train_raw.columns if col not in gm.EXCLUDED]
    local_raw = pd.concat([train_raw, test_raw], ignore_index=True)
    agg_lower, agg_upper = gm.iqr_bounds(train_raw["aggregated_score"])
    local_agg = pd.to_numeric(local_raw["aggregated_score"], errors="coerce")
    local_skills = pd.to_numeric(local_raw["skills_count"], errors="coerce")
    decile_edges = unique_edges(local_agg, np.linspace(0, 1, 11))
    ventile_edges = unique_edges(local_agg, np.linspace(0, 1, 21))
    skills_decile_edges = unique_edges(local_skills, np.linspace(0, 1, 11))
    tail_thresholds = {
        "p01": float(local_agg.quantile(0.01)),
        "p05": float(local_agg.quantile(0.05)),
        "p90": float(local_agg.quantile(0.90)),
        "p95": float(local_agg.quantile(0.95)),
        "p99": float(local_agg.quantile(0.99)),
    }
    skills_tail_thresholds = {
        "p90": float(local_skills.quantile(0.90)),
        "p95": float(local_skills.quantile(0.95)),
        "p99": float(local_skills.quantile(0.99)),
    }

    X_local = add_data_centric_features(
        local_raw[base_features],
        agg_lower,
        agg_upper,
        decile_edges,
        ventile_edges,
        skills_decile_edges,
        skills_tail_thresholds,
        tail_thresholds,
        drop_numeric_agg,
        drop_raw_feature_cols,
    ).reset_index(drop=True)
    y_local = local_raw[gm.REGRESSION_TARGET].reset_index(drop=True).astype(float)

    private_features = [col for col in base_features if col in private_raw.columns]
    X_private = add_data_centric_features(
        private_raw[private_features],
        agg_lower,
        agg_upper,
        decile_edges,
        ventile_edges,
        skills_decile_edges,
        skills_tail_thresholds,
        tail_thresholds,
        drop_numeric_agg,
        drop_raw_feature_cols,
    ).reset_index(drop=True)

    cat_cols = X_local.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    for frame in [X_local, X_private]:
        for col in cat_cols:
            if col in frame.columns:
                frame[col] = frame[col].astype("object").where(frame[col].notna(), "Missing").astype(str)
    num_cols = [col for col in X_local.columns if col not in cat_cols]

    local_adv_weights, adv_summary = gm.compute_adversarial_weights(X_local, X_private, num_cols, cat_cols)

    X_master = pd.concat([X_local, X_private], ignore_index=True)
    y_master = pd.Series(
        np.r_[y_local.to_numpy(dtype=float), pseudo["prediction"].to_numpy(dtype=float)],
        name="salary_or_pseudo",
    )
    master_weights = np.r_[local_adv_weights, np.full(len(X_private), gm.PSEUDO_WEIGHT, dtype=float)]

    preprocessor = gm.KFoldTargetEncodingPreprocessor(
        numeric_cols=num_cols,
        categorical_cols=cat_cols,
        n_splits=5,
        smoothing=gm.TE_SMOOTHING,
        random_state=gm.RANDOM_STATE,
    )
    X_train = preprocessor.fit_transform(X_master, y_master)
    X_test = preprocessor.transform(X_private)

    return {
        "private_ids": private_raw["id"],
        "X_train": X_train,
        "X_test": X_test,
        "y_train_log": np.log1p(y_master.to_numpy(dtype=float)),
        "weights": master_weights,
        "min_salary": float(y_local.min()),
        "max_salary": float(y_local.max()),
        "reference": pseudo["prediction"].to_numpy(dtype=float),
        "feature_names": num_cols + [f"{col}_target_mean" for col in cat_cols],
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "adv_summary": adv_summary,
        "tail_thresholds": tail_thresholds,
        "skills_tail_thresholds": skills_tail_thresholds,
    }


def train_lgbm_variant(name: str, prepared: dict) -> dict:
    preds = []
    importances = []
    for seed in SEEDS:
        print(f"Training {name} LGBM seed {seed}...")
        model = LGBMRegressor(**{**gm.LGBM_PARAMS, "random_state": seed})
        model.fit(prepared["X_train"], prepared["y_train_log"], sample_weight=prepared["weights"])
        preds.append(np.expm1(model.predict(prepared["X_test"])))
        importances.append(model.feature_importances_)

    pred = np.clip(np.mean(preds, axis=0), prepared["min_salary"], prepared["max_salary"])
    out_name = f"submission_gm_lgbm_{name}.csv"
    gm.save_submission(prepared["private_ids"], pred, out_name, prepared["reference"])

    sample = pd.read_csv("submission_gm_lgbm_seed_ensemble.csv")
    best = sample[sample.columns[1]].to_numpy(dtype=float)
    diff = pred - best

    importance = pd.DataFrame({
        "feature": prepared["feature_names"],
        "importance": np.mean(importances, axis=0),
    }).sort_values("importance", ascending=False)
    importance_path = f"feature_importance_lgbm_{name}.csv"
    importance.to_csv(importance_path, index=False)

    return {
        "file": out_name,
        "importance_file": importance_path,
        "diff_vs_seed5_mean": float(diff.mean()),
        "diff_vs_seed5_abs_mean": float(np.abs(diff).mean()),
        "diff_vs_seed5_std": float(diff.std()),
        "corr_vs_seed5": float(np.corrcoef(pred, best)[0, 1]),
        "mean": float(pred.mean()),
        "std": float(pred.std()),
        "top_features": importance.head(15).to_dict(orient="records"),
    }


def main() -> None:
    start = time.time()
    summary = {
        "idea": "Data-centric LGBM v2: bucket aggregated_score, skills_count, experience_years and certifications; add interaction target encoding.",
        "seeds": SEEDS,
        "outputs": [],
    }
    for name, drop_numeric_agg in [
        ("aggbins_plus_keepraw_seed5", False),
        ("aggbins_plus_dropraw_seed5", True),
    ]:
        prepared = prepare_frames(drop_numeric_agg=drop_numeric_agg)
        row = train_lgbm_variant(name, prepared)
        row["drop_numeric_aggregated_score"] = drop_numeric_agg
        row["tail_thresholds"] = prepared["tail_thresholds"]
        row["skills_tail_thresholds"] = prepared["skills_tail_thresholds"]
        row["adversarial_summary"] = prepared["adv_summary"]
        row["n_numeric_features"] = len(prepared["num_cols"])
        row["n_categorical_features"] = len(prepared["cat_cols"])
        summary["outputs"].append(row)

    summary["seconds"] = round(time.time() - start, 2)
    OUT_SUMMARY.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
