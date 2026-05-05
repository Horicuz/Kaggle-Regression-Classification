# Cod recuperat pentru perfectionarea regresiei

Acest folder pastreaza artefactele importante care au dus la modelul final de regresie. Am scos experimentele moarte din radacina proiectului, dar am pastrat aici codul/logurile care explica drumul.

## Artefacte principale

- `main_recovered_full_2026-05-02.ipynb` si `main_recovered_regression_2026-05-02.ipynb`: notebook-uri vechi recuperate din zip-urile de predare intermediare. De aici provin celulele initiale cu adversarial validation, K-Fold Target Encoding, CatBoost si K-Fold regression CV.
- `make_gm_seed_ensemble.py`: scriptul mare folosit pentru pseudo-labeling, K-Fold Target Encoding, ponderi adversariale si seed ensemble LightGBM/CatBoost/XGBoost.
- `gm_seed_ensemble_summary.json`: logul rularii seed ensemble. Include adversarial validation RandomForest cu AUC `0.7744343062500001`, parametrii LGBM/XGB/CatBoost, seed-urile si fisierele generate.
- `testlike_holdout_optuna_summary.json`: logul rularii test-like holdout cu detector CatBoost local-vs-private. Include AUC OOF `0.911454216796875`, top features de drift, splitul test-like si parametrii Optuna pentru CatBoost log-target. Codul `.py` separat pentru aceasta rulare nu mai exista in proiect, dar logul pastreaza configuratia si rezultatele.
- `make_gm_data_centric_lgbm.py`: scriptul pentru etapele data-centric: bucket-uri pentru `aggregated_score`, `skills_count`, `experience_years`, `certifications`, interaction target encoding si ablation keep/drop raw.
- `gm_data_centric_lgbm_summary.json`, `gm_data_centric_lgbm_v2_summary.json`, `gm_data_centric_v2_raw_ablation_summary.json`, `gm_drop75_v2drop_followup_summary.json`: logurile deciziilor finale data-centric.
- `feature_importance_*.csv`: importanțe pentru variantele LGBM data-centric relevante.

## Scripturi recuperate din VSCode Local History

- `vscode_feature_engineering_sqrt_optuna.py`: scriptul vechi cu cele 4 interactiuni (`edu_company`, `job_company`, `loc_company`, `edu_job`), CatBoostClassifier adversarial, weights `sqrt(p/(1-p))`, Optuna si post-procesari alpha.
- `vscode_super_adversarial_optuna.py`: varianta anterioara "super adversarial" cu CatBoostClassifier si raw odds weights.
- `vscode_feature_engineering_sqrt_seedavg.py`: seed averaging peste modelul CatBoost/Optuna.
- `vscode_gridsearch_fintuning.py`: grid search local in jurul parametrilor gasiti anterior.
- `vscode_make_postprocess_submissions.py`: post-procesari si probabilitati adversariale pentru submission-uri.

Acestea sunt pastrate ca dovada a incercarilor care au dus la forma finala. Nu toate sunt rulate in `main.ipynb`, pentru ca unele sunt experimente intermediare sau costisitoare, dar ele arata codul folosit inainte sa fie curatat proiectul.

## Cum se citeste firul

1. Adversarial validation a aratat ca testul privat nu are exact aceeasi distributie ca train/test local.
2. Target Encoding cu K-Fold si smoothing a fost folosit pentru categorice si interactiuni, ca sa evitam leakage-ul si mediile instabile.
3. Pseudo-labeling-ul a permis modelului sa vada distributia testului privat.
4. LightGBM seed ensemble a devenit ancora cea mai buna.
5. Feedback-ul Kaggle a aratat ca blendurile pozitive Cat/XGB nu mai ajutau mult dupa seed ensemble; de aceea au aparut sweep-urile mici si directia `away`.
6. Ultimul castig a venit din reprezentarea datelor: bucket-uri pe coloanele cu drift/cozi lungi si interaction target encoding pe combinatii cu sens salarial.
