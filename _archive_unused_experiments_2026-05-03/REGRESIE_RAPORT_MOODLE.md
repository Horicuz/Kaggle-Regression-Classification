# Raport partial Moodle - Regresie salary

Acest raport acopera doar partea de regresie a temei: predictia coloanei `salary`.
Clasificarea pentru `vacation` va fi adaugata separat, dupa finalizarea acestei parti.

## Cerinte PDF acoperite

Am verificat cerintele din PDF si partea de regresie acopera:

- EDA general: tipuri de atribute, valori lipsa, plaje de valori, outlieri, distributii;
- analiza corelatiilor numerice si relatia feature-urilor cu `salary`;
- preprocessing justificat: imputare, tratare outlieri, encoding, standardizare unde are sens;
- baseline cu `LinearRegression`;
- experimente de regularizare cu `Ridge` si `Lasso`;
- modele neliniare si boosting: arbori, RandomForest, ExtraTrees, XGBoost, LightGBM, CatBoost;
- K-Fold Cross-Validation pe `train + test local`;
- tabel comparativ cu metrici MAE/MSE/RMSE/R2;
- grafice de eroare train vs test;
- jurnalizarea pasilor spre cea mai buna solutie.

## Date si target

Seturile folosite:

- `CC_education_economy_train.csv`: 64.000 randuri, cu `salary` si `vacation`;
- `CC_education_economy_test.csv`: 16.000 randuri locale, cu `salary` si `vacation`;
- `CC_private_test.csv`: 16.000 randuri private Kaggle, fara `salary`.

Pentru regresie, targetul este `salary`. Coloanele excluse din regresie:

- `salary`: targetul;
- `vacation`: targetul taskului de clasificare, risc de leakage;
- `total_days_worked`: aproape redundant cu `experience_years`.

## Preprocessing

Pentru modelele liniare:

- numerice: imputare cu mediana + `StandardScaler`;
- categorice: imputare cu `Missing` + `OneHotEncoder`;
- motiv: modelele liniare sunt sensibile la scara variabilelor.

Pentru arbori si boosting sklearn/XGBoost/LightGBM:

- numerice: imputare cu mediana;
- categorice: imputare cu `Missing` + `OneHotEncoder` sau Target Encoding;
- fara standardizare, deoarece arborii nu au nevoie de scale comparabile.

Pentru CatBoost:

- am testat si varianta nativa cu categorice text;
- varianta cea mai buna local a folosit K-Fold Target Encoding + `log1p(salary)`.

Feature engineering important:

- `remote_work_missing`;
- mapari ordinale pentru `education_level`, `company_size`, `skill_bracket`;
- `skills_per_year`, `experience_x_skills`, `certifications_per_year`;
- interactiuni de drift: `edu_company`, `job_company`, `loc_company`, `edu_job`.

## Fir experimental

| Etapa | Motiv | Rezultat / decizie |
|---|---|---|
| `LinearRegression` | baseline minim cerut de PDF | stabileste reperul initial |
| `Ridge` si `Lasso` | cerinta de regularizare | arata efectul penalizarii coeficientilor |
| Arbori si ensemble-uri | captam neliniaritati | boosting-ul si CatBoost devin mai bune |
| CatBoost + Target Encoding | categorice multe si importante | CV MSE aproximativ `27.56M` |
| CatBoost + log target | target salary asimetric | CV MSE aproximativ `27.52M` |
| Optuna CatBoost | tuning controlat al hiperparametrilor | CV MSE aproximativ `27.36M` |
| Adversarial validation | testul privat difera de local | AUC adversarial mare, drift confirmat |
| Public-guided ensemble | combinam familii pe baza scorurilor publice | Kaggle `273.718M` |
| Pseudo-labeling | modelul vede distributia privata | Kaggle `271.904M` |
| LightGBM-heavy blend | feedback Kaggle arata ca LGBM corecteaza mai bine privatul | Kaggle `271.404M` |

## Model final de regresie

Varianta finala confirmata pentru regresie este:

- `submission_regression_final.csv`

Ea este obtinuta din:

- `25%` CatBoost pseudo-label model;
- `75%` LightGBM pseudo-label model.

Scor Kaggle cunoscut pentru aceasta varianta:

- MSE: `271,404,123.79`.

## Interpretare

Local, CatBoost parea mai bun decat LightGBM. Totusi, adversarial validation a aratat ca testul privat are distributie diferita de train/test local. Din acest motiv, validarea locala simpla nu a fost suficienta pentru alegerea finala.

Pseudo-labeling-ul a ajutat pentru ca modelul a fost expus la distributia privata prin etichete slabe, iar LightGBM a corectat o parte din bias-ul CatBoost pe acea distributie. Blend-ul final nu este doar o medie intamplatoare: a fost ales dupa o secventa de rezultate locale si feedback Kaggle.

## Fisiere relevante pentru predare

- `main.ipynb`: implementarea EDA + regresie + notebook de lucru;
- `README.md`: jurnalul complet al experimentelor;
- `REGRESIE_RAPORT_MOODLE.pdf`: acest raport exportat in PDF;
- `submission_regression_final.csv`: submission final pentru regresie;
- `pruned_submission_csv_manifest.json`: lista fisierelor de submission pastrate/sterse.
