# Tema 1 CC - Machine Learning

Acest folder contine varianta curata pentru predare. Notebook-ul principal este
`main.ipynb` si contine:

- partea A: EDA general pentru setul de date;
- partea B: regresie pentru `salary`;
- partea C: clasificare multiclasă pentru `vacation`.

Raportul compact pentru citire/predare este `Raport_Tema1_CC.pdf`.

## Date

Fisierele folosite:

- `CC_education_economy_train.csv`
- `CC_education_economy_test.csv`
- `CC_private_test.csv`

Train si test local contin ambele tinte (`salary`, `vacation`). Private test contine
doar features si `id`, deci submission-urile finale au formatul `id,prediction`.

## Regresie

Repere acoperite:

- baseline `LinearRegression`;
- regularizare `Ridge` si `Lasso`;
- metrici `MAE`, `MSE`, `RMSE`, `R2`;
- grafice/analiza train vs test;
- analiza erorilor pe test local;
- submission final Kaggle.

Firul de perfectionare a fost comprimat in notebook in cateva etape clare:

1. modele locale mai puternice si Target Encoding;
2. pseudo-labeling pe private test;
3. LightGBM seed ensemble;
4. directie `away` fata de blenduri CatBoost/XGBoost care stricau scorul;
5. data-centric V1 pe `aggregated_score`;
6. data-centric V2 cu bins pentru `skills_count`, `experience_years`,
   `certifications` si interaction target encoding.

Pentru laborator, codul si logurile care dovedesc aceste etape sunt pastrate in:

- `regression_recovered_code`

Folderul include notebook-uri vechi recuperate, scripturile mari de seed ensemble si
data-centric LGBM, logul adversarial RandomForest cu AUC aproximativ 0.77 si logul
detectorului CatBoost test-like cu AUC aproximativ 0.91.

Submission-ul final de regresie este:

- `submission_regression_final.csv`

CSV-urile reprezentative pentru regresie sunt in:

- `regression_csv_selected/01_baseline_minim`
- `regression_csv_selected/02_perfectionare_kaggle`

## Clasificare

Repere acoperite:

- analiza echilibrului de clase pentru `vacation`;
- baseline `DecisionTreeClassifier`;
- variatie de hiperparametri pentru arbore (`max_depth`, `min_samples_leaf`);
- metrici `accuracy`, `precision`, `recall`, `F1 macro`, `F1 weighted`;
- confusion matrix;
- perfectionare cu modele de tip ensemble/boosting;
- versiune V1 data-centric cu bins si interactiuni categoriale;
- Stratified K-Fold pentru modelul final.

Rezultatul local cel mai bun obtinut:

| Model | Accuracy | F1 macro | F1 weighted |
|---|---:|---:|---:|
| `HistGradientBoosting` | 0.7611 | 0.6777 | 0.7568 |
| `CatBoost_native` | 0.7606 | 0.6801 | 0.7581 |
| `LightGBM_multiclass` | 0.7603 | 0.6759 | 0.7560 |
| `DecisionTree_depthNone_leaf100` | 0.7256 | 0.6686 | 0.7374 |
| `Dummy_most_frequent` | 0.5420 | 0.1757 | 0.3810 |

Submission-ul final de clasificare este:

- `submission_classification_final.csv`

CSV-urile reprezentative pentru clasificare sunt in:

- `classification_csv_selected`

## Structura finala

Fisiere importante in radacina:

- `main.ipynb`
- `README.md`
- `Raport_Tema1_CC.pdf`
- `submission_regression_final.csv`
- `submission_classification_final.csv`
- cele trei fisiere de date CSV
- PDF-ul cerintei
- `regression_recovered_code`, cu artefactele importante ale perfectionarii regresiei

Experimentele brute care nu au dus la decizii utile au ramas in
`_archive_unused_experiments_2026-05-03` si nu sunt incluse in pachetul final de predare.
