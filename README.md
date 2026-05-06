
# Kaggle Project — Regression & Classification (Complete)

⭐ If you find this project useful, please give it a star on GitHub — it helps a lot!

## Theory & Context

This repository contains a complete solution for two tasks on educational data: a regression task (continuous prediction) and a classification task (categorical prediction). The main goals are:

- Understand and preprocess tabular data (feature engineering, handling missing values, encoding categoricals).
- Train robust models using ensemble algorithms (LightGBM, XGBoost, etc.).
- Evaluate performance with cross-validation (K-Fold) and produce final submission files.

Methodological principles used:

- Clear separation between raw and aggregated features, and ablation studies to compare strategies.
- Multi-seed training and blending/stacking to reduce variance and improve generalization.
- Metric choices: regression metrics (RMSE, MAE); classification metrics (AUC-ROC, accuracy, F1 where appropriate).

## Results Summary

Final output files are included in the repository:

- [submission_regression_final.csv](submission_regression_final.csv) — final regression predictions.
- [submission_classification_final.csv](submission_classification_final.csv) — final classification predictions.

Experiment reports and summaries are stored in `_archive_unused_experiments_2026-05-03/`.

## Repository Structure (selected)

- [main.ipynb](main.ipynb) — main notebook for exploration, training and submission generation.
- [main copy.ipynb](main%20copy.ipynb) — auxiliary notebook copy.
- [CC_education_economy_train.csv](CC_education_economy_train.csv) — training dataset.
- [CC_education_economy_test.csv](CC_education_economy_test.csv) — test dataset.
- [CC_private_test.csv](CC_private_test.csv) — private test dataset (if applicable).
- `_archive_unused_experiments_2026-05-03/` — experiment history, results and helper scripts.

## Installation

Prerequisites: Python 3.8+ recommended, `pip`, and a virtual environment tool (`venv`, `virtualenv`, or `conda`).

Example (venv):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

If `requirements.txt` is not present, install minimal packages manually:

```bash
pip install jupyterlab numpy pandas scikit-learn lightgbm xgboost matplotlib seaborn
```

## Usage — reproduce experiments

1. Activate the virtual environment (see Installation).
2. Start JupyterLab or Jupyter Notebook:

```bash
jupyter lab
# or
jupyter notebook
```

3. Open [main.ipynb](main.ipynb) and run the cells in order to:
   - Load the datasets.
   - Execute preprocessing and feature engineering.
   - Run training loops (K-Fold, multi-seed).
   - Produce submission files `submission_*.csv`.

Practical notes:
- To speed up experiments, follow a stepwise approach: feature engineering → local validation → final training.
- For batch runs, extract relevant notebook cells into a Python script.

## Best Practices & Recommendations

- Fix random seeds to ensure reproducibility.
- Keep logs (metrics per fold/seed) and save trained models (pickle / joblib / model.save).
- Use nested cross-validation for hyperparameter tuning if the dataset is small.

## Contributing

Contributions are welcome:

- Open an issue to discuss proposed changes or new features.
- Submit a pull request for fixes, new features, or model optimizations.

Please include clear descriptions and minimal reproducible examples in PRs.

## License

This repository is provided as-is. If you want an explicit license, `MIT` is recommended for wide permissiveness.

## Contact

For questions or suggestions, open an issue on GitHub.

---

Thanks for checking out this project — if you like it, please give it a star ⭐!
