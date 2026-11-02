# Climate-Aware Crop Yield Prediction & Model Benchmarking

## What this project does
A small, reproducible Python workflow that demonstrates data validation, exploratory machine learning, and regression-model comparison for a crop-yield prediction use case.

**Important:** The included dataset is synthetic and illustrative. It is only for checking that the workflow runs. Results from it are NOT evidence that the models predict real crop yields. Before presenting this as a finished portfolio project, replace the sample CSV with a credible public agricultural dataset, cite the source, confirm units and target definition, rerun the experiment, and report the real results.

## Features
- Validates required CSV columns and handles missing feature values using a median imputer.
- Splits data into train and test sets.
- Compares a mean DummyRegressor baseline, Linear Regression, and Random Forest.
- Reports MAE, RMSE, and R².
- Saves model comparison and predictions as CSV files.
- Saves an actual-vs-predicted plot.

## Requirements
Python 3.10+ recommended.

```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate
pip install -r requirements.txt
python crop_yield_analysis.py
```

The first run uses `sample_crop_yield_data.csv` and creates an `outputs/` folder.

## Use your own dataset
Prepare a CSV with these columns (rename them in the script if your dataset differs):
- `rainfall_mm`
- `mean_temp_c`
- `soil_moisture_pct`
- `fertilizer_kg_ha`
- `yield_tonnes_ha` (target)

Run:
```bash
python crop_yield_analysis.py --data path/to/your_dataset.csv
```

## How to interpret results
- MAE: average absolute prediction error, in the target's units.
- RMSE: penalises larger errors more strongly, in the target's units.
- R²: compares the model against a mean-target baseline; it can be negative on poor predictions.
- Always compare models on the same held-out test data. Do not select or tune models using the test set repeatedly.

## Before adding to your resume
1. Replace the illustrative data with a sourced real dataset.
2. Record dataset source, date accessed, units, and limitations.
3. Run the code and inspect outputs.
4. Add only the metrics you actually obtained on the real held-out test set.
5. Push the project to your GitHub and add the repository link.
6. Describe the work as “built” only after you have completed these steps.

## Suggested future improvements
- Add cross-validation and hyperparameter tuning.
- Compare feature importance carefully (not as causal proof).
- Evaluate by crop type/region if the data supports it.
- Add time-aware splitting if the dataset is chronological.
