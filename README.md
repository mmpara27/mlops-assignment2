# Athletes Feature Store + ML Pipeline

## 1. ML pipeline
MLflow Projects runs cleaning, feature creation, retrieval, training, and comparisons. MLflow logs a parent run and four child experiments.

## 2. Feature store
Feast registers athlete_v1 and athlete_v2 with local Parquet snapshots. Training features are retrieved through get_historical_features().

## 3. Data and feature versions
Source: athletes.csv. The supplied cleaning code is in clean_data.py. The undefined v2 reference was corrected to data; the original deadlift OR logic is preserved. Cleaning counts are in results/cleaning.json.
V1: age, height, weight, male indicator. V2: V1 + BMI, weight/height ratio, age squared. BMI assumes pounds/inches. Deadlift is the target. Feast row IDs are generated after cleaning; dates are synthetic snapshots.

## 4. Four experiments
Same algorithm: RandomForestRegressor. H1: 50 trees, depth 6. H2: 100 trees, depth 12. Combinations: V1/H1, V1/H2, V2/H1, V2/H2. No AutoML. Same 80/20 row split, seed 42, one worker.

## 5. Model comparison
| run   |     mae |    rmse |       r2 |   baseline_mae |
|:------|--------:|--------:|---------:|---------------:|
| v1_h1 | 46.193  | 60.9273 | 0.606765 |        80.0428 |
| v1_h2 | 46.6137 | 61.3423 | 0.601389 |        80.0428 |
| v2_h1 | 45.918  | 60.6036 | 0.610931 |        80.0428 |
| v2_h2 | 46.3916 | 61.1586 | 0.603773 |        80.0428 |

Lowest MAE: v2_h1 (45.92 lb).

1. Diagonal Alignment:
All four models show a cluster following the red diagonal line well for average-range deadlifts (roughly 150 lb to 450 lb).
However, as deadlift weights exceed 500 lb, all models significantly underpredict performance, with predicted values flattening out and capping around 550 lb.

2. High Deadlifts:
High deadlift values (e.g., above 600 lb) are severely underpredicted across all configurations. This behavior is expected from Random Forest models, which cannot predict outside the range of the training target values and tend to predict towards the mean near the boundaries.

3. Residual Spread and Bias:
The residual plots demonstrate a distinct negative trend for higher actual values. This downward slope indicates systematic bias at the extremes: overprediction of very low deadlifts and severe underprediction of very high deadlifts.

4. Configuration Differences:
The v2 configurations (incorporating BMI, weight/height ratio, and age squared) show slightly lower Mean Absolute Error (MAE) compared to their v1 counterparts, indicating that these engineered features provide helpful non-linear signals. The shallower H1 models (max_depth=6) performed slightly better on the test set than the deeper H2 models, suggesting that the deeper trees may have slightly overfit the training data.

![Predictions and residuals](results/predictions_residuals.png)

This is an exploratory comparison on one shared test split. A new holdout is needed after selecting a model. Self-reported values and filters limit generalization.

## 6. Carbon comparison
| run   |   estimated_co2_g |   fit_predict_seconds |
|:------|------------------:|----------------------:|
| v1_h1 |        0.00403128 |              0.738541 |
| v1_h2 |        0.0185503  |              3.44542  |
| v2_h1 |        0.00814547 |              1.51048  |
| v2_h2 |        0.031386   |              5.82664  |

Lowest estimated carbon: v1_h1. Estimates cover fitting and prediction only, excluding cleaning, store setup, plotting, and installation. Host country assumed: USA; location unverified. Shared Colab hardware and short measurements make these estimates noisy.

![Comparison](results/comparison.png)

## Run locally
Use a Python 3.11 or 3.12 virtual environment. Place athletes.csv in data/.

```bash
python -m pip install -r requirements.txt
python -m mlflow run . --env-manager=local -P data=data/athletes.csv
```

