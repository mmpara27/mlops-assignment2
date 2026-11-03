import argparse
import hashlib
import json
import os
from pathlib import Path
from datetime import timedelta
import time
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from feast import Entity, FeatureStore, FeatureView, Field, FileSource
from feast.types import Float32
from feast import ValueType
from clean_data import clean_data
from codecarbon import OfflineEmissionsTracker
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
BASE_FEATURES = ["age", "height", "weight", "is_male"]
EXTENDED_FEATURES = BASE_FEATURES + ["bmi", "weight_height_ratio", "age_squared"]

FEATURE_VERSIONS = {"v1": BASE_FEATURES, "v2": EXTENDED_FEATURES}
MODEL_SETTINGS = {
    "h1": {"n_estimators": 50, "max_depth": 6},
    "h2": {"n_estimators": 100, "max_depth": 12},
}

def load_and_prepare_data(path, out):
    raw = pd.read_csv(path)
    # 1. Use the assignment's cleaning code.
    data = clean_data(raw).reset_index(drop=True)

    # Feast needs a unique key. This is a new row ID, not the dropped athlete ID.
    data["row_id"] = np.arange(len(data), dtype="int64")

    # 2. Create the two feature versions.
    data["is_male"] = (data.gender == "Male").astype("float32")
    data["bmi"] = 703 * data.weight / data.height**2
    data["weight_height_ratio"] = data.weight / data.height
    data["age_squared"] = data.age**2
    data[EXTENDED_FEATURES] = data[EXTENDED_FEATURES].astype("float32")
    # Dataset has no observation dates: synthetic snapshot dates only, not temporal validation.
    data["event_timestamp"] = pd.Timestamp("2020-01-01", tz="UTC")
    report = {
        "raw_rows": len(raw),
        "clean_rows": len(data),
        "removed_rows": len(raw) - len(data),
        "input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    (out / "cleaning.json").write_text(json.dumps(report, indent=2))
    return data, report


def create_feature_store(data, repo, out):
    """Save both feature versions and register them in Feast."""
    (repo / "data").mkdir(parents=True, exist_ok=True)
    (repo / "feature_store.yaml").write_text("""project: athletes
provider: local
registry: registry.db
offline_store:
  type: file
online_store:
  type: sqlite
  path: online.db
entity_key_serialization_version: 3
""")
    store = FeatureStore(repo_path=str(repo))
    entity = Entity(name="athlete", join_keys=["row_id"], value_type=ValueType.INT64)
    views = []
    for version, names in FEATURE_VERSIONS.items():
        file = repo / "data" / f"{version}.parquet"
        data[["row_id", "event_timestamp"] + names].to_parquet(file, index=False)
        source = FileSource(path=str(file), timestamp_field="event_timestamp")
        view = FeatureView(
            name=f"athlete_{version}",
            entities=[entity],
            ttl=timedelta(days=365),
            schema=[Field(name=name, dtype=Float32) for name in names],
            source=source,
            online=True,
            tags={"version": version},
        )
        views.append(view)
    store.apply([entity] + views)
    (out / "feature_versions.json").write_text(json.dumps(FEATURE_VERSIONS, indent=2))
    return store


def retrieve_training_data(store, data, version, names):
    """Read features through Feast, then attach the target by row ID."""
    entities = data[["row_id"]].copy()
    entities["event_timestamp"] = pd.Timestamp("2020-01-02", tz="UTC")
    features = store.get_historical_features(
        entity_df=entities,
        features=[f"athlete_{version}:{name}" for name in names],
    ).to_df()
    training_data = features.merge(
        data[["row_id", "deadlift"]], on="row_id", validate="one_to_one"
    ).set_index("row_id")
    if training_data[names].isna().any().any():
        raise ValueError("Feast returned missing feature values.")
    return training_data


def train_and_measure(
    Xtrain, Xtest, ytrain, ytest, version, names, config, label, country, out
):
    """Train one forest and log its accuracy and carbon estimate."""
    # Start a fresh emissions file for this run.
    (out / f"{label}_emissions.csv").unlink(missing_ok=True)
    tracker = OfflineEmissionsTracker(
        country_iso_code=country,
        project_name=label,
        output_dir=str(out),
        output_file=f"{label}_emissions.csv",
        measure_power_secs=1,
        tracking_mode="process",
        log_level="error",
    )
    tracker.start()
    start = time.perf_counter()
    try:
        model = RandomForestRegressor(**config, random_state=42, n_jobs=1)
        model.fit(Xtrain, ytrain)
        pred = model.predict(Xtest)
    finally:
        elapsed = time.perf_counter() - start
        emissions = tracker.stop()
    if emissions is None:
        raise RuntimeError("CodeCarbon returned no estimate; do not substitute zero.")
    metrics = {
        "mae": mean_absolute_error(ytest, pred),
        "rmse": np.sqrt(mean_squared_error(ytest, pred)),
        "r2": r2_score(ytest, pred),
        "carbon_kg": emissions,
        "fit_predict_seconds": elapsed,
    }
    baseline = np.repeat(ytrain.median(), len(ytest))
    metrics["baseline_mae"] = mean_absolute_error(ytest, baseline)
    mlflow.log_params(
        {
            "feature_version": version,
            "features": ",".join(names),
            **config,
            "random_state": 42,
            "n_jobs": 1,
            "country": country,
        }
    )
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, artifact_path="model", input_example=Xtrain.head(3))
    mlflow.log_artifact(str(out / f"{label}_emissions.csv"))
    return metrics, pred


def plot_predictions(axes, col, label, ytest, pred):
    """Show predictions and errors for one run."""
    axes[0, col].scatter(ytest, pred, s=3, alpha=0.1)
    axes[0, col].plot([20, 1200], [20, 1200], "r--")
    axes[0, col].set(
        xlabel="Actual deadlift (lb)", ylabel="Predicted (lb)", title=label
    )
    axes[1, col].scatter(pred, ytest - pred, s=3, alpha=0.1)
    axes[1, col].axhline(0, color="red", linestyle="--")
    axes[1, col].set(xlabel="Predicted (lb)", ylabel="Residual (lb)")


def save_comparison(rows, fig, out):
    """Save the four-run table and comparison plots."""
    # 6. Save the metrics and plots.
    fig.tight_layout()
    fig.savefig(out / "predictions_residuals.png", dpi=140)
    plt.close(fig)
    results = pd.DataFrame(rows)
    results.to_csv(out / "comparison.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, metric in zip(axes, ["mae", "rmse", "carbon_kg"]):
        ax.bar(results.run, results[metric])
        ax.set_title(metric)
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "comparison.png", dpi=140)
    plt.close(fig)
    best = results.loc[results.mae.idxmin()]
    text = f"Lowest held-out MAE: {best['run']} ({best.mae:.2f} lb).\n\n"
    text += results.to_string(index=False) + "\n\n"
    text += (
        "Compare scatter points against the diagonal and residuals against zero. "
        "Systematic residual slopes indicate bias at the extremes. All runs use the same test rows. "
        "Use a new holdout to check the final model after choosing a run. "
        "Carbon estimates cover fit and predict only, not shared cleaning, Feast setup, plotting, or dependency installation. "
        "They are hardware and country dependent; short single runs are noisy and do not establish a reliable carbon ranking. "
        "Snapshot dates are synthetic. Self-reported data and plausibility filters limit generalization."
    )
    (out / "REPORT.txt").write_text(text)
    return results


def main(cli_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=ROOT / "data/athletes.csv")
    parser.add_argument(
        "--country", default="USA", help="Country where this computer runs"
    )
    args = parser.parse_args(cli_args)
    data_path = args.data.resolve()
    os.chdir(ROOT)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("athletes-feature-store")

    with mlflow.start_run(run_name="full_pipeline"):
        # 1. Load and clean athletes.csv.
        data, cleaning = load_and_prepare_data(data_path, out)
        mlflow.log_params(cleaning)

        # 2. Register both feature versions in Feast.
        store = create_feature_store(data, ROOT / "feature_repo", out)

        # 3. Keep the same 80/20 split for every run.
        train_ids, test_ids = train_test_split(
            data["row_id"].to_numpy(), test_size=0.2, random_state=42
        )
        split = pd.DataFrame(
            {
                "row_id": data["row_id"],
                "split": np.where(data["row_id"].isin(test_ids), "test", "train"),
            }
        )
        split.to_csv(out / "split.csv", index=False)

        # 4. Run both settings for each feature version.
        rows = []
        fig, axes = plt.subplots(2, 4, figsize=(17, 8))
        column = 0
        for version, names in FEATURE_VERSIONS.items():
            training_data = retrieve_training_data(store, data, version, names)
            Xtrain = training_data.loc[train_ids, names]
            Xtest = training_data.loc[test_ids, names]
            ytrain = training_data.loc[train_ids, "deadlift"]
            ytest = training_data.loc[test_ids, "deadlift"]

            for setting, config in MODEL_SETTINGS.items():
                label = f"{version}_{setting}"
                with mlflow.start_run(run_name=label, nested=True):
                    metrics, predictions = train_and_measure(
                        Xtrain,
                        Xtest,
                        ytrain,
                        ytest,
                        version,
                        names,
                        config,
                        label,
                        args.country,
                        out,
                    )
                    rows.append({"run": label, **config, **metrics})
                    plot_predictions(axes, column, label, ytest, predictions)
                    column += 1

        # 5. Save metrics, model plots, and carbon comparisons.
        results = save_comparison(rows, fig, out)
        print(results.to_string(index=False))
        print(f"Saved results to {out}")
        mlflow.log_artifacts(str(out), artifact_path="comparison")


if __name__ == "__main__":
    main()
