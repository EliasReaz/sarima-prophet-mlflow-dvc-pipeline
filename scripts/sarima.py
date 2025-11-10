# sarima_pipeline.py

import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from urllib.parse import urlparse
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from requests.auth import HTTPBasicAuth
import mlflow
import joblib


from mlflow.models.signature import infer_signature

# === Environment Setup ===
from dotenv import load_dotenv
load_dotenv()


os.makedirs("output/plots", exist_ok=True)
    
os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")


mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))



BASE_DIR = Path(__file__).parent.parent


# === Config Loader ===
def load_config(config_path: str) -> dict:
    config_file = BASE_DIR / config_path
    with open(config_file, "r") as file:
        return yaml.safe_load(file)


# === SARIMA Evaluation with TimeSeriesSplit ===
def evaluate_sarima_with_tscv(y: pd.Series, order: tuple, seasonal_order: tuple, test_size: int, n_splits: int = 3):
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size)
    metrics = []
    mae_log, rmse_log, mae_orig, rmse_orig = [], [], [], []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y)):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        forecast_obj = model_fit.get_forecast(steps=len(y_test))
        preds_log = forecast_obj.predicted_mean
        conf_int_log = forecast_obj.conf_int()
        preds_orig = np.expm1(preds_log)

        # Metrics
        mae_log.append(mean_squared_error(y_test, preds_log))
        rmse_log.append(root_mean_squared_error(y_test, preds_log))
        mae_orig.append(mean_squared_error(np.expm1(y_test), preds_orig))
        rmse_orig.append(root_mean_squared_error(np.expm1(y_test), preds_orig))

        metrics.append({
            "fold": fold + 1,
            "mae (log scale)": mae_log[-1],
            "rmse (log scale)": rmse_log[-1],
            "mae (original scale)": mae_orig[-1],
            "rmse (original scale)": rmse_orig[-1]
        })

        print(f"Fold {fold+1} - MAE: {mae_log[-1]:.4f}, RMSE: {rmse_log[-1]:.4f}")
        # plot_forecast(y_train, y_test, preds_log, preds_orig, conf_int_log, fold)

    print(f"\nMean MAE (log): {np.mean(mae_log):.4f}, RMSE (log): {np.mean(rmse_log):.4f}")
    print(f"Mean MAE (orig): {np.mean(mae_orig):.4f}, RMSE (orig): {np.mean(rmse_orig):.4f}")

    return pd.DataFrame(metrics)


# === Forecast Plotting ===
def plot_forecast(y_train, y_test, preds_log, preds_orig, conf_int_log, fold):
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 6))

    # Log scale plot
    axes[0].plot(y_train.index[-50:], y_train[-50:], label='Train')
    axes[0].plot(y_test.index, y_test, label='Observed', color='green')
    axes[0].plot(preds_log.index, preds_log, label='Forecast', color='orange', linestyle='dashed')
    axes[0].fill_between(conf_int_log.index,
                         conf_int_log.iloc[:, 0],
                         conf_int_log.iloc[:, 1],
                         label="Confidence Interval (log)", color='pink', alpha=0.3)
    axes[0].set_title(f"SARIMA Forecast (log scale) - Fold {fold+1}")
    axes[0].legend()

    # Original scale plot
    axes[1].plot(y_train.index[-50:], np.expm1(y_train[-50:]), label='Train')
    axes[1].plot(y_test.index, np.expm1(y_test), label='Observed', color='green')
    axes[1].plot(preds_orig.index, preds_orig, label='Forecast', color='orange', linestyle='dashed')
    axes[1].set_title(f"SARIMA Forecast (original scale) - Fold {fold+1}")
    axes[1].legend()

    plt.tight_layout()
    # plt.show()
    plt.pause(2) # for 5 seconds
    plt.close(fig=fig)


# === MLflow Logging ===
def log_sarima_model(final_model, order, seasonal_order, results_df):

    # tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

    with mlflow.start_run(run_name="SARIMA_Modeling") as run:
        try:
            print(f"Started MLflow run: {run.info.run_id}")
            mlflow.log_param("order", order)
            mlflow.log_param("seasonal_order", seasonal_order)
            
            print("Order, seasonal order worked")

            # mlflow.log_metric("avg_MAE (log scale)", results_df["mae (log scale)"].mean())
            # mlflow.log_metric("avg_RMSE (log scale)", results_df["rmse (log scale)"].mean())
            
            # print("Mean worked")
            
            mlflow.log_metric("AIC", final_model.aic)
            mlflow.log_metric("BIC", final_model.bic)
            mlflow.log_metric("HQIC", final_model.hqic)
            
            print("AIC, BIC worked")

            mlflow.log_text(final_model.summary().as_text(), "model_summary.txt")
            
            print("Before Plot worked")

            mlflow.log_artifact(BASE_DIR / "output" / "plots" / "daily_redemption_count_2022_onwards.png", artifact_path="plots")
            mlflow.log_artifact(BASE_DIR / "output" / "plots" / "log_transformed_redemption_counts.png", artifact_path="plots")
            mlflow.log_artifact(BASE_DIR / "output" / "plots" / "acf_pacf_first_diff_of_logtransformed_data.png", artifact_path="plots")
            mlflow.log_artifact(BASE_DIR / "output" / "plots" / "residual_plot_diagnostic.png", artifact_path="plots")
            print("After Plot worked")
            
            mlflow.log_artifact(BASE_DIR / "output" / "plots" / "sarima_forecast.png", artifact_path="plots")

            # Ensure models directory exists
            # os.makedirs(BASE_DIR / "models", exist_ok=True)
            # joblib.dump(final_model, BASE_DIR / "models" / "sarima_model.pkl")
            # mlflow.log_artifact(BASE_DIR / "models" / "sarima_model.pkl", artifact_path="model")


            
            print(f"Logged SARIMA model with run_id: {run.info.run_id}")
            
        except Exception as e: 
            print(f"MLflow logging failed: {e}")

# === Main Execution ===
def main():
    config = load_config("config/config.yaml")
    sarima_cfg = config["sarima"]

    order = tuple(sarima_cfg["order"])
    seasonal_order = tuple(sarima_cfg["seasonal_order"])
    test_size = sarima_cfg.get("test_size", 7)

    data_path = BASE_DIR / "data" / "processed" / "daily_tickets_2022_25.parquet"
    df = pd.read_parquet(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    df.index.freq = 'D'

    y = df["Redemption Count"]
    y_log = np.log1p(y)

    results_df = evaluate_sarima_with_tscv(
        y_log,
        order=order,
        seasonal_order=seasonal_order,
        test_size=test_size,
        n_splits=5
    )

    print("\n=== SARIMA Evaluation Results ===\n", results_df)
    
    final_model = SARIMAX(y_log, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
    final_model = final_model.fit(disp=False)
   
    
    print("\n=== Calling log_sarima_model ===\n", results_df)
    log_sarima_model(final_model, order, seasonal_order, results_df)


if __name__ == "__main__":
    main()
