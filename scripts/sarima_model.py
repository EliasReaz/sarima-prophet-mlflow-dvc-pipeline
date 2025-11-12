import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import TimeSeriesSplit
import mlflow
import yaml
from pathlib import Path
from IPython.display import display
import os
from dotenv import load_dotenv

from utils.evaluate_forecast import evaluate_forecast
from utils.mlflow_logger import log_cv_results

load_dotenv()

os.makedirs("output/plots", exist_ok=True)
    
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_experiment("Forecasting_Toronto_Ferry_Tickets_Experiment")

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path().cwd().parent

# Plotting
def plot_forecast(y_train, y_test, preds_log, conf_int_log, fold):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Log-transformed scale
    ax[0].plot(y_train.index[-50:], y_train[-50:], label="Train", color="steelblue")
    ax[0].plot(y_test.index, y_test, label="Observed", color="steelblue")
    ax[0].plot(preds_log.index, preds_log, label="Predicted", color="orange")
    ax[0].fill_between(conf_int_log.index, conf_int_log.iloc[:, 0], conf_int_log.iloc[:, 1], color='red', alpha=0.2)
    ax[0].legend()
    ax[0].set_title(f"Fold {fold+1}: SARIMA Prediction (log-transformed)")
    ax[0].grid(alpha=0.3)

    # Original scale
    ax[1].plot(y_train.index[-50:], np.expm1(y_train[-50:]), label="Train", color="steelblue")
    ax[1].plot(y_test.index, np.expm1(y_test), label="Observed", color="steelblue")
    ax[1].plot(preds_log.index, np.expm1(preds_log), label="Predicted", color="orange")
    ax[1].legend()
    ax[1].set_title(f"Fold {fold+1}: SARIMA Prediction (Original Scale)")
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.pause(1)
    plot_path = BASE_DIR / "output" / "plots" / f"sarima_fold_{fold}.png"
    fig.savefig(plot_path)
    plt.close(fig=fig)
   
   
# MLflow logging stub
def log_sarima_result(model, summary_evaluation_metric, order, seasonal_order):
    
    df_eval = pd.DataFrame(summary_evaluation_metric)

    # Create output directory
    output_path = BASE_DIR / "output" / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save DataFrame to CSV
    csv_path = output_path / "sarima_model_cv_results.csv"
    df_eval.to_csv(csv_path, index=False)

    # Log the CSV file as an artifact
    mlflow.log_artifact(str(csv_path), artifact_path='sarima/cv_results')
    
    cv_avg_mae_log_values = round(np.mean(summary_evaluation_metric['mae_list_log']), 4)
    cv_avg_rmse_log_values = round(np.mean(summary_evaluation_metric['rmse_list_log']), 4)
    cv_avg_smape_log_values = round(np.mean(summary_evaluation_metric['smape_list_log']), 4)
    cv_avg_mae_original_values = round(np.mean(summary_evaluation_metric['mae_list_original']), 4)
    cv_avg_rmse_original_values = round(np.mean(summary_evaluation_metric['rmse_list_original']), 4)
    cv_avg_smape_original_values = round(np.mean(summary_evaluation_metric['smape_list_original']), 4)
    
             
    mlflow.log_metric("rmse", cv_avg_rmse_original_values)  # for filtering
    mlflow.log_metric("rmse_log", cv_avg_rmse_log_values)
    mlflow.log_metric("mae", cv_avg_mae_original_values)
    mlflow.log_metric("mae_log", cv_avg_mae_log_values)
    mlflow.log_metric("smape", cv_avg_smape_original_values)
    mlflow.log_metric("smape_log", cv_avg_smape_log_values)

    
    mlflow.log_metric("AIC", model.aic)
    mlflow.log_metric("BIC", model.bic)
    mlflow.log_metric("HQIC", model.hqic)
    
    mlflow.log_param("sarima_order", order)
    mlflow.log_param("sarima_seasonal_order", seasonal_order)

            
    mlflow.log_text(model.summary().as_text(), "model_summary.txt")

    
    mlflow.log_param("model_type", "SARIMA")
    mlflow.log_text(str(model), "model_summary.txt")
    
        
# CV loop
def sarima_cv(y_log, order, seasonal_order, forecast_horizon, n_splits=5):
    
    mae_list_log, rmse_list_log, smape_list_log = [], [], []
    mae_list_original, rmse_list_original, smape_list_original = [], [], []
    
    # TimeSeriesSplit with 7-day test windows to simulate weekly forecasting
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=forecast_horizon)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(y_log)):
        
        y_train = y_log.iloc[train_idx]
        y_test = y_log.iloc[test_idx]

        model = SARIMAX(y_train, order=order, seasonal_order=seasonal_order,
                        enforce_stationarity=False, enforce_invertibility=False)
        model_fit = model.fit(disp=False)

        forecast_obj = model_fit.get_forecast(steps=len(y_test))
        preds_log = forecast_obj.predicted_mean
        conf_int_log = forecast_obj.conf_int()
        preds_orig = np.expm1(preds_log)

        evaluation_metric_dict = evaluate_forecast(y_test, preds_log)
        
        rmse_list_log.append(evaluation_metric_dict['rmse_log'])
        mae_list_log.append(evaluation_metric_dict['mae_log'])
        smape_list_log.append(evaluation_metric_dict['smape_log'])

        rmse_list_original.append(evaluation_metric_dict['rmse_original'])
        mae_list_original.append(evaluation_metric_dict['mae_original'])
        smape_list_original.append(evaluation_metric_dict['smape_original'])
       
        plot_forecast(y_train, y_test, preds_log, conf_int_log, fold)
        
    summary_evaluation_metric = {"fold":list(range(1, n_splits+1)), 
                                  "rmse_list_log": rmse_list_log, 
                                      "mae_list_log": mae_list_log,
                                      "smape_list_log":smape_list_log,
                                      "rmse_list_original": rmse_list_original,
                                      "mae_list_original": mae_list_original,
                                      "smape_list_original": smape_list_original,}

    return model_fit, summary_evaluation_metric

def load_data():

    data_path = BASE_DIR / "data" / "processed" / "daily_tickets_2022_25.parquet"

    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_parquet(data_path)
    except Exception as e:
        raise RuntimeError(f"Unable to load data {e}")         
    
    return data


# === Config Loader ===
def load_config(config_path: str) -> dict:
    config_file = BASE_DIR / config_path
    with open(config_file, "r") as file:
        return yaml.safe_load(file)

# Main entry
def main():

    config = load_config("config/config.yaml")
    sarima_cfg = config["sarima"]

    order = tuple(sarima_cfg["order"])
    seasonal_order = tuple(sarima_cfg["seasonal_order"])
    forecast_horizon = sarima_cfg.get("test_size", 7)

    data_path = BASE_DIR / "data" / "processed" / "daily_tickets_2022_25.parquet"
    df = pd.read_parquet(data_path)
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df.set_index("Timestamp", inplace=True)
    df.index.freq = 'D'

    y = df["Redemption Count"]
    y_log = np.log1p(y)
    
    with mlflow.start_run(run_name="SARIMA_Modeling"):

        model, summary_metric = sarima_cv(y_log, order, seasonal_order, forecast_horizon, n_splits=5)
        
        
        log_cv_results(
            model_name="SARIMA",
            model_object=model,
            summary_metrics=summary_metric,
            order=config["sarima"]["order"],
            seasonal_order=config["sarima"]["seasonal_order"],
            base_dir=BASE_DIR,
            artifact_prefix="sarima"
        )
   
        # log_sarima_result(model_fit, summary_metric, order, seasonal_order)
        # mlflow.log_metric("rmse", round(np.mean(summary_metric["rmse_list_original"])), 4)  # or use np.mean(...) if preferred
        # mlflow.log_param("model", "tree")

        mlflow.log_artifacts(BASE_DIR / "output" / "plots", artifact_path="plots")
  

if __name__ == "__main__":
    main()