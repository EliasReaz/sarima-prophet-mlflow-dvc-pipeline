    # utils/mlflow_logger.py

import pandas as pd
import numpy as np
import mlflow
from pathlib import Path

def log_cv_results(
    model_name: str,
    model_object,
    summary_metrics: dict,
    order=None,
    seasonal_order=None,
    holiday_region=None,
    base_dir: Path = Path("."),
    artifact_prefix: str = "model"
):
    # Convert to DataFrame
    df_eval = pd.DataFrame(summary_metrics)

    # Save fold-wise results
    results_dir = base_dir / "output" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    csv_path = results_dir / f"{artifact_prefix}_cv_results.csv"
    df_eval.to_csv(csv_path, index=False)
    mlflow.log_artifact(str(csv_path), artifact_path=f"{artifact_prefix}/cv_results")

    # Log fold-wise metrics
    for i in range(len(summary_metrics["fold"])):
        mlflow.log_metric(f"fold_{i+1}_rmse_log", round(summary_metrics["rmse_list_log"][i], 4))
        mlflow.log_metric(f"fold_{i+1}_rmse_original", round(summary_metrics["rmse_list_original"][i], 4))
        mlflow.log_metric(f"fold_{i+1}_mae_log", round(summary_metrics["mae_list_log"][i], 4))
        mlflow.log_metric(f"fold_{i+1}_mae_original", round(summary_metrics["mae_list_original"][i], 4))
        mlflow.log_metric(f"fold_{i+1}_smape_log", round(summary_metrics["smape_list_log"][i], 4))
        mlflow.log_metric(f"fold_{i+1}_smape_original", round(summary_metrics["smape_list_original"][i], 4))


    # Log averages
    mlflow.log_metric("rmse", round(np.mean(summary_metrics["rmse_list_original"]), 4))
    mlflow.log_metric("rmse_log", round(np.mean(summary_metrics["rmse_list_log"]), 4))
    mlflow.log_metric("mae", round(np.mean(summary_metrics["mae_list_original"]), 4))
    mlflow.log_metric("mae_log", round(np.mean(summary_metrics["mae_list_log"]), 4))
    mlflow.log_metric("smape", round(np.mean(summary_metrics["smape_list_original"]), 4))
    mlflow.log_metric("smape_log", round(np.mean(summary_metrics["smape_list_log"]), 4))

    # Log model summary
    if hasattr(model_object, "summary"):
        mlflow.log_text(model_object.summary().as_text(), "model_summary.txt")
    else:
        mlflow.log_text(str(model_object), "model_summary.txt")

    # Log params
    mlflow.log_param("model", "tree")
    mlflow.log_param("model_type", model_name)
    if order:
        mlflow.log_param("sarima_order", order)
    if seasonal_order:
        mlflow.log_param("sarima_seasonal_order", seasonal_order)
    if holiday_region:
        mlflow.log_param("holiday_region", holiday_region)


