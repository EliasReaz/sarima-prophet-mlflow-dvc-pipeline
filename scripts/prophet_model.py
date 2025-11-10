import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.model_selection import TimeSeriesSplit
import mlflow
from pathlib import Path
from IPython.display import display
# Ontario holidays setup
import holidays
import os
from dotenv import load_dotenv
from utils import evaluate_forecast

load_dotenv()

os.makedirs("output/plots", exist_ok=True)
    
os.environ['MLFLOW_TRACKING_URI'] = os.getenv("MLFLOW_TRACKING_URI")
os.environ['MLFLOW_TRACKING_USERNAME'] = os.getenv("MLFLOW_TRACKING_USERNAME")
os.environ['MLFLOW_TRACKING_PASSWORD'] = os.getenv("MLFLOW_TRACKING_PASSWORD")

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))

mlflow.set_experiment("Prophet_Modeling_Experiment")

try:
    BASE_DIR = Path(__file__).resolve().parent.parent
except NameError:
    BASE_DIR = Path().cwd().parent


def get_ontario_holidays(years):
    ontario_holidays = holidays.CA(prov='ON', years=years)
    return pd.DataFrame([
        {'ds': date, 'holiday': name}
        for date, name in ontario_holidays.items()
    ])
    
# Plotting
def plot_forecast(model, train_df, test_df, forecast_test, fold):
    
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    # Log-transformed
    ax[0].plot(train_df['ds'][-50:], train_df['y'][-50:], label="Train", color='steelblue')
    ax[0].plot(test_df['ds'], test_df['y'], label="Observed", color='blue')
    ax[0].plot(forecast_test.index, forecast_test['yhat'], label="Predicted", color='orange')
    ax[0].fill_between(forecast_test.index, forecast_test['yhat_lower'], forecast_test['yhat_upper'], color='red', alpha=0.2)
    ax[0].legend()
    ax[0].set_title(f"Fold {fold+1}: Prophet Prediction (log-transformed)")
    ax[0].grid(alpha=0.3)

    # Original scale
    ax[1].plot(train_df['ds'][-50:], np.expm1(train_df['y'][-50:]), label="Train", color='steelblue')
    ax[1].plot(test_df['ds'], np.expm1(test_df['y']), label="Observed", color='blue')
    ax[1].plot(forecast_test.index, np.expm1(forecast_test['yhat']), label="Predicted", color='orange')
    ax[1].legend()
    ax[1].set_title(f"Fold {fold+1}: Prophet Prediction (Original Scale)")
    ax[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.pause(1)
    ## save plot
    plot_path = BASE_DIR / "output" / "plots" / f"prophet_fold_{fold}.png"
    fig.savefig(plot_path)
    plt.close(fig=fig)
    
     
# MLflow logging stub
def prophet_log_model(model, summary_evaluation_metric):
    
    df_eval = pd.DataFrame(summary_evaluation_metric)

    # Create output directory
    output_path = BASE_DIR / "output" / "results"
    output_path.mkdir(parents=True, exist_ok=True)

    # Save DataFrame to CSV
    csv_path = output_path / "prophet_model_cv_results.csv"
    df_eval.to_csv(csv_path, index=False)

    # Log the CSV file as an artifact
    mlflow.log_artifact(str(csv_path), artifact_path='prophet/cv_results')
             
    mlflow.log_metric("MAE_mean (log scale forecast)", np.mean(summary_evaluation_metric['mae_list_log']))
    mlflow.log_metric("RMSE_mean (log scale forecast)", np.mean(summary_evaluation_metric['rmse_list_log']))  
    mlflow.log_metric("SMAPE_mean (log scale forecast)", np.mean(summary_evaluation_metric['smape_list_log']))
    
    mlflow.log_metric("MAE_mean (original scale forecast)", np.mean(summary_evaluation_metric['mae_list_original']))
    mlflow.log_metric("RMSE_mean (original scale forecast)", np.mean(summary_evaluation_metric['rmse_list_original']))
    mlflow.log_metric("SMAPE_mean (original scale forecast)", np.mean(summary_evaluation_metric['smape_list_original']))
    
    mlflow.log_param("model_type", "Prophet")
    mlflow.log_param("holiday_region", "Ontario")
    mlflow.log_text(str(model), "model_summary.txt")
    
        
# CV loop
def prophet_cv(df, holiday_df, n_splits=5):
    
    mae_list_log, rmse_list_log, smape_list_log = [], [], []
    mae_list_original, rmse_list_original, smape_list_original = [], [], []
    
    # TimeSeriesSplit with 7-day test windows to simulate weekly forecasting
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=7)

    for fold, (train_idx, test_idx) in enumerate(tscv.split(df)):
        
        train_df = df.iloc[train_idx]
        test_df = df.iloc[test_idx]

        model = Prophet(
            growth='linear',
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            holidays=holiday_df
        )
        model.fit(train_df)

        future = model.make_future_dataframe(periods=len(test_df), freq='D')
        forecast = model.predict(future)
        # forecast_test = forecast.set_index('ds').loc[test_df['ds'].values]
        forecast_test = forecast.set_index('ds').reindex(test_df['ds'])

        evaluation_metric_dict = evaluate_forecast(test_df['y'], forecast_test['yhat'])
        
        rmse_list_log.append(evaluation_metric_dict['rmse_log'])
        mae_list_log.append(evaluation_metric_dict['mae_log'])
        smape_list_log.append(evaluation_metric_dict['smape_log'])

        rmse_list_original.append(evaluation_metric_dict['rmse_original'])
        mae_list_original.append(evaluation_metric_dict['mae_original'])
        smape_list_original.append(evaluation_metric_dict['smape_original'])
       
        plot_forecast(model, train_df, test_df, forecast_test, fold)
        
    summary_evaluation_mertric = {"rmse_list_log": rmse_list_log, 
                                      "mae_list_log": mae_list_log,
                                      "smape_list_log":smape_list_log,
                                      "rmse_list_original": rmse_list_original,
                                      "mae_list_original": mae_list_original,
                                      "smape_list_original": smape_list_original,}

    return model, summary_evaluation_mertric

def load_data():

    data_path = BASE_DIR / "data" / "processed" / "daily_tickets_2022_25.parquet"

    print(f"Loading data from: {data_path}")
    try:
        data = pd.read_parquet(data_path)
    except Exception as e:
        raise RuntimeError(f"Unable to load data {e}")         
    
    return data

# Main entry
def main():

    data = load_data()
    data["log_transformed_redem"] = np.log1p(data["Redemption Count"])
    data["log_transformed_sales"] = np.log1p(data["Sales Count"])
    # Optional feature: daily change in log-redemption (not used in Prophet)
    # data["diff_log_redem"] = data["log_transformed_redem"].diff()

    data = data.dropna().reset_index(drop=True)

    display(data.head())
    
    df = data.copy()  # or copy original dataframe

    df = df[['Timestamp', 'log_transformed_redem', 'log_transformed_sales']].dropna()

    df.rename(columns={"Timestamp":"ds", "log_transformed_redem":"y"}, inplace=True)
    
    df['ds'] = pd.to_datetime(df['ds'])
    
    years = df['ds'].dt.year.unique()
    
    holiday_df = get_ontario_holidays(years)
    
    with mlflow.start_run(run_name="Prophet_Modeling"):

        model, summary_metric = prophet_cv(df, holiday_df, n_splits=5)
   
        prophet_log_model(model, summary_metric)
        mlflow.log_artifacts(BASE_DIR / "output" / "plots", artifact_path="plots")
  

if __name__ == "__main__":
    main()