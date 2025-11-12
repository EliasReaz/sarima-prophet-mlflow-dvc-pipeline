## Toronto Ferry Ticket Sales Forecasting using SARIMA, Prophet: along with DVC and Mlflow

This project demonstrates a complete time series forecasting workflow using SARIMA and Prophet. It includes preprocessing, exploratory analysis, model fitting, and forecasting for a 30-day horizon. The pipeline is modular, reproducible, and designed for extensibility.

## Project structure

```text
sarima-prophet-mlflow-dvc-pipeline/
├── .dvc/                                         # DVC configuration
├── .venv/                                        # Virtual environment
├── data/
│   ├── raw/
│   │   └── TorontoIslandFerryTicketCount.csv
│   └── processed/
│       └── daily_tickets_2022_25.parquet
├── models/
│   ├── sarima_model.pkl
│   └── prophet_model.pkl
├── notebooks/
│   ├── exploratory_analysis.ipynb
│   ├── sarima_exploration.ipynb
│   └── prophet_exploration.ipynb
├── output/
│   ├── plots/
│   └── results/
│       ├── sarima_cv_results.csv
│       └── prophet_cv_results.csv
├── scripts/
│   ├── __init__.py
│   ├── sarima_model.py
│   └── prophet_model.py
├── utils/
│   ├── __init__.py
│   ├── evaluate_forecast.py
│   └── mlflow_logger.py
├── .env
├── requirements.txt
└── README.md
```
### Toronto Ferry Ticket Sales Redemption Data

![Daily redemption count from 2022 onwards](./output/plots/daily_redemption_count_2022_onwards.png)

*Figure: Daily redemption count from 2022 onwards.*
- We observe yearly seasonality - winter low and summer high 
- There is also weekly seasonality - weekend high
- Presence of sudden spikes are visible

### Basic Statistics 

* The mean ticket count is approximately 3,700, while the median (50th percentile) is around 1,600, indicating a right-skewed distribution. This suggests that a few extreme high values are pulling the mean above the median.

* A high standard deviation of 4,440 tickets reflects a wide dispersion in daily ticket sales, pointing to substantial variability across days. 

### Log-transformation for variance stability and first-difference to remove trend and make the series stationary

![Log-transformation](./output/plots/log_transformed_redemption_counts.png)
*Figure: log-transformed daily redemption stabilizes the variance.

### First differencing of log-transformed series
![First difference](./output/plots/first_differencing_of_logtransformed_series.png)
*Figure: We see that the series mean is zero, with almost stable variance and sudden spikes*

### One-step differencing of log-transformed time-series

* Log transformation stabilizes the variance and reduce the impact of spikes

* one-step differencing now removes the trend, making the time-series stationary. 

* one-step differencing represents the relative changes or growth rate in redemption counts from one day to the next.

* it centers around zero with occasional spkies and dips

### ACF and PACF plots
![ACF and PACF plot](./output/plots/acf_pacf_first_diff_of_logtransformed_data.png)

### Auto-correlation and Partial Auto-correlation of 1-step differencing of log-transformed time series

* PACF suggests significant auto-correlation at lag 1, 2, 3, 4, 5 and 6. After that the correlation dies. We can try with AR(p) p from 1 to 6 and see the BIC

* ACF suggests signification relationships at lag 1 and 2. q could be 1 or 12

### Augmented Dickey-Fuller test

```python
from statsmodels.tsa.stattools import adfuller
import pandas as pd

# Run the test
result = adfuller(data_diff['diff_redemption_count'].dropna())

# Organize results
adf_output = {
    'ADF Statistic': result[0],
    'p-value': result[1],
    'Number of Lags Used': result[2],
    'Number of Observations Used': result[3],
    'Critical Values': result[4],
    'IC Best (AIC)': result[5]
}

# Display as DataFrame
adf_df = pd.DataFrame.from_dict(adf_output, orient='index', columns=['Value'])
print(adf_df)
```
```
                                                                         Value
ADF Statistic                                                       -16.285116
p-value                                                                    0.0
Number of Lags Used                                                         12
Number of Observations Used                                               1264
Critical Values                                 {'1%': -3.4355, '5%': -2.8638}
IC Best (AIC)                                                      2274.831503
```

* p-value = 0 suggestes that the data does not provide enough evidence in favor of the null hypothesis that the time series has a unit root - the series is non-staionary.

* So we reject the Null Hypothesis that the series is non-stationary. 

* 1-step differencing of log-transformation has made the seties stationary.

### SARIMA






