def evaluate_forecast(y_true, y_pred, verbose=False):
    """
    Evaluate forecast performance using RMSE, MAE, and sMAPE in both log scale and original scale.

    This utility is designed for models (e.g., SARIMA, Prophet) that operate in log-transformed space.
    It computes error metrics in both the transformed (log) domain and the original scale using np.expm1.

    Parameters
    ----------
    y_true : array-like
        Ground truth values in log scale (e.g., np.log1p(actuals)).
    y_pred : array-like
        Forecasted values in log scale (e.g., np.log1p(predictions)).
    verbose : bool, optional
        If True, prints diagnostic info about skipped sMAPE cases due to zero denominators.

    Returns
    -------
    dict
        Dictionary containing the following keys:
        - 'rmse_log', 'mae_log', 'smape_log' : metrics in log scale
        - 'rmse_original', 'mae_original', 'smape_original' : metrics in original scale


    Notes
    -----
    - sMAPE is computed with masking to avoid division by zero.
    - Use this function in cross-validation loops or final evaluation to log metrics to MLflow.
    - Compatible with np.log1p / np.expm1 transformations for stable inverse scaling.
    """
    import numpy as np

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Convert back to original scale
    y_true_original = np.expm1(y_true)
    y_pred_original = np.expm1(y_pred)

    # RMSE
    rmse_log = np.sqrt(np.mean((y_true - y_pred) ** 2))
    rmse_original = np.sqrt(np.mean((y_true_original - y_pred_original) ** 2))

    # MAE
    mae_log = np.mean(np.abs(y_true - y_pred))
    mae_original = np.mean(np.abs(y_true_original - y_pred_original))

    # SMAPE in log scale
    denom_log = (np.abs(y_true) + np.abs(y_pred)) / 2
    valid_mask_log = denom_log != 0
    smape_log_values = np.abs(y_true[valid_mask_log] - y_pred[valid_mask_log]) / denom_log[valid_mask_log]
    smape_log = np.mean(smape_log_values) * 100 if smape_log_values.size > 0 else np.nan

    # SMAPE in original scale
    denom_orig = (np.abs(y_true_original) + np.abs(y_pred_original)) / 2
    valid_mask_orig = denom_orig != 0
    smape_orig_values = np.abs(y_true_original[valid_mask_orig] - y_pred_original[valid_mask_orig]) / denom_orig[valid_mask_orig]
    smape_original = np.mean(smape_orig_values) * 100 if smape_orig_values.size > 0 else np.nan

    if verbose:
        skipped_log = len(y_true) - np.count_nonzero(valid_mask_log)
        skipped_orig = len(y_true_original) - np.count_nonzero(valid_mask_orig)
        print(f"[SMAPE-log] Skipped {skipped_log} zero-denominator cases out of {len(y_true)}")
        print(f"[SMAPE-original] Skipped {skipped_orig} zero-denominator cases out of {len(y_true_original)}")

    return {
        "rmse_log": round(rmse_log, 4),
        "mae_log": round(mae_log, 4),
        "smape_log": round(smape_log, 4),
        "rmse_original": round(rmse_original,4),
        "mae_original": round(mae_original,4),
        "smape_original": round(smape_original,4)
    }
    