import os
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    return {
        'MAE': round(mae, 4),
        'MSE': round(mse, 4),
        'RMSE': round(rmse, 4)
    }

def prepare_true_values(
        test_df: pd.DataFrame,
        target_col: str,
        target_delta_cols: Optional[List[str]] = None
):
    if len(target_delta_cols) > 1:
        y_true = test_df[target_delta_cols].add(test_df[target_col], axis=0).reset_index(drop=True).values
    else:
        y_true = test_df[target_col].reset_index(drop=True).values
    return y_true

def performance_evaluate(
        test_df: pd.DataFrame,
        y_test: Union[np.ndarray, pd.Series],
        y_test_pred: Union[np.ndarray, pd.Series],
        model_name: str,
        target_cols_list: List[str],
        id_col: str = 'caseID'
):

    test_df = test_df.reset_index(drop=True)
    y_test = np.asarray(y_test)
    y_test_pred = np.asarray(y_test_pred)

    if y_test.ndim == 1:
        y_test = y_test.reshape(-1, 1)
    if y_test_pred.ndim == 1:
        y_test_pred = y_test_pred.reshape(-1, 1)
    
    records = []
    unique_ids = test_df[id_col].unique()

    for idx, uid in enumerate(unique_ids):
        mask = (test_df[id_col] == uid).values
        if not np.any(mask):
            continue

        first_row = test_df[mask].iloc[0]
        time_val = first_row.get('time', pd.NA)
        num_val = first_row.get('NUM', pd.NA)

        for col_idx, target_name in enumerate(target_cols_list):
            y_t = y_test[mask, col_idx]
            y_p = y_test_pred[mask, col_idx]

            metrics = calculate_metrics(y_t, y_p)

            record = {
                'time': time_val,
                'NUM': num_val,
                'Model': model_name,
                id_col: uid,
                'Target': target_name,
                **metrics
            }
            records.append(record)

    summary_df = pd.DataFrame(records)
    return summary_df


def save_performance_matrix(
        summary_df: pd.DataFrame,
        file_name: str,
        output_dir: str = 'results',
):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f'{file_name}.csv')
        
    file_exists = os.path.exists(filepath)

    summary_df.to_csv(
        filepath, 
        mode='a', 
        index=False, 
        header=not file_exists
    )

    if file_exists:
        print(f"Appended performance data to existing file: {filepath}")
    else:
        print(f"Created new performance matrix file: {filepath}")