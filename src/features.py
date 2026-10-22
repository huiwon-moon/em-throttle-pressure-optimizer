import numpy as np
import pandas as pd
from typing import List, Optional

import sys
import os
sys.path.append(os.path.abspath(".."))

from config.columns import Columns

def derived_variable(data: pd.DataFrame):
    
    df = data.copy()
    
    if 'time' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['time']):
        df['time'] = pd.to_datetime(df['time'])
    if 'NUM' in df.columns:
        df['NUM'] = df['NUM'].astype('category')

    # 오일 제한 조건
    df['AB'] = (df[Columns.AB_cols].sum(axis=1) >= 2).astype(int)
    df['CD'] = (df[Columns.CD_cols].sum(axis=1) >= 2).astype(int)
    df['EF'] = (df[Columns.EF_cols].sum(axis=1) >= 2).astype(int)
    df['OIL_ELEV'] = df[['AB', 'CD', 'EF']].sum(axis=1)

    df[Columns.FD_cols] = np.where(df[Columns.FD_cols] < 0, 0.0, df[Columns.FD_cols])
    df['FD'] = df[Columns.FD_cols].sum(axis=1)
    df['Total_COAL'] = df['FD'] * 0.476
    df['SUM_COAL_OIL'] = df['Total_COAL'] * 10 + df[Columns.OIL]
    df['Total_COAL_x_10'] = df['Total_COAL'] * 10
    df['COAL_Ratio'] = df['Total_COAL_x_10'] / df['SUM_COAL_OIL']
    df['OIL_Ratio'] = df[Columns.OIL] / df['SUM_COAL_OIL']

    origin_cols = [Columns.OIL, 'FD']
    derived_cols = ['Total_COAL', 'SUM_COAL_OIL', 'Total_COAL_x_10','COAL_Ratio', 'OIL_Ratio']

    feature_cols = origin_cols + derived_cols
    
    return df, feature_cols

def add_features(data: pd.DataFrame, feature_cols: list, lag_steps: int, groupby_col: str = 'caseID'):
    df = data.copy()
    new_cols = {}

    if groupby_col in df.columns:
        grouped = df.groupby(groupby_col)
    else:
        grouped = None

    for col in feature_cols:
        group_col = grouped[col] if grouped else df[col]
        
        for i in range(1, lag_steps + 1):
            new_cols[f'{col}_current_diff_{i}'] = group_col.diff(i).fillna(0.0)

        spans = [5, 10]
        for s in spans:
            new_cols[f'{col}_rolling_sum_{s}'] = (
                group_col.transform(lambda x: x.shift(1).rolling(window=s, min_periods=1).sum())
                .fillna(0.0)
            )
    
    new_df = pd.DataFrame(new_cols, index=df.index)
    df = pd.concat([df, new_df], axis=1)
    
    df = df.dropna().reset_index(drop=True)
    
    feature_cols += list(new_cols)
    
    return df, feature_cols


def create_target_deltas(data: pd.DataFrame, target_col: str, groupby_col: str = 'caseID', horizons: Optional[List[int]] = None):
    df = data.copy()
    target_delta_cols = []

    if groupby_col in df.columns:
        grouped = df.groupby(groupby_col)
    else:
        grouped = df

    if horizons is None:
        col_name = f"target_delta"
        past_val = grouped[target_col].shift(1)
        df[col_name] = df[target_col] - past_val
        target_delta_cols.append(col_name)

    else:
        for h in horizons:
            col_name = f"target_delta_{h}"
            future_val = grouped[target_col].shift(-h)
            df[col_name] = future_val - df[target_col]
            target_delta_cols.append(col_name)

    df = df.dropna(subset=target_delta_cols).reset_index(drop=True)

    return df, target_delta_cols