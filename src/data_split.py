from pathlib import Path
import os
import pandas as pd
from typing import List

def mode_split(data: pd.DataFrame, mode: str):
    mode_map = {'WARM': 0, 'HOT': 1}
    if mode not in mode_map:
        raise ValueError(
            f"지원하지 않는 mode입니다: {mode}. ('WARM' 또는 'HOT' 사용 가능)"
        )
    return data[data['mode'] == mode_map[mode]]


def split_time_series_data(
        data: pd.DataFrame,
        output_dir: str = '../data',
        group_col: str = 'caseID',
        test_ratio: float = 0.2,
        ):

    os.makedirs(output_dir, exist_ok=True)
    df = data.copy()

    case_summary = df.groupby(group_col)['NUM'].first().reset_index()

    train_cases = []
    test_cases = []

    # 호기별로 반복하여 순차 분할
    for _, group in case_summary.groupby('NUM'):
        unique_cases = group[group_col].values

        split_idx = int(len(unique_cases) * (1 - test_ratio))

        train_cases.extend(unique_cases[:split_idx])
        test_cases.extend(unique_cases[split_idx:])

    train_df = df[df[group_col].isin(train_cases)]
    test_df = df[df[group_col].isin(test_cases)]

    return train_df, test_df

# --------------------------------------------------
# Train / Test 분할
# --------------------------------------------------
DATA_DIR = Path(r'..\data')

# 1) TP 기준 데이터 분할
tp_df = pd.read_csv(DATA_DIR / 'TP_data_preprocess.csv')

MODES = ['WARM', 'HOT']
for mode in MODES:
    
    filtered_df = mode_split(tp_df, mode)

    train_df, test_df = split_time_series_data(
        filtered_df, group_col='caseID', test_ratio=0.2
    )

    train_df.to_csv(DATA_DIR / f'{mode}_TP_train_df.csv', index=False)
    test_df.to_csv(DATA_DIR / f'{mode}_TP_test_df.csv', index=False)

# 2) MST 기준 데이터 분할
mst_df = pd.read_csv(DATA_DIR / 'MST_data_preprocess.csv')

for mode in MODES:

    filtered_df = mode_split(mst_df, mode)

    tp_test_df = pd.read_csv(DATA_DIR / f'{mode}_TP_test_df.csv')
    test_case_list = tp_test_df['caseID'].unique()

    train_df = filtered_df[~filtered_df['caseID'].isin(test_case_list)]
    test_df = filtered_df[filtered_df['caseID'].isin(test_case_list)]

    train_df.to_csv(DATA_DIR / f'{mode}_MST_train_df.csv', index=False)
    test_df.to_csv(DATA_DIR / f'{mode}_MST_test_df.csv', index=False)