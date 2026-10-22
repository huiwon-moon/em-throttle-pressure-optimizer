from pathlib import Path
import pandas as pd
import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".."))

from config.columns import Columns


DATA_DIR = Path(r'../data')
IMPUTATION_FILE_NAME = 'data_imputation.csv'

def project_section():
    df = pd.read_csv(DATA_DIR / IMPUTATION_FILE_NAME)
    df['time'] = pd.to_datetime(df['time'])

    sampled_df = (
        df.groupby('caseID')
        .apply(lambda x: x.set_index('time').asfreq('30s'))
        .reset_index()
    )

    filtered_dfs = []
    for case_id, group in sampled_df.groupby('caseID'):
      cond_start = group[Columns.BLR_FIRE_ON] == 1
      cond_end = group[Columns.TP] > 84

      if cond_start.any() and cond_end.any():
        start_idx = cond_start.idxmax()
        end_idx = cond_end.idxmax()

        if start_idx <= end_idx:
          max_idx = group.index[-1]
          target_end_idx = min(end_idx + 5, max_idx)
          # 조건에 맞는 구간만 추출해서 리스트에 담기
          filtered_dfs.append(group.loc[start_idx:target_end_idx])

    # 4. 최종 데이터프레임으로 결합
    if filtered_dfs:
      final_df = pd.concat(filtered_dfs, ignore_index=True)
    else:
      final_df = pd.DataFrame(columns=sampled_df.columns)
    
    final_df.to_csv('../data/data.csv', index=False)

    return None


SECTION_FILE_NAME = 'data.csv'

TP_THRESHOLD = 4.2
MST_THRESHOLD = 5.0
LIMIT_VAL = 84.0

def remove_trip_cases(data: pd.DataFrame, target_col: str, threshold: float, limit_val: float = 84.0) -> pd.DataFrame:
    
    # caseID별 shift 연산 (30초 단위 데이터에서 1분 간격으로 확인)
    target_acc = data[target_col] - data.groupby('caseID')[target_col].shift(2)
    
    # 조건 위배 케이스 추출
    invalid_mask = (target_acc > threshold) & (data[target_col] <= limit_val)
    del_cases = data.loc[invalid_mask, 'caseID'].unique()
    
    # 정상 케이스만 반환
    return data[~data['caseID'].isin(del_cases)]


def run_preprocess():
    input_path = DATA_DIR / SECTION_FILE_NAME
    if not input_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {input_path.resolve()}")

    print(f"1. 원본 데이터 로드: {input_path}")
    df = pd.read_csv(input_path)
    df.columns = df.columns.str.replace(':', '_')

    # 음수 값을 0.0으로 보정
    df[Columns.FD_cols] = np.where(df[Columns.FD_cols] < 0, 0.0, df[Columns.FD_cols]) # COAL FEEDER 총투입량

    # 조건에 맞는 케이스 ID 필터링 (TP < 80 이전에 FD 투입이 발생한 caseID)
    condition_mask = (df[Columns.TP] < 80) & (df[Columns.FD_cols].sum(axis=1) > 0) # TP 80k 이전에 COAL 투입된 경우
    valid_case_ids = df.loc[condition_mask, 'caseID'].unique()

    df_base = df[df['caseID'].isin(valid_case_ids)].copy()
    base_out_path = DATA_DIR / 'data_preprocess.csv'
    df_base.to_csv(base_out_path, index=False)
    print(f"기본 전처리 저장 완료")

    df_tp_valid = remove_trip_cases(df_base, target_col=Columns.TP, threshold=TP_THRESHOLD, limit_val=LIMIT_VAL)
    tp_out_path = DATA_DIR / 'TP_data_preprocess.csv'
    df_tp_valid.to_csv(tp_out_path, index=False)
    print(f"TP 전처리 저장 완료")

    df_mst_valid = remove_trip_cases(df_tp_valid, target_col=Columns.MST, threshold=MST_THRESHOLD, limit_val=LIMIT_VAL)
    mst_out_path = DATA_DIR / 'MST_data_preprocess.csv'
    df_mst_valid.to_csv(mst_out_path, index=False)
    print(f"MST 전처리 저장 완료")

if __name__ == '__main__':
    project_section() # 승온승압 구간 데이터 전처리
    run_preprocess() # Trip 조건 데이터 전처리