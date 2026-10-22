import pandas as pd

import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from config.columns import Columns

ffill_cols = Columns.AB_cols + Columns.CD_cols + Columns.EF_cols + Columns.FD_cols + [Columns.BLR_FIRE_ON]
linear_cols = [Columns.TP, Columns.MST, Columns.OIL]

def imputation():
    df = pd.read_csv(r'data\origin_data\origin_data.csv')
    df.columns = df.columns.str.replace(':', '_')
    
    grouped = df.groupby('caseID')
    for col in ffill_cols:
        df[col] = grouped[col].ffill()

    for col in linear_cols:
        df[col] = grouped[col].transform(
            lambda group: group.interpolate(method='linear')
        )

    nan_case_ids = df[df.isna().any(axis=1)]['caseID'].unique()
    cleaned_df = df[~df['caseID'].isin(nan_case_ids)]

    cleaned_df.to_csv('data/data_imputation.csv', index=False)

if __name__ == "__main__":
  imputation()