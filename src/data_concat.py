from pathlib import Path
import pandas as pd
import numpy as np

import re
import os
os.chdir(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

def horizontal_concat_with_different_filenames():

    base_dir = Path("data/origin_data")
    output_base_dir = base_dir / "concat_data"

    machines = ['#1', '#2', '#3', '#4']
    join_key = 'time'

    source_file_map = {
        "tp": "TP.csv",
        "mst": "MS_TEMP.csv",
        "elev": "elev_flag.csv",
        "coal_fdr": "ignition.csv",
    }

    output_file_name = "STARTSTOP.csv"

    for machine in machines:
        first_source = list(source_file_map.keys())[0]
        first_machine_dir = base_dir / first_source / machine
        if not first_machine_dir.exists():
            continue

        for date_dir in first_machine_dir.iterdir():
            if not date_dir.is_dir():
                continue

            date_str = date_dir.name

            dfs = []
            for source, file_name in source_file_map.items():
                file_path = base_dir / source / machine / date_str / file_name
                if file_path.exists():
                    try:
                        df = pd.read_csv(file_path)
                        dfs.append(df)
                    except Exception as e:
                        print(f"파일 읽기 오류 ({file_path}): {e}")
                else:
                    print(f"파일이 존재하지 않습니다: {file_path}")

            if not dfs:
                continue

            try:
                combined_df = dfs[0]

                for next_df in dfs[1:]:
                    if join_key in combined_df.columns and join_key in next_df.columns:

                        overlap_cols = (
                            set(combined_df.columns).intersection(set(next_df.columns))
                            - {join_key}
                        )

                        if overlap_cols:
                            next_df = next_df.drop(columns=list(overlap_cols))

                        combined_df = pd.merge(
                            combined_df, next_df, on=join_key, how='inner'
                        )

                    else:
                        cols_to_use = [
                            c for c in next_df.columns if c not in combined_df.columns
                        ]
                        if cols_to_use:
                            combined_df = pd.concat([combined_df, next_df[cols_to_use]], axis=1)

                output_dir = output_base_dir / machine / date_str
                output_dir.mkdir(parents=True, exist_ok=True)

                output_file = output_dir / output_file_name
                combined_df.to_csv(output_file, index=False)
                print(f"병합 완료: {output_file}")

            except Exception as e:
                print(f"데이터 병합 중 오류 발생 ({machine}/{date_str}): {e}")


def concat_all_machines():

    startup_mode_df = pd.read_excel(r'docs\운영예비초과정지입력(20251223_132103)_20260107.xlsx') # 기동 모드 데이터
    
    case_id = 0
    dfs = []
    for NUM in [1, 2, 3, 4]:
        
        startup_events = startup_mode_df[(startup_mode_df['호기(*)'] == NUM) & startup_mode_df['WARM/HOT\n(23년, 24년, 25년)\n125번'].notna()][['병입 출력재한 해제일시(*)', '병입 출력재한 해제일시(*).1', 'WARM\n(23, 24, 25)\n67번', 'HOT\n(23, 24, 25)\n58번']].copy()
        startup_events['base_datetime'] = pd.to_datetime(
            startup_events['병입 출력재한 해제일시(*)'].astype(str) + ' ' + startup_events['병입 출력재한 해제일시(*).1'].astype(str),
            utc=False  # tz-naive 생성
        ).dt.tz_localize(None)  # tz 완전 제거

        
        base_dir = Path(f"data/origin_data/concat_data/#{NUM}")
        for _, folder_name in enumerate(sorted(os.listdir(base_dir))):
            folder_path = os.path.join(base_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            csv_path = os.path.join(folder_path, "STARTSTOP.csv")
            if not os.path.exists(csv_path):
                continue

            df = pd.read_csv(csv_path)
            df['time'] = pd.to_datetime(df['time'])
            df['caseID'] = case_id
            df['NUM'] = NUM
            # ── 컬럼명 앞 숫자 두 자리 제거 (예: '11abc' → 'abc')
            df.columns = [re.sub(r"^\d{2}", "", c) for c in df.columns]
            
            for _, event in startup_events.iterrows():
                if event['HOT\n(23, 24, 25)\n58번'] == 'O':
                    hot_mask = (df['time'] >= event['base_datetime'] - pd.Timedelta(hours=5)) \
                                & (df['time'] <= event['base_datetime'] + pd.Timedelta(hours=5))
                    df.loc[hot_mask, 'mode'] = 1
                elif event['WARM\n(23, 24, 25)\n67번'] == 'O':
                    warm_mask = (df['time'] >= event['base_datetime'] - pd.Timedelta(hours=5)) \
                                & (df['time'] <= event['base_datetime'] + pd.Timedelta(hours=5))
                    df.loc[warm_mask, 'mode'] = 0

            dfs.append(df)

            case_id += 1

    merged_df = pd.concat(dfs, ignore_index=True)
    merged_df.to_csv(f"data/origin_data/origin_data.csv", index=False)


if __name__ == "__main__":
  horizontal_concat_with_different_filenames()
  concat_all_machines()