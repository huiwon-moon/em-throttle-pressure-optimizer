import os
import sys
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List
from tqdm import tqdm

from config.columns import Columns
from src.features import derived_variable, add_features
from src.model import LGBMMultiOutputModel, RidgeRegressionModel
from src.visualization import plot_and_save_all_trajectories_compare_optimal_line

def parse_args():
    parser = argparse.ArgumentParser(description="Control Value Optimization and Recommendation System")

    # 모델 파라미터 대상
    parser.add_argument('--mode', type=str, default='HOT', choices=['HOT', 'WARM'], help='기동 상태 선택')
    parser.add_argument('--caseID', type=int, default=116, choices=[116, 118], help='HOT: 116 / WARM: 118')
    parser.add_argument('--lead_time', type=int, default=66, choices=[66, 71], help='목표 도달 시간(분)') # WARM 기동 69로 설정 시 Trip 조건으로 인해 중지

    # 데이터 경로 지정
    parser.add_argument('--data_dir',type=str, default='data', help='데이터 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='models', help='학습된 모델 저장 디렉토리 경로')

    parser.add_argument('--lag_steps', type=int, default=20, help='window size')
    parser.add_argument('--target_tp', type=float, default=84.0, help="목표 TP 값")
    parser.add_argument('--tp_threshold', type=float, default=2.1, help="Trip 조건 (분당 4.2)")

    parser.add_argument('--max_coal', type=float, default=86.0, help="최대 석탄 제약값")
    parser.add_argument('--max_fuel', type=float, default=600.0, help="최대 연료 제약값")

    parser.add_argument('--action_type', type=str, default='real', choices=['real', 'simulation'], help="real: 1 Step, simulation: total")

    parser.add_argument('--output_dir', type=str, default='results', help='시뮬레이션 결과 저장 경로')
    parser.add_argument('--figure_dir', type=str, default='fig/recommend_sim', help='그래프 저장 경로')
    
    args, _ = parser.parse_known_args()

    return args

def optimal_band(
        cur_time: pd.Timestamp,
        optimal_band_df: pd.DataFrame
    ):

    optimal_band_columns = ['pressure_upper', 'pressure_lower', 'pressure_matched_raw',
                            'oil_upper', 'oil_lower', 'oil_matched_raw', 
                            'FD_upper', 'FD_lower','FD_matched_raw', 'time']

    optimal_df = optimal_band_df[optimal_band_df['time'] == cur_time + pd.Timedelta(seconds=30)]

    if optimal_df.empty or pd.isna(optimal_df['oil_matched_raw'].values[0]):
        last_row = optimal_band_df.iloc[-1]
        optim_oil = last_row['oil_matched_raw']
        optim_coal = last_row['FD_matched_raw']
    else:
        optim_oil = optimal_df['oil_matched_raw'].values[0]
        optim_coal = optimal_df['FD_matched_raw'].values[0]

    return optim_oil, optim_coal

def update_state(window_df, rec_controls, next_TP, args):
    update_df = window_df.iloc[1:].copy()
    update_df = pd.concat([update_df, update_df.iloc[[-1]]], ignore_index=True)

    last_idx = update_df.index[-1]

    update_df.loc[last_idx, 'time'] = update_df.loc[last_idx, 'time'] + pd.Timedelta(seconds=30)
    update_df.loc[last_idx, 'FD'] = float(rec_controls[0])
    update_df.loc[last_idx, Columns.OIL] = float(rec_controls[1])
    update_df.loc[last_idx, Columns.TP] = float(next_TP)

    update_df, orig_derived_feature_cols = derived_variable(update_df)

    rec_df, _ = add_features(update_df, feature_cols=orig_derived_feature_cols, lag_steps=args.lag_steps)
    rec_state = rec_df.iloc[-1].copy()

    return update_df, rec_state

class ControlOptimizer:
    def __init__(self, TP_model, MST_model): 
        self.TP_model = TP_model
        self.MST_model = MST_model

    def objective(self, window_df, rec_controls, feature_cols, target_TP_rate, next_TP, args):
        _, rec_state = update_state(window_df, rec_controls, next_TP, args)
        X_sim = pd.DataFrame([rec_state[feature_cols + ['NUM']]]).astype(float)
        X_sim['NUM'] = X_sim['NUM'].astype('category')
        
        TP_pred_multi = self.TP_model.predict(X_sim, rec_state[Columns.TP])
        MST_pred = self.MST_model.predict(X_sim, rec_state[Columns.MST])

        TP_rate = TP_pred_multi[0] - rec_state[Columns.TP]
        MST_rate = MST_pred - rec_state[Columns.MST]

        diffs = TP_rate - target_TP_rate
        mae_loss = np.mean(np.abs(diffs))

        TP_over_penalty = 0
        if TP_rate[0] - target_TP_rate[0] > 1e-7:
            TP_over_penalty = TP_rate[0] - target_TP_rate[0]
        
        MST_over_penalty = 0
        if MST_rate > 2.5:
            MST_over_penalty = MST_rate - 2.5
        
        return mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi[0], MST_pred[0]

def predict_single_step(window_df, optimal_band_df, start_time, models, args):
    
    TP_model, MST_model = models['TP_model'], models['MST_model']
    HORIZONS = [1, 2, 3, 4, 5, 6]

    # 파생변수 및 특징량 생성
    window_df, orig_derived_feature_cols = derived_variable(window_df)
    init_df, feature_cols = add_features(window_df, feature_cols=orig_derived_feature_cols, lag_steps=args.lag_steps)
    current_state = init_df.iloc[-1].copy()

    cur_time = current_state['time']
    cur_coal = current_state['FD']
    cur_oil = current_state[Columns.OIL]
    cur_oil_elev = current_state['OIL_ELEV']
    cur_TP = current_state[Columns.TP]
    cur_MST = current_state[Columns.MST]

    # 최적선 연동
    optim_oil, optim_coal = optimal_band(cur_time, optimal_band_df)
    
    if cur_TP >= 21.0:
        optim_oil_upper = optim_oil + 20.0
        optim_coal_upper = optim_coal + 4.0
    else:
        optim_oil, optim_oil_upper = None, None
        optim_coal, optim_coal_upper = None, None

    # COAL FEEDER 제한
    if cur_coal == 0:
        coal_lb, coal_ub = 0.0, 0.0
    elif window_df.iloc[-3]['FD'] == 0.0 and cur_coal > 0.0:
        coal_lb, coal_ub = 25.0, 25.0
    else:
        coal_lb = max(cur_coal, optim_coal - 4.0) if optim_coal is not None else cur_coal
        coal_ub = min(args.max_coal, coal_lb + 4.0)

    # COAL FEEDER 후보군 생성
    if np.isclose(coal_lb, coal_ub):
        coal_candidates = [float(coal_lb)]
    else:
        step_size = 4.0
        num_steps = int(np.floor((coal_ub - coal_lb + 1e-7) / step_size)) + 1
        coal_candidates = [coal_lb + i * step_size for i in range(num_steps)]

        if optim_coal is not None:
            coal_candidates = [min(val, max(cur_coal, 25.0, optim_coal_upper)) for val in coal_candidates]
            coal_candidates = sorted(list(set(coal_candidates)))

    # OIL 제한
    ELEV_OIL_MAX = 150 if cur_oil_elev == 0 else (260 if cur_oil_elev == 1 else 370)
    oil_lb = 220.0 if cur_oil >= 220.0 else min(220.0, cur_oil)

    # 상승률 제한 산출
    max_rate_limits = np.array([args.tp_threshold * i for i in (np.array(HORIZONS) + 1)])
    step_count = (cur_time - start_time) / pd.Timedelta(seconds=30)
    remaining_steps = max(1, args.lead_time * 2 - step_count)
    target_rate = (84.0 - cur_TP) / float(remaining_steps)
    target_rate_limits = target_rate * (np.array(HORIZONS) + 1)

    # 모델 TP 예측 (현재 및 유지 상태)
    X_current = pd.DataFrame([current_state[feature_cols + ['NUM']].values], columns=feature_cols + ['NUM'])
    X_current['NUM'] = X_current['NUM'].astype(int).astype('category')
    y_pred_multi = TP_model.predict(X_current, cur_TP)[0]
    next_TP = y_pred_multi[0]

    _, maintain_state = update_state(window_df, [cur_coal, cur_oil], next_TP, args)
    X_maintain = pd.DataFrame([maintain_state[feature_cols + ['NUM']].values], columns=feature_cols + ['NUM'])
    X_maintain['NUM'] = X_maintain['NUM'].astype(int).astype('category')
    maintain_pred_multi = TP_model.predict(X_maintain, next_TP)[0]
    maintain_rate = np.array(maintain_pred_multi) - next_TP

    target_TP_rate = target_rate_limits.copy() if target_rate_limits[0] >= maintain_rate[0] else maintain_rate.copy()

    # Trip 조건 초과 여부 검증
    exceeded_mask = target_TP_rate > max_rate_limits
    if np.any(exceeded_mask):
        first_exceeded_idx = np.argmax(exceeded_mask)
        print(f"[알림] step {step_count}: HORIZON t+{HORIZONS[first_exceeded_idx]+1} 필요 상승 속도"
              f"({target_TP_rate[first_exceeded_idx]:.4f})가 허용치({max_rate_limits[first_exceeded_idx]:.4f})를 초과했습니다.")
        return None, None

    # 최적 제어값 Grid Search
    optimizer = ControlOptimizer(TP_model=TP_model, MST_model=MST_model)
    best_loss = float('inf')
    best_controls = None
    next_MST = 0.0
    found_zero_penalty = False

    coal_candidates.sort(reverse=True)
    for cand_coal in coal_candidates:
        if cand_coal > 0:
            total_coal_val = cand_coal * 0.476
            max_allowed_oil = args.max_fuel - (total_coal_val * 10.0)
            effective_oil_ub = min(ELEV_OIL_MAX, max(max_allowed_oil, 220.0), cur_oil + 10)
            effective_oil_lb = max(220.0, cur_oil + 1) if cur_oil >= 220.0 else max(220.0, oil_lb, cur_oil + 1)
        else:
            effective_oil_lb = max(oil_lb, cur_oil + 1)
            effective_oil_ub = min(ELEV_OIL_MAX, cur_oil + 10)

        if effective_oil_lb >= effective_oil_ub or np.isclose(effective_oil_lb, effective_oil_ub):
            oil_candidates = [np.ceil(effective_oil_ub)]
        else:
            step_size = 1.0
            num_steps = int(np.floor((effective_oil_ub - effective_oil_lb + 1e-7) / step_size)) + 1
            oil_candidates = [np.ceil(effective_oil_lb + i * step_size) for i in range(num_steps)]

        if optim_oil is not None:
            oil_candidates = [min(val, optim_oil_upper) for val in oil_candidates]
            oil_candidates = sorted(list(set(oil_candidates)))

        oil_candidates.sort(reverse=True)
        for cand_oil in oil_candidates:
            mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi, MST_pred = optimizer.objective(
                window_df, [cand_coal, cand_oil], feature_cols, target_TP_rate, next_TP, args
            )

            is_zero_penalty = (TP_over_penalty <= 1e-7) and (MST_over_penalty <= 1e-7)

            if found_zero_penalty and is_zero_penalty:
                if mae_loss < best_loss:
                    best_loss = mae_loss
                    best_controls = [cand_coal, cand_oil]
                    best_mae_loss, best_TP_over_penalty, best_MST_over_penalty, best_TP_pred_multi, next_MST = mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi, MST_pred

            elif not found_zero_penalty and is_zero_penalty:
                found_zero_penalty = True
                best_loss = mae_loss
                best_controls = [cand_coal, cand_oil]
                best_mae_loss, best_TP_over_penalty, best_MST_over_penalty, best_TP_pred_multi, next_MST = mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi, MST_pred

            elif not found_zero_penalty:
                best_controls = [min(cur_coal, min(coal_candidates)), min(cur_oil, min(oil_candidates))]
                mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi, MST_pred = optimizer.objective(
                    window_df, best_controls, feature_cols, target_TP_rate, next_TP, args
                )
                best_mae_loss, best_TP_over_penalty, best_MST_over_penalty, best_TP_pred_multi, next_MST = mae_loss, TP_over_penalty, MST_over_penalty, TP_pred_multi, MST_pred

    rec_coal, rec_oil = best_controls[0], best_controls[1]
    
    res_dict = {
        'rec_coal': rec_coal,
        'rec_oil': rec_oil,
        'next_TP': next_TP,
        'next_MST': next_MST,
        'next_time': cur_time + pd.Timedelta(seconds=30),
        'best_mae_loss': best_mae_loss,
        'best_TP_over_penalty': best_TP_over_penalty,
        'best_MST_over_penalty': best_MST_over_penalty,
        'best_TP_pred_multi': best_TP_pred_multi
    }

    return (rec_coal, rec_oil), res_dict


def main():
    args = parse_args()

    print("=" * 65)
    print(f"[Control Value Optimization and Recommendaton Pipeline Start] MODE: {args.mode} CaseID: {args.caseID}")
    print("=" * 65)

    # 1. 데이터 로드
    print("\n 1. Loading Dataset...")
    data_path = os.path.join(args.data_dir, 'data_preprocess.csv')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at: {data_path}")

    df = pd.read_csv(data_path, index_col=False)
    df = df[df['caseID'] == args.caseID]
    df['time'] = pd.to_datetime(df['time'])

    optimal_band_path = os.path.join(args.data_dir, 'optimal_band_data', args.mode, f'similarity_check_{args.caseID}_bands.csv')
    if not os.path.exists(optimal_band_path):
            raise FileNotFoundError(f"Optimal band not found at: {optimal_band_path}")
    
    optimal_band_df = pd.read_csv(optimal_band_path, index_col=False)
    optimal_band_df['time'] = pd.to_datetime(optimal_band_df['time'])

    # 2. 모델 로드
    print("\n 2. Loading Models...")
    TP_model_path = os.path.join(args.model_dir, f'{args.mode}_TP_lgbm_multi.pkl')
    MST_model_path = os.path.join(args.model_dir, f'{args.mode}_MST_ridge_single.pkl')

    TP_model = LGBMMultiOutputModel.load(TP_model_path)
    MST_model = RidgeRegressionModel.load(MST_model_path)
    models = {'TP_model': TP_model, 'MST_model': MST_model}

    print("\n 3. Simulation...")
    WINDOW_SIZE = args.lag_steps + 1
    start_time = pd.to_datetime(df['time'].iloc[0])
    
    if args.action_type == 'real':
        # window_df = df.iloc[-WINDOW_SIZE:].copy()
        window_df = df.iloc[:WINDOW_SIZE].copy()
        best_controls, res_dict = predict_single_step(window_df, optimal_band_df, start_time, models, args)
        
        if best_controls is not None:
            print(f"\n[추천 결과] Coal (FD): {res_dict['rec_coal']:.2f}, Oil: {res_dict['rec_oil']:.2f}")
            return res_dict['rec_coal'], res_dict['rec_oil'], res_dict['best_mae_loss'], res_dict['best_TP_over_penalty'], res_dict['best_MST_over_penalty'], res_dict['best_TP_pred_multi']
        else:
            print("\n[알림] 제어한계 초과로 추천 실패")
            return None
    else:
        sim_df = df.copy()
        simulated_results = []
        i = 0

        print("\n[시뮬레이션 루프 시작]")
        pbar = tqdm(total=len(sim_df) - WINDOW_SIZE)

        while i + WINDOW_SIZE < len(sim_df):
            window_df = sim_df.iloc[i : i + WINDOW_SIZE].copy()
            cur_TP = window_df.iloc[-1][Columns.TP]

            if cur_TP >= args.target_tp:
                print(f"\n[성공] 목표 TP({args.target_tp}) 달성으로 시뮬레이션 종료.")
                break

            # 1스텝 공통 예측 헬퍼 호출
            best_controls, res_dict = predict_single_step(window_df, optimal_band_df, start_time, models, args)

            if best_controls is None:
                print("\n[알림] 제어 불가 한계 도달로 시뮬레이션 중단")
                break

            rec_coal, rec_oil = res_dict['rec_coal'], res_dict['rec_oil']
            next_TP, next_MST = res_dict['next_TP'], res_dict['next_MST']
            # rec_TP_pred_multi = res_dict['best_TP_pred_multi']

            # 다음 행 데이터 덮어쓰기 (피드백 반영)
            next_row_idx = sim_df.index[i + WINDOW_SIZE]
            sim_df.loc[next_row_idx, 'FD'] = rec_coal
            sim_df.loc[next_row_idx, Columns.OIL] = rec_oil
            sim_df.loc[next_row_idx, Columns.TP] = next_TP
            sim_df.loc[next_row_idx, Columns.MST] = next_MST

            simulated_results.append({
                'time': res_dict['next_time'],
                'FD': rec_coal,
                Columns.OIL: rec_oil,
                Columns.TP: next_TP,
                Columns.MST: next_MST,
                # f'{Columns.TP}_180': rec_TP_pred_multi[-1],
            })

            i += 1
            pbar.update(1)

        pbar.close()

        simulated_df = pd.DataFrame(simulated_results)
        simulated_df.to_csv(f'{args.output_dir}/{args.caseID}_rec_sim.csv', index=False)
        
        figure_df, _ = derived_variable(df)
        df_with_optimal_band = pd.merge(
            figure_df,
            optimal_band_df[['pressure_matched_raw', 'oil_matched_raw', 'FD_matched_raw', 'time']],
            on=['time'],
            how='left',
        )

        index_84 = df_with_optimal_band[df_with_optimal_band[Columns.TP] >= 84].index[0]

        plot_and_save_all_trajectories_compare_optimal_line(df_with_optimal_band.loc[:index_84], 
                                                            simulated_df, 
                                                            case_id=args.caseID,
                                                            save_dir=args.figure_dir,
                                                            TP=Columns.TP,
                                                            OIL=Columns.OIL,
                                                            MST=Columns.MST,
                                                            lead_time=args.lead_time)
        
    return None
    
if __name__ == '__main__':
    main()

