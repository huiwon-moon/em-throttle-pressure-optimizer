import os
import sys
import random
import argparse
import joblib
import numpy as np
import pandas as pd
from typing import List

from config.columns import Columns
from src.features import create_target_deltas, derived_variable, add_features
from src.model import LGBMMultiOutputModel, RidgeRegressionModel
from src.evaluation import prepare_true_values, performance_evaluate, save_performance_matrix
from src.visualization import plot_restored_predictions

def parse_args():
    parser = argparse.ArgumentParser(description="Throttle Pressure & Main Steam Temperature Pipeline")

    # 모델 파라미터 대상
    parser.add_argument('--mode', type=str, default='HOT', choices=['HOT', 'WARM'], help='기동 상태 선택')
    parser.add_argument('--target', type=str, default='TP', choices=['TP', 'MST'], help='예측 타겟 선택')

    # 데이터 경로 지정
    parser.add_argument('--data_dir',type=str, default='data', help='데이터 디렉토리 경로')
    parser.add_argument('--model_dir', type=str, default='models', help='학습된 모델 저장 디렉토리 경로')

    # LightGBM 하이퍼파라미터
    parser.add_argument('--n_estimators', type=int, default=3000, help='최대 트리 개수')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='학습률')
    parser.add_argument('--max_depth', type=int, default=5, help='트리 최대 깊이')
    parser.add_argument('--num_leaves', type=int, default=31, 
                        help='트리 하나당 최대 잎(Leaf) 노드 개수 (복잡도 제어 및 과적합 방지)')
    parser.add_argument('--min_child_samples', type=int, default=30, 
                        help='한 잎(Leaf) 노드가 가져야 하는 최소 데이터 개수 (노이즈 학습 방지)')
    parser.add_argument('--subsample', type=float, default=0.8, 
                        help='개별 트리를 학습할 때 사용할 데이터 샘플링 비율 (0.0~1.0, 과적합 방지 및 학습 속도 향상)')
    parser.add_argument('--colsample_bytree', type=float, default=0.8, 
                        help='개별 트리를 학습할 때 사용할 피쳐 샘플링 비율')
    parser.add_argument('--reg_alpha', type=float, default=0.1, help='L1(필요없는 가중치 0)')
    parser.add_argument('--reg_lambda', type=float, default=0.1, help='L2(가중치가 너무 커지는 것 방지)')
    parser.add_argument('--random_state', type=int, default=42, help='seed')

    # Ridge 하이퍼파리미터
    parser.add_argument('--alpha', type=float, default=1.0, help=
                        'Ridge 모델의 L2 규제 강도 (alpha값이 클수록 가중치 규제 강화)')

    # 모델 정보
    parser.add_argument('--model_name', type=str, default='lgbm', choices=['lgbm', 'ridge'], help='모델 이름')
    parser.add_argument('--task_type', type=str, default='multi', choices=['multi', 'single'], help='단일 vs 다중')
    parser.add_argument('--output_dir', type=str, default='results', help='성능 평가 저장 경로')
    parser.add_argument('--figure_dir', type=str, default='fig', help='그래프 저장 경로')
    
    args, _ = parser.parse_known_args()

    return args

def prepare_features_and_targets(
        df: pd.DataFrame,
        feature_cols: List[str],
        target_cols: List[str],
):
    X = df[feature_cols + ['NUM']]
    y = df[target_cols]

    return X, y

def set_seed(seed: int=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def main():
    args = parse_args()
    set_seed(args.random_state)

    print("=" * 65)
    print(f"[Train Pipeline Start] MODE: {args.mode} | Target: {args.target}")
    print("=" * 65)

    # 1. 데이터 로드
    print("\n 1. Loading Pre-split CSV Datasets...")
    train_path = os.path.join(args.data_dir, f'{args.mode}_{args.target}_train_df.csv')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Train dataset not found at: {train_path}")

    train_df = pd.read_csv(train_path, index_col=False)

    # TP: 미래 multi-horizon / MST: 현재 추정
    if args.target == 'TP':
        target_col = Columns.TP
        horizons = [1, 2, 3, 4, 5, 6]
        train_df, target_cols_list = create_target_deltas(train_df, target_col, horizons=horizons)
    else:
        target_col = Columns.MST
        train_df, target_cols_list = create_target_deltas(train_df, target_col, horizons=None)

    # Feature Engineering
    train_df, orig_derived_feature_cols = derived_variable(train_df)
    train_df, feature_cols = add_features(train_df, orig_derived_feature_cols, lag_steps=20)

    # X, y 분리
    X_train, y_train = prepare_features_and_targets(train_df, feature_cols, target_cols_list)

    print(f"   - Number of Input Features: {X_train.shape[1]}")

    # 2. 모델 학습
    if args.target == 'TP':
        print("\n 2. Training LightGBM Model...")
        model = LGBMMultiOutputModel(
            args.n_estimators,
            args.learning_rate,
            args.max_depth,
            args.num_leaves,
            args.min_child_samples,
            args.subsample,
            args.colsample_bytree,
            args.reg_alpha,
            args.reg_lambda,
            args.random_state,
        )
    else:
        print("\n 2. Training Ridge Regression...")
        model = RidgeRegressionModel(
            feature_cols=feature_cols,
            cat_cols=['NUM'],
            alpha=args.alpha
        )

    model.fit(X_train, y_train)

    # 3. 모델 저장
    print("\n 3. Saving Trained Model")
    model.save(
        model_name=args.model_name,
        mode=args.mode,
        target=args.target,
        task_type=args.task_type,
        save_dir=args.model_dir,
    )

    print("=" * 65)
    print(f"[Inference Pipeline Start] MODE: {args.mode} | Target: {args.target}")
    print("=" * 65)

    # 4. Test 데이터 전처리 및 파이프라인 적용
    print("\n 4. Loading & Preprocessing Test Dataset...")
    test_path = os.path.join(args.data_dir, f'{args.mode}_{args.target}_test_df.csv')
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test dataset not found at: {test_path}")

    test_df = pd.read_csv(test_path, index_col=False)

    # TP: 미래 multi-horizon / MST: 현재 추정
    if args.target == 'TP':
        target_col = Columns.TP
        horizons = [1, 2, 3, 4, 5, 6]
        test_df, target_cols_list = create_target_deltas(test_df, target_col, horizons=horizons)
    else:
        target_col = Columns.MST
        horizons = None
        test_df, target_cols_list = create_target_deltas(test_df, target_col, horizons=horizons)

    # Feature Engineering
    test_df, orig_derived_feature_cols = derived_variable(test_df)
    test_df, feature_cols = add_features(test_df, orig_derived_feature_cols, lag_steps=20)

    # X, y 분리
    X_test, y_test = prepare_features_and_targets(test_df, feature_cols, target_cols_list)

    print(f"   - Number of Input Features: {X_test.shape[1]}")

    # 5. 추론 실행
    print("\n 5. Running Inference...")
    model_path = os.path.join(args.model_dir, f'{args.mode}_{args.target}_{args.model_name}_{args.task_type}.pkl')
    if args.target == 'TP':
        loaded_model = LGBMMultiOutputModel.load(model_path)
        y_pred = loaded_model.predict(X_test, base_target_series=test_df[[target_col]].values)
    else:
        loaded_model = RidgeRegressionModel.load(model_path)
        y_pred = loaded_model.predict(X_test, base_target_series=(test_df[[target_col]].values - test_df[target_cols_list].values))


    # 6. 성능 평가 및 결과 저장
    file_name = f'{args.mode}_{args.target}_performance_matrix'

    print("\n 6. Evaluating Predictions...")
    y_true = prepare_true_values(test_df, target_col, target_cols_list)

    summary_df = performance_evaluate(test_df, y_true, y_pred, model_name=args.model_name, target_cols_list=target_cols_list)
    save_performance_matrix(summary_df, file_name, args.output_dir)
    
    # 7. 시각화 및 예측 결과 저장
    print("\n 7. Saving Results & Generating Plot...")
    plot_restored_predictions(
        test_df,
        y_true,
        y_pred,
        case_id=None,
        HORIZONS=horizons,
        save_dir=os.path.join(args.figure_dir, args.target, args.mode)
    )

    print("\n" + "=" * 65)
    print("[Inference Pipeline Finished Successfully]")
    print("=" * 65)

if __name__ == '__main__':
    main()