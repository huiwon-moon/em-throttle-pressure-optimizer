import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

def plot_restored_predictions(
    test_df: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    case_id: str = None,
    HORIZONS: list = None,
    save_dir: str = None,
):
    test_df = test_df.reset_index(drop=True)

    if case_id is None or case_id == 'all':
        target_case_ids = test_df['caseID'].unique()
    else:
        target_case_ids = [case_id]

    y_test_arr = np.asarray(y_test)
    y_pred_arr = np.asarray(y_pred)

    for tc in target_case_ids:
        mask = (test_df['caseID'] == tc).values
        chunk_df = test_df[mask].copy().reset_index(drop=True)

        if len(chunk_df) == 0:
            continue

        num_id = str(chunk_df['NUM'].iloc[0])
        time_series = chunk_df['time']

        chunk_y_true = y_test_arr[mask]
        chunk_y_pred = y_pred_arr[mask]

        current_save_dir = save_dir

        # -------------------------------------------------------------
        # Multi-Horizon 예측 (TP)
        # -------------------------------------------------------------
        if chunk_y_pred.ndim > 1 and chunk_y_pred.shape[1] > 1:

            target_name = 'TP'

            n_horizons = chunk_y_pred.shape[1]
            fig, axes = plt.subplots(nrows=n_horizons, ncols=1, figsize=(15, 3.5 * n_horizons), sharex=True)
            if n_horizons == 1:
                axes = [axes]

            for h_idx in range(n_horizons):
                ax = axes[h_idx]
                h_step = HORIZONS[h_idx]

                y_true_plot = chunk_y_true[:, h_idx]
                y_pred_plot = chunk_y_pred[:, h_idx]

                mae = np.mean(np.abs(y_true_plot - y_pred_plot))

                ax.plot(time_series, y_true_plot, 'k-', linewidth=2, label=f'Actual (t+{h_step})')
                ax.plot(time_series, y_pred_plot, 'r--', linewidth=1.8, label=f'Pred (t+{h_step})')

                ax.set_title(f"Horizon +{h_step} Step Ahead Prediction | MAE: {mae:.4f}",
                             fontsize=12, fontweight='bold', loc='left')
                ax.set_ylabel(f'Throttle Pressure (TP)', fontsize=11)
                ax.legend(loc='upper left', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=30)

            axes[-1].set_xlabel('Time', fontsize=12)
            axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))

            plt.suptitle(f"[{num_id}호기] CaseID: {tc} | Multi-Horizon (+1 ~ +{n_horizons}) Prediction",
                         fontsize=14, fontweight='bold', y=1.01)

        # -------------------------------------------------------------
        # Single-Step 예측 (MST)
        # -------------------------------------------------------------
        else:
            target_name = 'MST'

            y_true_plot = chunk_y_true.flatten()
            y_pred_plot = chunk_y_pred.flatten()

            orig_mae = np.mean(np.abs(y_true_plot - y_pred_plot))

            fig, ax = plt.subplots(figsize=(15, 4.5))

            ax.plot(time_series, y_true_plot, 'k-', linewidth=2, label='Actual (Real Unit)', alpha=0.85)
            ax.plot(time_series, y_pred_plot, 'r--', linewidth=1.8, label='Ridge Pred (Restored)')

            ax.set_title(f"[{num_id}호기] CaseID: {tc} | Original {target_name} Unit Prediction (MAE: {orig_mae:.4f})",
                         fontsize=13, fontweight='bold')
            ax.set_xlabel('Time', fontsize=11)
            ax.set_ylabel(f'MST (Real Unit)', fontsize=11)
            ax.legend(loc='upper left', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.tick_params(axis='x', rotation=30)

        plt.tight_layout()

        os.makedirs(current_save_dir, exist_ok=True)
        fig_path = os.path.join(current_save_dir, f'case_{tc}_{target_name}.png')
        fig.savefig(fig_path, dpi=300, bbox_inches='tight')

        plt.close(fig)

    print(f"총 {len(target_case_ids)}개 CaseID에 대한 그래프 생성 및 저장 완료.")


def plot_and_save_all_trajectories_compare_optimal_line(
    df,
    simulated_df,
    case_id=None,
    save_dir=None,
    TP=None,
    OIL=None,
    MST=None,
    lead_time=None,
    ):
    
    start_time = pd.to_datetime(df["time"].iloc[0])
    get_mins = (
        lambda d: (pd.to_datetime(d["time"]) - start_time).dt.total_seconds()
        / 60.0
    )
    num_id = df["NUM"].iloc[0]

    mins_df = get_mins(df)
    mins_rec = get_mins(simulated_df)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    sim_end_min = mins_rec.iloc[-1]
    actual_end_min = mins_df.iloc[-1]
    time_diff = abs(actual_end_min - sim_end_min)

    # =========================================================================
    # 1. TP Trajectory (기존 유지)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))

    tp_opt_col = "pressure_matched_raw"
    tp_offset = 7.0
    opt_end_min = None

    if tp_opt_col in df.columns:
        tp_opt_line = df[tp_opt_col]
        tp_lower_bound = tp_opt_line - tp_offset
        tp_upper_bound = tp_opt_line + tp_offset

        valid_opt = tp_opt_line.dropna()
        if not valid_opt.empty:
            last_idx = valid_opt.index[-1]
            opt_end_min = mins_df.loc[last_idx]

        ax.fill_between(
            mins_df,
            tp_lower_bound,
            tp_upper_bound,
            color="green",
            alpha=0.15,
            label=f"Target TP Band (±{tp_offset:.0f})",
        )
        ax.plot(
            mins_df,
            tp_lower_bound,
            color="green",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )
        ax.plot(
            mins_df,
            tp_upper_bound,
            color="green",
            linestyle=":",
            linewidth=1,
            alpha=0.5,
        )

        ax.plot(
            mins_df,
            tp_opt_line,
            label=f"Optimal TP ({tp_opt_col})",
            color="green",
            linestyle="-.",
            linewidth=1.8,
            alpha=0.85,
        )

    ax.plot(
        mins_df,
        df[TP],
        label="Actual TP",
        color="black",
        linestyle="--",
        linewidth=2,
        alpha=0.7,
    )
    ax.plot(
        mins_rec,
        simulated_df[TP],
        label="Simulated TP",
        color="red",
        linewidth=2.5,
    )
    ax.axhline(y=84, color="gray", linestyle="-.", alpha=0.5)


    if simulated_df.iloc[[-1]][TP].values[0] >= 84:
        
        ax.axvline(
            x=sim_end_min,
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Simulation Time ({sim_end_min:.1f}분)",
        )
        ax.axvline(
            x=actual_end_min,
            color="blue",
            linestyle=":",
            linewidth=2,
            label=f"Actual Time ({actual_end_min:.1f}분)",
        )

        if opt_end_min is not None:
            ax.axvline(
                x=opt_end_min,
                color="green",
                linestyle="--",
                linewidth=2,
                label=f"Optimal End ({opt_end_min:.1f}분)",
            )

        y_min, y_max = ax.get_ylim()

        arrow_y1 = y_min + (y_max - y_min) * 0.93
        ax.annotate(
            "",
            xy=(sim_end_min, arrow_y1),
            xytext=(actual_end_min, arrow_y1),
            arrowprops=dict(
                arrowstyle="<->", color="purple", lw=2, linestyle="--"
            ),
        )
        ax.text(
            (sim_end_min + actual_end_min) / 2,
            arrow_y1,
            f" 실제 대비 단축: {time_diff:.1f}분 ",
            ha="center",
            va="center",
            fontsize=10,
            fontweight="bold",
            color="purple",
            bbox=dict(
                boxstyle="round,pad=0.2",
                facecolor="white",
                edgecolor="purple",
                alpha=0.9,
            ),
        )

        if opt_end_min is not None:
            opt_diff = opt_end_min - sim_end_min
            arrow_y2 = y_min + (y_max - y_min) * 0.83

            ax.annotate(
                "",
                xy=(sim_end_min, arrow_y2),
                xytext=(opt_end_min, arrow_y2),
                arrowprops=dict(
                    arrowstyle="<->", color="darkgreen", lw=2, linestyle="--"
                ),
            )
            ax.text(
                (sim_end_min + opt_end_min) / 2,
                arrow_y2,
                f" 최적선 대비 단축: {opt_diff:.1f}분 ",
                ha="center",
                va="center",
                fontsize=10,
                fontweight="bold",
                color="darkgreen",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="darkgreen",
                    alpha=0.9,
                ),
            )

    ax.set_title(
        f"[{num_id}호기] CaseID: {case_id} - TP Trajectory Comparison: Actual vs Simulations",
        fontsize=14,
        fontweight="bold",
    )
    ax.set_xlabel("Elapsed Time (Minutes)", fontsize=12)
    ax.set_ylabel("TP Value", fontsize=12)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'case_{case_id}_TP.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    plt.close()

    # =========================================================================
    # 2. Control Trajectory (FD, OIL) - 총합(적분값) 계산 및 시각화 추가
    # =========================================================================
    fig, axes = plt.subplots(2, 1, figsize=(12, 11), sharex=True)
    controls = ["FD", OIL]

    control_config = {
        "FD": {
            "opt_col": "FD_matched_raw",
            "lower_offset": -4.0,
            "upper_offset": +4.0,
        },
        OIL: {
            "opt_col": "oil_matched_raw",
            "lower_offset": -20.0,
            "upper_offset": +20.0,
        },
    }

    for i, col in enumerate(controls):

        if col == 'FD':
            label_col_name = 'COAL'
        else:
            label_col_name = 'OIL'

        ax = axes[i]
        cfg = control_config[col]
        opt_col = cfg["opt_col"]

        # ---------------------------------------------------------------------
        # 연료량 총합
        # ---------------------------------------------------------------------
        # 1) Actual 총합
        actual_sum = sum(df[col])

        # 2) Simulation 총합
        sim_sum = sum(simulated_df[col])

        # 3) Optimal 총합
        opt_end_min = None
        opt_sum = None
        if opt_col in df.columns:
            opt_line = df[opt_col]
            valid_opt = opt_line.dropna()
            if not valid_opt.empty:
                last_idx = valid_opt.index[-1]
                opt_end_min = mins_df.loc[last_idx]
                opt_sum = sum(valid_opt)

            lower_bound = opt_line + cfg["lower_offset"]
            upper_bound = opt_line + cfg["upper_offset"]

            # 오프셋 밴드
            ax.fill_between(
                mins_df,
                lower_bound,
                upper_bound,
                color="green",
                alpha=0.15,
                label=f"{label_col_name} Optimal Band ({cfg['lower_offset']:+.0f} / {cfg['upper_offset']:+.0f})",
            )
            ax.plot(
                mins_df,
                lower_bound,
                color="green",
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )
            ax.plot(
                mins_df,
                upper_bound,
                color="green",
                linestyle=":",
                linewidth=1,
                alpha=0.5,
            )

            # 최적선 플롯 (범례에 총합 표시)
            opt_label = (
                f"Optimal {label_col_name} (총합: {opt_sum:.1f})"
                if opt_sum
                else f"Optimal {label_col_name}"
            )
            ax.plot(
                mins_df,
                opt_line,
                label=opt_label,
                color="green",
                linestyle="-.",
                linewidth=1.8,
                alpha=0.85,
            )

        # 실제값 및 시뮬레이션 값 플롯 (범례에 총합 표시)
        ax.plot(
            mins_df,
            df[col],
            label=f"Actual {label_col_name} (총합: {actual_sum:.1f})",
            color="black",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
        )
        ax.plot(
            mins_rec,
            simulated_df[col],
            label=f"Simulated {label_col_name} (총합: {sim_sum:.1f})",
            color="red",
            linewidth=2.5,
        )


        if simulated_df.iloc[[-1]][TP].values[0] >= 84:
            
            # 수직선에 총합 정보 함께 표기
            ax.axvline(
                x=sim_end_min,
                color="orange",
                linestyle="-",
                linewidth=2,
                label=f"Simulation ({sim_end_min:.1f}분)",
            )
            ax.axvline(
                x=actual_end_min,
                color="blue",
                linestyle=":",
                linewidth=2,
                label=f"Actual ({actual_end_min:.1f}분)",
            )

            if opt_end_min is not None:
                ax.axvline(
                    x=opt_end_min,
                    color="green",
                    linestyle="--",
                    linewidth=2,
                    label=f"Optimal ({opt_end_min:.1f}분)",
                )

            # 시간 단축 및 차이 라벨 표시
            y_min, y_max = ax.get_ylim()

            arrow_y1 = y_min + (y_max - y_min) * 0.93
            ax.annotate(
                "",
                xy=(sim_end_min, arrow_y1),
                xytext=(actual_end_min, arrow_y1),
                arrowprops=dict(
                    arrowstyle="<->", color="purple", lw=2, linestyle="--"
                ),
            )
            ax.text(
                (sim_end_min + actual_end_min) / 2,
                arrow_y1,
                f" 실제 대비 단축: {time_diff:.1f}분 ",
                ha="center",
                va="center",
                fontsize=9,
                fontweight="bold",
                color="purple",
                bbox=dict(
                    boxstyle="round,pad=0.2",
                    facecolor="white",
                    edgecolor="purple",
                    alpha=0.9,
                ),
            )
            
            if opt_end_min is not None:
                opt_diff = opt_end_min - sim_end_min
                arrow_y2 = y_min + (y_max - y_min) * 0.83

                ax.annotate(
                    "",
                    xy=(sim_end_min, arrow_y2),
                    xytext=(opt_end_min, arrow_y2),
                    arrowprops=dict(
                        arrowstyle="<->", color="darkgreen", lw=2, linestyle="--"
                    ),
                )
                ax.text(
                    (sim_end_min + opt_end_min) / 2,
                    arrow_y2,
                    f" 최적선 대비 단축: {opt_diff:.1f}분 ",
                    ha="center",
                    va="center",
                    fontsize=9,
                    fontweight="bold",
                    color="darkgreen",
                    bbox=dict(
                        boxstyle="round,pad=0.2",
                        facecolor="white",
                        edgecolor="darkgreen",
                        alpha=0.9,
                    ),
                )

        ax.set_ylabel(f"{label_col_name} Value", fontsize=12)
        ax.set_title(
            f"[{num_id}호기] CaseID: {case_id} - {label_col_name} Control Trajectory Comparison",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3)
        # 범례를 그래프 우측 상단 밖이나 내부 보기 좋은 위치에 배치
        ax.legend(loc="upper left", fontsize=9, framealpha=0.8)

    plt.xlabel("Elapsed Time (Minutes)", fontsize=12)
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    fig_path = os.path.join(save_dir, f'case_{case_id}_control.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

    plt.close()

    print("[알림] 모든 시각화 그래프 저장이 완료되었습니다.")
