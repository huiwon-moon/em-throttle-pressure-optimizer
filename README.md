# Throttle Pressure Optimizer Pipeline

실시간 공정 데이터를 바탕으로 30초 단위의 미래 주증기 압력(t+1~t+6)을 예측하고, 제어 조건을 반영하여 최적의 제어값을 산출·추천하는 파이프라인 시스템입니다.

---

## Key Features

- **다중 타겟/작업 지원**:
  - `THROTTLE PRESSURE`: 미래 $t+1 \sim t+6$ 스텝 Multi-horizon 예측 (단위: 30s)
  - `MAIN STEAM TEMPERATURE`: 현재 시점 $t$ 추정
- **제어값 최적화 지원**:
  - `RECOMMEND CONTROL`: 목표 시간에 적합한 최적의 제어값 추천 알고리즘 (단위: 30s)
- **Feature Engineering 파이프라인**:
  - `caseID` 단위 시계열 그룹핑 기반 Diff, Rolling(Sum) 자동 생성
- **통합 CLI & Orchestration**:
  - `main.py`를 통합 제어 타워로 사용하여 `train`, `inference` 모드 일괄 제어

---

## Directory Structure

```text
├── data/                   # (비공개) 시계열 CSV 데이터
├── models/                 # (비공개) 학습 완료된 기동 유형별 TP, MST 모델 (.pkl)
├── config/
│   └── columns.py          # (비공개) feature & target 태그 정보
├── docs/                   # (추후 공개) 기술 설명서
├── fig/                    # (비공개) TP, MST, Recommend Simulation 평가 시각화
├── results/                # (비공개) 추론 결과 CSV
├── src/
│   └── data_split.py       # HOT/WARM 기동 유형 분리 & 훈련/테스트 데이터 분할
│   └── evaluation.py       # 성능 평가 코드
│   └── features.py         # feature engineering
│   └── model.py            # 모델 구조
│   └── visualization.py    # 시각화 코드
├── main.py                 # 예측 모델파이프라인 실행 진입점 (Entry Point)
├── recommend_control.py    # 추천 시뮬레이션 실행 코드
└── requirements.txt        # 의존성 라이브러리 목록
```
---

## Usage
### 1. 가상환경 생성 및 라이브러리 설치
```bash
python -m venv venv
source venv\Scripts\activate
pip install -r requirements.txt
```

### 2. 파이프라인 실행
```bash
python main.py --mode WARM --target TP --model_name lgbm --task_type multi
python recommend_control.py --mode HOT --caseID 116 --lead_time 66 --action_type simulation
```

---

## Validation & Saving
- **추론 메트릭**
    - `results/` 폴더 내에 Horizon별로 MAE, MSE, RMSE 메트릭이 포함된 CSV 파일 저장
    - `results/` 폴더 내에 recommend_control의 simulation 결과 CSV 파일로 저장
- **시각화 결과**
    - `fig/TP/` 폴더 내에 실제 Throttle Pressure vs 모델 예측값 시계열 그래프 생성 (.png)
    - `fig/MST/` 폴더 내에 실제 Main Steam Temperature vs 모델 추정값 시계열 그래프 생성 (.png)
    - `fig/recommend_sim/` 폴더 내에 추천 제어 알고리즘 실행 결과 그래프 생성 (.png)

---

## CLI Arguments

| 옵션 (Argument) | 타입 (Type) | 기본값 (Default) | 선택 가능 값 (Choices) | 설명 (Description) |
| :--- | :--- | :--- | :--- | :--- |
| `--n_estimators` | `int` | `3000` | 양의 정수 | 최대 트리 개수 |
| `--learning_rate` | `float` | `0.01` | 양의 실수 | 모델 학습률 |
| `--random_state` | `int` | `42` | 정수 | 재현성을 위한 난수 고정 시드값 |
| `--model_name` | `str` | `lgbm` | `lgbm`, `ridge` | 모델 종류 |
| `--task_type` | `str` | `multi` | `multi`, `single` | 모델 타입 (다중 예측 vs 단일 추정) |
