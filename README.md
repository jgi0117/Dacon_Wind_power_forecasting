# DACON 풍력 발전량 예측 — Version 3

기상 예보 기반 풍력 발전량 예측 파이프라인입니다. Version 2 실험에서 가장 높은 validation 및 DACON 점수를 기록한 RealMLP을 최종 baseline으로 선정했습니다. Version 3는 RealMLP만 사용해 학습 길이와 FICR-aware loss를 집중적으로 실험합니다.

- [Version 1 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v1.0.0): 5개 baseline 비교
- [Version 2 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v2.0.0): FICR-aware 모델 비교와 RealMLP 선정

## 1. 프로젝트 개요

기상 예보와 시간 정보를 이용해 2025년의 시간별 풍력 발전량 3개 그룹(kpx_group_1~3)을 예측하는 회귀 프로젝트입니다. Version 3에서 지원하는 모델은 RealMLP 하나뿐입니다.

평가식은 다음과 같습니다.

    score = 0.5 × (1-NMAE) + 0.5 × FICR

## 2. 데이터 전처리 방식

원본 데이터는 Git에 포함하지 않습니다. 전처리 결과는 2022-01-01 01:00부터 2025-01-01 00:00까지의 학습 데이터 26,304행과, 2025-01-01 01:00부터 2026-01-01 00:00까지의 테스트 데이터 8,760행으로 구성됩니다.

현재 hybrid 특성 세트는 총 655개입니다.

- 시간: 월·일·시각의 주기형(sin/cos) 특성
- 기상: LDAPS 16개 격자와 GFS 9개 격자의 예보 변수
- 풍속·풍향: 벡터 성분, 풍속의 제곱·세제곱, 고도 간 shear와 비율
- 공간 요약: 격자별 평균·표준편차·최솟값·최댓값과 선택 격자값
- 결측 처리: 동일한 예보 배치와 격자 안에서만 보간하고, 남은 값은 학습 구간 격자 중앙값으로 채우며 결측 표시 특성을 유지
- 타깃: 발전량을 설비용량으로 나눈 capacity factor로 학습하고 예측 후 원 단위로 복원

미래 정보 누수를 막기 위해 예보 생성 시각과 사용 가능 시각을 엄격하게 제한합니다.

    .\.venv313\Scripts\python.exe preprocessing.py --data-dir data --output-dir artifacts --mode hybrid

## 3. Version 3 학습 전략

### 시간 순 검증

1. 2023-12-31 14:00 이전 데이터를 학습에 사용합니다.
2. 경계에서 11시간을 purge합니다.
3. 2024년 1~3월을 epoch 선택용 validation으로 사용합니다.
4. 2024년 7~12월을 최종 비교 구간으로 유지합니다.
5. 선택된 epoch로 각 타깃의 사용 가능한 전체 과거 데이터를 다시 학습해 테스트를 예측합니다.

세 타깃은 독립적으로 학습합니다. 최대 200 epoch를 실행하고 FICR-aware validation loss가 가장 낮은 epoch를 최종 전체 데이터 재학습 길이로 사용합니다.

### FICR-aware loss

원래 FICR의 6%와 8% 오차 경계를 sigmoid로 근사한 soft-FICR을 사용합니다.

    loss = 0.25 × smooth-MAE + 0.75 × (1-soft-FICR)

기본 FICR 비중은 0.75, sigmoid temperature는 0.01입니다. 각 epoch의 train loss, validation loss, competition score를 모두 기록합니다.

## 4. 실행 방법

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs --max-epochs 200

FICR 설정을 변경하는 예:

    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs --max-epochs 200 --ficr-weight 0.8 --ficr-temperature 0.008

모델 산출물은 model_outputs/v3/runs/realmlp에 저장되고 통합 결과는 reports/v3에 생성됩니다.

- results.csv: validation 및 DACON 지표
- group_metrics.csv: 타깃별 지표
- monthly_metrics.csv: 월별 지표
- training_summary.csv: 최적 epoch와 학습 시간
- training_history.csv: epoch별 train/validation loss와 score
- figures/training_curves.png: RealMLP train/validation loss 변화

## 5. 선정 기준점

Version 2의 20-epoch RealMLP 결과는 다음과 같습니다.

| Validation score | DACON score | DACON 1-NMAE | DACON FICR |
|---:|---:|---:|---:|
| 0.651523 | 0.630536 | 0.856565 | 0.404507 |

Version 1 RealMLP 대비 DACON score는 0.006237, FICR은 0.025914 상승했습니다. Version 3에서는 200-epoch history를 이용해 추가 학습이 계속 유효한지, 가장 좋은 epoch가 어디인지 확인합니다.

![RealMLP train/validation loss](reports/v3/figures/training_curves.png)
