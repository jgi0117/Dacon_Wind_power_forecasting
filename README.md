# DACON 풍력 발전량 예측 — Version 3

기상 예보 기반 풍력 발전량 예측 파이프라인입니다. Version 2 실험에서 가장 높은 validation 및 DACON 점수를 기록한 RealMLP을 최종 baseline으로 선정했습니다. Version 3는 학습 길이를 200 epoch로 늘린 RealMLP에서 learning rate에 따른 수렴과 일반화 성능 차이를 비교합니다.

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

세 타깃은 독립적으로 학습합니다. 200 epoch를 모두 실행하고 FICR-aware validation loss가 가장 낮은 epoch를 최종 전체 데이터 재학습 길이로 사용합니다. RealMLP-TD 회귀의 tuned defaults를 기반으로 `lr_sched=coslog4`, dropout 0.15, Adam을 고정하고 LR 0.2, 0.02, 0.002를 비교합니다. Classical early stopping은 사용하지 않으며 validation은 최적 epoch 선택에만 사용합니다.

### FICR-aware loss

원래 FICR의 6%와 8% 오차 경계를 sigmoid로 근사한 soft-FICR을 사용합니다.

    loss = 0.25 × smooth-MAE + 0.75 × (1-soft-FICR)

기본 FICR 비중은 0.75, sigmoid temperature는 0.01입니다. 각 epoch의 train loss, validation loss, competition score를 모두 기록합니다.

## 4. 실행 방법

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs @('--max-epochs','200','--learning-rate','0.02')

FICR 설정을 변경하는 예:

    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs @('--max-epochs','200','--learning-rate','0.02','--ficr-weight','0.8','--ficr-temperature','0.008')

실험 디렉터리의 v3/v4는 모델 버전이 아니라 실행 결과를 분리하기 위한 저장 경로입니다. 200-epoch LR 비교 전체를 Version 3 실험으로 취급합니다.

- results.csv: validation 및 DACON 지표
- group_metrics.csv: 타깃별 지표
- monthly_metrics.csv: 월별 지표
- training_summary.csv: 최적 epoch와 학습 시간
- training_history.csv: epoch별 train/validation loss와 score
- figures/training_curves.png: RealMLP train/validation loss 변화

## 5. Version 3 LR 비교 결과

Version 3는 RealMLP을 200 epoch까지 학습하면서 LR만 변경하는 실험입니다. 아래 DACON 점수는 제출 화면의 값을 기록했습니다.

| LR | Validation score | DACON score | DACON 1-NMAE | DACON FICR | 상태 |
|---:|---:|---:|---:|---:|:---|
| 0.2 | 0.653255 | 0.631055 | 0.861265 | 0.400845 | 완료 |
| 0.02 | 0.650715 | 0.625799 | 0.861005 | 0.390592 | 완료 |
| 0.002 | - | - | - | - | 진행 중 |

LR 0.02는 LR 0.2보다 DACON score가 0.005256 낮았습니다. 1-NMAE 차이는 0.000260에 불과하지만 FICR은 0.010253 낮아, 현재 점수 하락은 주로 임계 오차 구간을 반영하는 FICR에서 발생했습니다.

### Epoch별 train/validation loss

| LR | 타깃 | 최저 validation loss (epoch) | 마지막 validation loss | 마지막 train loss |
|---:|:---|---:|---:|---:|
| 0.2 | kpx_group_1 | 0.436019 (39) | 0.500106 | 0.151207 |
| 0.2 | kpx_group_2 | 0.438524 (45) | 0.466366 | 0.132179 |
| 0.2 | kpx_group_3 | 0.500893 (100) | 0.571429 | 0.224855 |
| 0.02 | kpx_group_1 | 0.418604 (66) | 0.524344 | 0.107267 |
| 0.02 | kpx_group_2 | 0.429704 (64) | 0.459624 | 0.085195 |
| 0.02 | kpx_group_3 | 0.479859 (81) | 0.564102 | 0.214852 |

두 LR 모두 train loss가 마지막 epoch까지 감소하므로 수치적으로 발산하지는 않았습니다. 반면 validation loss는 최저점 이후 다시 상승합니다. 특히 LR 0.02는 더 낮은 train loss와 최저 validation loss를 기록했지만 최종 비교 구간과 DACON 성능은 낮아졌습니다. 이는 학습 실패보다는 과적합 및 epoch 선택 구간과 실제 평가 구간 사이의 일반화 차이로 해석합니다.

#### LR 0.2

![LR 0.2 RealMLP train/validation loss](reports/v2/figures/training_curves.png)

#### LR 0.02

![LR 0.02 RealMLP train/validation loss](reports/v3/figures/training_curves.png)

## 6. 다음 단계
