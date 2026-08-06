# DACON 풍력 발전량 예측 — Version 4

기상 예보 기반 풍력 발전량 예측 파이프라인입니다. Version 3에서 RealMLP을 200 epoch로 학습하며 LR 0.2, 0.02, 0.002를 비교했습니다. Version 4는 LR 0.02를 기준으로 세 발전 그룹을 함께 학습하는 multi-task learning을 실험합니다.

- [Version 1 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v1.0.0): 5개 baseline 비교
- [Version 2 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v2.0.0): FICR-aware 모델 비교와 RealMLP 선정
- [Version 3 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v3.0.0): RealMLP 200-epoch learning-rate 비교

## 1. 프로젝트 개요

기상 예보와 시간 정보를 이용해 2025년의 시간별 풍력 발전량 3개 그룹(kpx_group_1~3)을 예측하는 회귀 프로젝트입니다. Version 4의 목표는 그룹별 독립 RealMLP을 shared trunk와 그룹별 prediction head를 갖는 하나의 multi-task 모델로 바꾸는 것입니다.

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

## 3. Version 4 학습 전략

### 시간 순 검증

1. 2023-12-31 14:00 이전 데이터를 학습에 사용합니다.
2. 경계에서 11시간을 purge합니다.
3. 2024년 1~3월을 epoch 선택용 validation으로 사용합니다.
4. 2024년 7~12월을 최종 비교 구간으로 유지합니다.
5. 선택된 epoch로 사용 가능한 전체 과거 데이터를 다시 학습해 테스트를 예측합니다.

하나의 shared trunk가 세 그룹의 공통 기상·시간 표현을 학습하고, 3-output prediction layer의 그룹별 head가 각 발전량을 출력합니다. 특정 그룹의 타깃이 없는 행은 해당 그룹 loss에서만 제외하는 target mask를 사용합니다. 따라서 group 1 head는 그룹 2·3의 정답을 직접 입력받지 않지만, shared trunk에 전달되는 gradient를 통해 다른 그룹의 학습 신호를 간접적으로 활용할 수 있습니다.

Version 3에서 train loss 수렴이 가장 안정적이었던 LR 0.02를 기준값으로 고정합니다. 최대 200 epoch를 실행하고 세 그룹의 평균 validation loss가 가장 낮은 epoch를 최종 전체 데이터 재학습 길이로 사용합니다. Version 3와 동일하게 `lr_sched=coslog4`, dropout 0.15, Adam을 유지해 모델 구조 변화의 효과만 비교합니다.

### FICR-aware loss

각 그룹에 대해 FICR의 6%와 8% 오차 경계를 sigmoid로 근사한 soft-FICR을 사용합니다.

    group_loss = 0.25 × smooth-MAE + 0.75 × (1-soft-FICR)
    total_loss = mean(valid group losses)

기본 FICR 비중은 0.75, sigmoid temperature는 0.01입니다. 전체 loss와 함께 그룹별 train loss, validation loss, competition score를 기록해 한 그룹의 개선이 다른 그룹의 악화에 가려지지 않도록 합니다.

## 4. 구현 및 산출물

Version 4는 하나의 RealMLP을 한 번 학습해 세 그룹을 동시에 예측합니다. 결측 타깃은 0 sentinel로 바꾼 뒤 competition eligibility 조건인 capacity factor 0.10 미만 mask에서 제외하므로 해당 head에는 gradient가 전달되지 않습니다. Version 3의 그룹별 독립 학습 코드는 [v3.0.0 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v3.0.0)에 보존되어 있습니다.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs @('--max-epochs','200','--learning-rate','0.02')

Version 4 산출물은 기존 결과와 섞이지 않도록 다음 경로를 사용합니다.

    model_outputs/v4/multitask_lr_0p02/
    reports/v4/multitask_lr_0p02/

- results.csv: 전체 validation 및 DACON 지표
- group_metrics.csv: 타깃별 지표
- monthly_metrics.csv: 월별 지표
- training_summary.csv: 최적 epoch와 학습 시간
- training_history.csv: epoch별 전체·그룹별 train/validation loss와 score
- figures/training_curves.png: 전체·그룹별 train/validation loss 변화

## 5. Version 4 기준선: Version 3 LR 비교

Version 3는 RealMLP을 200 epoch까지 학습하면서 LR만 변경하는 실험입니다. 아래 DACON 점수는 제출 화면의 값을 기록했습니다.

| LR | Validation score | DACON score | DACON 1-NMAE | DACON FICR | 상태 |
|---:|---:|---:|---:|---:|:---|
| 0.2 | 0.653255 | 0.631055 | 0.861265 | 0.400845 | 완료 |
| 0.02 | 0.650715 | 0.625799 | 0.861005 | 0.390592 | 완료 |
| 0.002 | 0.658018 | 0.630061 | 0.859115 | 0.401007 | 완료 |

LR 0.02는 LR 0.2보다 DACON score가 0.005256 낮았습니다. 1-NMAE 차이는 0.000260에 불과하지만 FICR은 0.010253 낮아, 현재 점수 하락은 주로 임계 오차 구간을 반영하는 FICR에서 발생했습니다.

LR 0.002는 가장 높은 validation score를 기록했지만 DACON score는 LR 0.2보다 0.000994 낮았습니다. FICR은 0.000162 높았으나 1-NMAE가 0.002150 낮아 최종 점수에서는 LR 0.2가 가장 좋았습니다. 따라서 이번 세 설정에서는 validation 순위와 DACON 순위가 일치하지 않습니다.

### Epoch별 train/validation loss

| LR | 타깃 | 최저 validation loss (epoch) | 마지막 validation loss | 마지막 train loss |
|---:|:---|---:|---:|---:|
| 0.2 | kpx_group_1 | 0.436019 (39) | 0.500106 | 0.151207 |
| 0.2 | kpx_group_2 | 0.438524 (45) | 0.466366 | 0.132179 |
| 0.2 | kpx_group_3 | 0.500893 (100) | 0.571429 | 0.224855 |
| 0.02 | kpx_group_1 | 0.418604 (66) | 0.524344 | 0.107267 |
| 0.02 | kpx_group_2 | 0.429704 (64) | 0.459624 | 0.085195 |
| 0.02 | kpx_group_3 | 0.479859 (81) | 0.564102 | 0.214852 |
| 0.002 | kpx_group_1 | 0.417816 (152) | 0.459712 | 0.405072 |
| 0.002 | kpx_group_2 | 0.432524 (159) | 0.451317 | 0.360022 |
| 0.002 | kpx_group_3 | 0.493675 (180) | 0.499069 | 0.461288 |

세 LR 모두 train loss가 마지막 epoch까지 감소하므로 수치적으로 발산하지는 않았습니다. LR 0.2와 0.02는 validation loss가 비교적 이른 최저점 이후 다시 상승해 과적합이 나타납니다. LR 0.002는 train loss가 상대적으로 높고 최적 epoch가 152~180으로 늦어 수렴 속도가 느리지만, 200 epoch 안에서 validation loss 최저점에는 도달했습니다. 낮은 validation loss가 DACON 개선으로 이어지지 않은 점은 epoch 선택 구간과 실제 평가 구간 사이의 일반화 차이도 함께 봐야 함을 보여줍니다.

#### LR 0.2

![LR 0.2 RealMLP train/validation loss](reports/v3/lr_0p2/figures/training_curves.png)

#### LR 0.02

![LR 0.02 RealMLP train/validation loss](reports/v3/lr_0p02/figures/training_curves.png)

#### LR 0.002

![LR 0.002 RealMLP train/validation loss](reports/v3/lr_0p002/figures/training_curves.png)

## 6. Version 4 실험 계획

Version 3의 세 설정은 DACON score 차이가 최대 0.005256으로 크지 않았고, 모두 train loss는 계속 감소하지만 validation loss는 0.4~0.5 부근에서 정체하거나 다시 상승했습니다. 독립 타깃 모델의 과적합과 타깃별 데이터 활용 한계를 Version 4의 핵심 문제로 봅니다.

1. LR 0.02의 그룹별 독립 RealMLP 결과를 baseline으로 사용합니다.
2. shared trunk와 세 개의 group head를 갖는 multi-task RealMLP을 구현합니다.
3. 동일한 split, loss, scheduler, dropout, 200-epoch 조건으로 구조 변화만 비교합니다.
4. 전체 점수뿐 아니라 그룹별 validation loss와 DACON 구성 지표를 비교합니다.
5. train loss는 감소하지만 validation loss가 정체하는 현상이 완화되는지 확인합니다.

Version 4 결과는 아직 없습니다.
