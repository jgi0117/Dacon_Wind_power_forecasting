# DACON 풍력 발전량 예측 — Final Competition Summary

기상 예보와 시간 정보를 이용해 2025년의 시간별 풍력 발전량 3개 그룹(`kpx_group_1~3`)을 예측한 DACON 프로젝트입니다.

최종적으로 **Version 5 Distilled RealMLP**이 프로젝트 최고 성능을 기록했으며, 대회 종료 후 Private leaderboard에서 **0.64094, 전체 2,116명 중 233위**로 마무리했습니다.

버전별 상세 구현과 실험 과정은 각 Git tag에 보존되어 있습니다.

- [Version 1 — v1.0.0](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v1.0.0): 5개 baseline 비교
- [Version 2 — v2.0.0](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v2.0.0): FICR-aware loss와 RealMLP 선정
- [Version 3 — v3.0.0](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v3.0.0): RealMLP 200-epoch learning-rate 비교
- [Version 4 — v4.0.0](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v4.0.0): multi-task RealMLP + activity auxiliary objective
- [Version 5 — v5.0.0](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v5.0.0): Teacher–Student Distillation
- Version 6: Group 3 FICR 개선을 위한 마지막 worst-group FICR regularization 실험

---

## 1. 프로젝트 개요

평가식은 다음과 같습니다.

```text
score = 0.5 × (1-NMAE) + 0.5 × FICR
```

따라서 평균 오차뿐 아니라 실제 발전량 대비 예측 오차가 6%, 8% 경계 안에 들어오는지를 반영하는 FICR이 최종 점수에 동일한 비중으로 반영됩니다.

학습 데이터는 2022-01-01 01:00부터 2025-01-01 00:00까지의 시간별 데이터이며, 시간 정보와 LDAPS/GFS 기상 예보를 입력으로 사용했습니다.

프로젝트 전반에서 Group 3의 FICR이 Group 1·2보다 낮은 경향이 지속되었습니다. Group 3는 2022년 target이 존재하지 않아 label history가 더 짧고, Group 1·2와 터빈 종류도 다릅니다.

- Group 1·2: VESTAS V126
- Group 3: UNISON U136
- Group 3 target 시작: 2023-01-01 01:00

---

## 2. Version 5 — 최종 최고 성능 모델

Version 5에서는 **Teacher–Student Distillation**을 적용했습니다. Teacher는 학습 단계에서만 과거 기상 및 실제 발전량 history를 사용하고, Student는 실제 Test와 동일하게 추론 시 사용할 수 있는 정보만 입력받도록 구성했습니다.

Teacher prediction은 chronological OOF 방식으로 생성하고 hard label과 혼합해 Student target으로 사용했습니다.

```text
y_distilled
= 0.80 × y_true
+ 0.20 × y_teacher
```

Version 5의 최고 Public score는 **0.638408**이며, 프로젝트 전체에서 가장 높은 Public 성능이었습니다.

Version 5의 구조, 세부 설정, validation 결과, 그룹별 분석 및 학습 과정은 [v5.0.0 tag](https://github.com/jgi0117/Dacon_Wind_power_forecasting/tree/v5.0.0)에 정리되어 있습니다.

---

## 3. Version 6 — 마지막 실험

Version 6의 여러 중간 시도는 최종 README에서 제외하고, 대회 종료 직전에 수행한 마지막 실험만 기록합니다.

### 목표

Version 5에서도 Group 3의 FICR이 Group 1·2보다 낮게 나타났습니다. Epoch selection 변경만으로는 그룹 간 성능 차이를 근본적으로 줄이기 어렵다고 판단해, 마지막 실험에서는 loss 수준에서 worst group을 직접 보정했습니다.

Teacher와 Student 모두 동일한 FICR-aware loss를 사용하고, 가장 FICR loss가 큰 그룹에 추가 regularization을 적용했습니다.

```text
base loss
+ worst_group_ficr_reg_weight
  × (worst_group_ficr_loss - mean_group_ficr_loss)
```

마지막 설정은 다음과 같습니다.

```text
worst_group_ficr_reg_weight = 0.50
```

즉 Teacher와 Student 모두 동일하게 worst-group FICR regularization을 적용했습니다.

### 실행 설정

```bash
python run_pipeline.py \
  --models realmlp \
  --device cuda \
  --artifacts-dir artifacts_compat \
  --data-dir data \
  --output-dir model_outputs/v6/worst_group_ficr_05 \
  --max-epochs 200 \
  --learning-rate 0.02 \
  --student-history-hours 5 \
  --teacher-history-hours 5 \
  --history-decay 0.8 \
  --teacher-epochs 100 \
  --teacher-oof-folds 3 \
  --teacher-min-train-rows 720 \
  --teacher-inner-validation-fraction 0.20 \
  --distillation-teacher-weight 0.20 \
  --worst-group-ficr-reg-weight 0.50
```

### 결과

| Model | Public score | Public 1-NMAE | Public FICR |
| :--- | ---: | ---: | ---: |
| **Version 5 Distilled RealMLP** | **0.638408** | **0.867068** | **0.409749** |
| Version 6 Final Worst-group FICR | 0.623200 | 0.867676 | 0.378724 |

Version 6에서는 1-NMAE가 `0.867068 → 0.867676`으로 소폭 상승했지만 FICR이 `0.409749 → 0.378724`로 크게 하락하면서 전체 Public score가 **0.623200**까지 감소했습니다.

즉 Group 3의 낮은 FICR을 개선하기 위해 worst-group penalty를 강하게 적용했지만, 실제 leaderboard에서는 전체 FICR optimization이 오히려 악화되었습니다. Competition FICR은 6%, 8% 오차 경계를 기준으로 보상이 비연속적으로 변하기 때문에, smooth surrogate loss에서 특정 그룹에 강한 penalty를 부여하는 것이 실제 FICR 상승으로 직접 이어지지는 않았습니다.

---

## 4. 버전별 리더보드 결과

각 버전에서 기록한 가장 높은 **Public score**를 기준으로 정리했습니다.

Private leaderboard는 대회 종료 후 **최종 선택 제출**에 대해서만 확인할 수 있었기 때문에, Version 1~4와 마지막 Version 6 실험에는 확인 가능한 Private score가 없습니다. 최종 선택 모델인 Version 5의 Private score만 실제 값으로 기록합니다.

| Version | 핵심 변경 | 최고 Public score | Private score | 비고 |
| :--- | :--- | ---: | ---: | :--- |
| Version 1 | 5개 baseline 비교 | 0.624299 | — | RealMLP baseline 최고 |
| Version 2 | FICR-aware loss | 0.630536 | — | RealMLP 선정 |
| Version 3 | 200 epoch LR 비교 | 0.631055 | — | LR 0.2 최고 |
| Version 4 | Multi-task + activity auxiliary | 0.632874 | — | FICR 개선 |
| **Version 5** | **Teacher–Student Distillation** | **0.638408** | **0.64094** | **최종 선택 / 최고 성능** |
| Version 6 | Worst-group FICR reg. 0.50 | 0.623200 | — | 마지막 실험, 성능 하락 |

Public score는 Version 1의 `0.624299`에서 Version 5의 `0.638408`까지 단계적으로 개선되었습니다. Version 6의 추가 loss regularization은 이 흐름을 이어가지 못했고 최종 모델 선정에는 반영하지 않았습니다.

### 최종 Private leaderboard

| 항목 | 최종 결과 |
| :--- | ---: |
| Private score | **0.64094** |
| Private 1-NMAE | **0.87144** |
| Private FICR | **0.41044** |
| 최종 순위 | **233 / 2,116** |
| 상위 비율 | **약 11.0%** |

최종 순위는 전체 2,116명 중 **233위**로 마무리했습니다.

Public `0.638408`과 Private `0.64094`의 차이가 크지 않았고, Private에서는 오히려 소폭 상승했습니다. 따라서 최종 선택한 Version 5 모델은 Public leaderboard에만 과도하게 맞춰진 모델이라기보다 unseen evaluation 구간에서도 비교적 안정적으로 일반화한 것으로 해석했습니다.

---

## 5. 구현 및 주요 산출물

핵심 구현 파일은 다음과 같습니다.

```text
src/baram/models/distilled_realmlp_model.py
src/baram/models/group_conditioned_realmlp_model.py
src/baram/models/realmlp_model.py
src/baram/metrics.py
```

주요 산출물은 다음 경로에 저장합니다.

```text
model_outputs/
reports/
```

- `evaluation_results.csv`: 전체 validation 결과
- `evaluation_results_by_month.csv`: 월별 validation 결과
- `validation_predictions.csv`: validation prediction
- `student_training_history.csv`: Student epoch별 학습 기록
- `teacher_oof_report.csv`: Teacher OOF 및 epoch 정보
- `training_report.csv`: 학습 요약
- `run_report.json`: 실행 configuration 및 metadata
- `submission_realmlp.csv`: DACON 제출 파일

---

## 6. 최종 결론

프로젝트에서 가장 효과적이었던 개선은 **Version 5의 Teacher–Student Distillation**이었습니다.

Test 시점에는 실제 과거 발전량을 직접 사용할 수 없지만, 학습 단계의 Teacher가 과거 target과 기상 history를 활용하고 이를 chronological OOF soft target으로 Student에 전달함으로써 실제 inference 조건을 유지하면서 temporal information을 간접적으로 학습시켰습니다.

그 결과 Version 1의 RealMLP baseline `0.624299`에서 Version 5 `0.638408`까지 Public score를 개선했으며, 최종 Private score는 **0.64094**를 기록했습니다.

반면 마지막 Version 6에서는 지속적으로 낮았던 Group 3 FICR을 직접 개선하기 위해 Teacher와 Student 양쪽에 `0.50`의 worst-group FICR regularization을 적용했지만, NMAE는 유지된 반면 FICR이 크게 감소해 Public score가 `0.623200`으로 하락했습니다.

이를 통해 다음을 확인했습니다.

1. **이 대회에서는 최종 score 변화가 NMAE보다 FICR 변화에 더 민감하게 나타나는 경우가 많았습니다.**
2. **FICR을 직접 겨냥한 surrogate loss가 항상 실제 leaderboard FICR 개선으로 이어지는 것은 아니었습니다.**
3. **Group 3의 성능 저하는 단순 loss weight 문제보다 짧은 label history, 다른 터빈 구성, temporal distribution 차이가 함께 작용한 문제로 보는 것이 타당했습니다.**
4. **복잡도를 추가한 Version 6보다 Version 5의 Teacher–Student 구조가 실제 unseen evaluation에서 더 안정적이었습니다.**
5. **Validation 성능뿐 아니라 실제 competition metric과 시간축 일반화 성능을 함께 확인하는 것이 중요했습니다.**

최종적으로 **Version 5 Distilled RealMLP을 프로젝트 최종 모델로 선정**했습니다.
