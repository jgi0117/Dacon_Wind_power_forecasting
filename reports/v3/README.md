# Version 3 RealMLP learning-rate experiments

| LR | Validation | DACON | 1-NMAE | FICR | Model outputs | Report |
|---:|---:|---:|---:|---:|:---|:---|
| 0.2 | 0.653255 | 0.631055 | 0.861265 | 0.400845 | `model_outputs/v3/lr_0p2/runs/realmlp` | `reports/v3/lr_0p2` |
| 0.02 | 0.650715 | 0.625799 | 0.861005 | 0.390592 | `model_outputs/v3/lr_0p02/runs/realmlp` | `reports/v3/lr_0p02` |
| 0.002 | 0.658018 | 0.630061 | 0.859115 | 0.401007 | `model_outputs/v3/lr_0p002/runs/realmlp` | `reports/v3/lr_0p002` |

통합 비교 값은 `lr_comparison.csv`에 정리되어 있습니다.

재정리 전 LR 0.2 중복본은 `archive/pre_v3_reorg` 아래에 보존되어 있습니다.
