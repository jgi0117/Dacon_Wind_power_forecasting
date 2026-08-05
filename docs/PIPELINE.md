# Version 2 model execution

LightGBM, CatBoost, TabM, and RealMLP run in Python 3.13. Each requested model executes in an independent process and writes to model_outputs/v2/runs/model-name.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models lightgbm,catboost,tabm,realmlp -Device cpu

All models run for up to 200 epochs or boosting rounds by default. The best FICR-aware validation-loss step is selected and then used for final full-history training. Use --max-epochs to change the limit.

The default differentiable surrogate is:

    0.25 * smooth-MAE + 0.75 * (1 - soft-FICR)

The two FICR thresholds are smoothed with sigmoid temperature 0.01. Both values are configurable.

    .\scripts\run_models.ps1 -Models tabm,realmlp -Device cpu -PipelineArgs --ficr-weight 0.8 --ficr-temperature 0.008 --max-epochs 200

Models execute sequentially in the supplied order. Use -ReuseCompleted only when the corresponding V2 validation report and submission are complete.

After execution, scripts/build_report.py consolidates available model reports into reports/v2. It creates overall, target-level, monthly, training-summary, and training-history CSV files together with score and loss-history figures.

Supported names are lightgbm, catboost, tabm, and realmlp. all is the only launcher alias.
