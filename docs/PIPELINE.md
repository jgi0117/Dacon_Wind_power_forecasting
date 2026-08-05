# Version 2 model execution

LightGBM, CatBoost, TabM, RealMLP, and xRFM run in Python 3.13. Each requested model executes in an independent process and writes to model_outputs/v2/runs/model-name.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models lightgbm,catboost,tabm,realmlp,xrfm -Device cpu

All models run for 20 epochs, boosting rounds, or recursive feature-learning iterations during validation. The best FICR-aware validation-loss step is selected and then used for final full-history training.

The default differentiable surrogate is:

    0.25 * smooth-MAE + 0.75 * (1 - soft-FICR)

The two FICR thresholds are smoothed with sigmoid temperature 0.01. Both values are configurable.

    .\scripts\run_models.ps1 -Models tabm,realmlp -Device cpu -PipelineArgs --ficr-weight 0.8 --ficr-temperature 0.008 --max-epochs 20

xRFM uses kernel-ridge fitting internally and does not accept an external gradient loss. Its internal objective remains unchanged, while FICR-aware loss selects the best recursive iteration. The library exposes mean leaf validation history but not leaf train loss.

Models execute sequentially in the supplied order. Use -ReuseCompleted only when the corresponding V2 validation report and submission are complete.

After execution, scripts/build_report.py consolidates available model reports into reports/v2. It creates overall, target-level, monthly, training-summary, and training-history CSV files together with score and loss-history figures.

Supported names are lightgbm, catboost, tabm, realmlp, and xrfm. all is the only launcher alias.
