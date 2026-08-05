# Version 3 RealMLP execution

Version 3 supports only RealMLP in Python 3.13.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs --max-epochs 200

The model uses the FICR-aware loss below and records train loss, validation loss, and competition score for every epoch.

    0.25 * smooth-MAE + 0.75 * (1 - soft-FICR)

The validation run executes up to 200 epochs. The epoch with the lowest FICR-aware validation loss is used for final full-history training.

Outputs are isolated under model_outputs/v3/runs/realmlp and reports/v3.
