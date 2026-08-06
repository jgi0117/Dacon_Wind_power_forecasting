# Version 4 multi-task RealMLP execution

Version 4 uses one shared-trunk RealMLP with three regression outputs in Python 3.13.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -EvaluationOnly -PipelineArgs @('--max-epochs','200','--learning-rate','0.02')

The model uses an equal-weighted mean of the valid per-group FICR-aware losses.

    group_loss = 0.25 * smooth-MAE + 0.75 * (1 - soft-FICR)
    total_loss = mean(valid group losses)

Missing group targets are replaced with a zero sentinel before entering PyTabKit.
The loss masks capacity factors below 0.10, so a missing target contributes no
gradient while the other available groups still update the shared trunk.
This `all-history-masked` strategy retains 2022 rows for groups 1 and 2 and
removes only rows where all three targets are missing.

The validation run executes up to 200 epochs. The epoch with the lowest joint
FICR-aware validation loss is used for final full-history training.

LR 0.02 outputs are stored under
`model_outputs/v4/multitask_lr_0p02/runs/realmlp` and
`reports/v4/multitask_lr_0p02`.
