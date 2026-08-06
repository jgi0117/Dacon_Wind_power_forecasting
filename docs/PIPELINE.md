# Version 4 multi-task RealMLP execution

Version 4 uses one shared-trunk RealMLP with three capacity outputs and three
auxiliary activity logits in Python 3.13.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs @('--max-epochs','200','--learning-rate','0.02','--activity-loss-weight','0.15')

The model adds an activity-classification auxiliary objective to the original
per-group FICR-aware capacity objective.

    group_loss = 0.25 * smooth-MAE + 0.75 * (1 - soft-FICR)
    train_loss = mean(valid group losses) + 0.15 * activity_BCE

Missing group targets use a -1 sentinel. Capacity loss masks capacity factors
below 0.10, while activity BCE uses every observed target including genuine
zero and low-output rows. Missing targets contribute to neither objective.
This `all-history-masked` strategy retains 2022 rows for groups 1 and 2 and
removes only rows where all three targets are missing.

The validation run executes up to 200 epochs. The epoch with the lowest joint
FICR-aware capacity validation loss is used for final full-history training;
the activity objective does not select the epoch. Submission uses only the
three capacity outputs.

LR 0.02 outputs are stored under
`model_outputs/v4/activity_aux_lr_0p02/runs/realmlp` and
`reports/v4/activity_aux_lr_0p02`.
