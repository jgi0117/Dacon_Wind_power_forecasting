# Version 5 monthly-balanced random validation

The experimental 24-hour temporal encoder/decoder has been removed. Version 5
again uses the official PyTabKit RealMLP structure from Version 4:

- hidden sizes: 256 x 3
- Mish activation and parametric activation
- dropout 0.15 with flat-cos schedule
- weight decay 0.02 with flat-cos schedule
- Adam, squared momentum 0.95
- eight-model internal ensemble
- coslog4 learning-rate schedule
- shared capacity and activity outputs for group 1/2/3

Training remains row-wise. A true 24-hour joint output requires a sequence
encoder/decoder and therefore cannot be added while keeping the model structure
unchanged. Any future use of the 24-hour NWP trajectory must be implemented as
leakage-safe input feature engineering outside RealMLP.

The RealMLP model and optimizer remain unchanged. Seven complete 24-hour
forecast days are randomly selected from each month of 2023 using seed 42.
This gives every month exactly 168 validation rows. An 11-hour purge around
each selected batch prevents adjacent forecast rows from entering training.
The minimum balanced validation loss selects the refit epoch. A single model
is then refitted on all 2022-2023 history and evaluated on all of 2024. The
final submission model is refitted once on all available 2022-2024 labels.
The loss uses the original sigmoid FICR surrogate with temperature 0.01;
reliability weighting, stacking, Boundary Consistency, and Temporal GroupDRO
are disabled.

    .\scripts\setup_env.ps1
    .\scripts\run_models.ps1 -Models realmlp -Device cpu -PipelineArgs @('--max-epochs','200','--learning-rate','0.02','--activity-loss-weight','0.15')

Submission generation remains enabled by default. Outputs are stored under:

    model_outputs/v5/monthly_random_sigmoid_lr_0p02/runs/realmlp
    reports/v5/monthly_random_sigmoid_lr_0p02
