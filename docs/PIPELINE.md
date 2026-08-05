# Model execution

LightGBM, CatBoost, TabM, RealMLP, and xRFM run in Python 3.13. The launcher
runs each requested model in an independent process.

```powershell
.\scripts\setup_env.ps1
.\scripts\run_models.ps1 -Models tabm,realmlp,xrfm -Device cpu
```

Models execute sequentially in exactly the supplied order. Outputs are written
to `model_outputs/runs/<model>/`. Use `-ReuseCompleted` to skip a model only
when its validation files, report, and submission already exist.

Common training controls can be forwarded to the pipeline:

```powershell
.\scripts\run_models.ps1 -Models tabm -Device cpu -PipelineArgs `
  '--max-epochs','20','--early-stopping-patience','10','--batch-size','256'
```

xRFM uses recursive feature-learning iterations rather than neural-network
epochs. Its defaults are 8 maximum iterations, 2,000 rows per leaf, and an AGOP
batch size of 64. Each leaf restores the iteration with the best chronological
competition score; the median selected iteration is used for final full-data
training.

```powershell
.\scripts\run_models.ps1 -Models xrfm -Device cpu -PipelineArgs `
  '--xrfm-iterations','8','--xrfm-max-leaf-samples','2000','--xrfm-m-batch-size','64'
```

The early-stopping metric is the target-level competition score, including
both nMAE and FICR. January-March 2024 is used for epoch selection and July 2024
onward remains untouched for final model comparison. Final submission training
uses all labeled rows before the test forecast cutoff for the selected epoch.

Supported names are `lightgbm`, `catboost`, `tabm`, `realmlp`, and `xrfm`.
`all` is the only launcher alias.
