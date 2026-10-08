# Data provenance and paper reproduction

## Included public demonstration
`data/demo_cooke.json` is based on the upstream Cooke example geometry, with
synthetic materials substituted. `data/demo_relative_costs.csv` contains assumed
teaching coefficients, NOT supplier quotations or values used in the paper.
The six `h-demo-*` names are fictitious; their dispersion is generated from
specified n/V values using a Cauchy equation. The filename CDGM.AGF is retained
only for compatibility with the loader; it is NOT a manufacturer's catalog.
The public demo checks software operation and cannot reproduce paper percentages.

## Paper configurations
`configs/paper/` preserves the existing experiment budgets and objectives but
uses `private_data/` paths. Engineering prescriptions, supplier spreadsheets,
AGF catalogs, private refractive-index tables and production records are omitted.
Authorized local users can import their data with:

```powershell
python tools/import_local_research_data.py --source-root "PATH_TO_YOUR_RESEARCH_PROJECT"
python run.py --config configs/paper/main.yml
```

The import tool copies reference and seed lens JSON, relative-price CSV and
catalog files into ignored `private_data/`. The runner propagates that catalog
directory to all optimization subprocesses. Do not upload this directory.
Full paper reproduction requires these exact inputs, error models, random seeds,
matching environment and sufficient GPU budget. Paper configurations alone do not
constitute a self-contained reproduction dataset. Before publication, choose
whether to provide authorized research inputs or documented replacement data.
