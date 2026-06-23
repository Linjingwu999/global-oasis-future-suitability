# Reproducibility notes

This is a curated review package, not a dump of the full working directory.
It keeps the scripts needed to inspect modelling, validation, sensitivity
analysis, feasibility screening, and workflow environment documentation.

## Suggested order

1. Request the project-derived input data from the authors if a full rerun is
   needed, and obtain third-party products from their original providers.
2. Run the core pipeline scripts in `scripts/01_core_pipeline/` in filename
   order.
3. Run validation and sensitivity scripts in `scripts/02_validation_sensitivity/`.

## Notes

- Large rasters, model objects, and source products are intentionally excluded
  from GitHub.
- Manuscript figure and table generation scripts are intentionally excluded
  from GitHub. Figure source data and table source data are not included in this
  repository.
- Original third-party datasets should be downloaded from their providers and
  cited as described in `docs/data_sources.md`.
- Several scripts contain absolute paths from the analysis workstation. These
  paths should be changed to local input/output directories before rerunning.
- The code was curated for manuscript review. Before a public software release,
  the authors should select a license and, if needed, add a pinned environment
  file.
