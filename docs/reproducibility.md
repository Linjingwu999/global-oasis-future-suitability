# Reproducibility notes

This is a curated review package, not a dump of the full working directory.
It keeps the scripts needed to inspect the methods, validation checks, key
figures, and Science Data Bank package construction.

## Suggested order

1. Prepare input data from the Science Data Bank package and original providers.
2. Run the core pipeline scripts in `scripts/01_core_pipeline/` in filename
   order.
3. Run validation and sensitivity scripts in `scripts/02_validation_sensitivity/`.
4. Generate manuscript figures and tables from `scripts/03_figures_tables/`.
5. Use `scripts/04_data_package/sciencedb_package_builder.py` only when
   rebuilding the data-deposit package.

## Notes

- Large rasters, model objects, and source products are intentionally excluded
  from GitHub.
- Original third-party datasets should be downloaded from their providers and
  cited as described in `docs/data_sources.md`.
- Several scripts contain absolute paths from the analysis workstation. These
  paths should be changed to local input/output directories before rerunning.
- The code was curated for manuscript review. Before a public software release,
  the authors should select a license and, if needed, add a pinned environment
  file.
