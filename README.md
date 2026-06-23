# global-oasis-future-suitability

Code accompanying the manuscript "Hydrological constraints and land-cover
filtering reshape the future oasis-suitability envelope across global drylands".

This repository is meant for peer review and reproducibility checking. It does
not include raw third-party data or large derived rasters. Those files are
provided through the associated Science Data Bank record
(https://doi.org/10.57760/sciencedb.39923), or should be obtained from the
original data providers listed in `docs/data_sources.md`.

## What is included

- `scripts/01_core_pipeline/`: sampling, model-table preparation, model
  training, future projection, and feasibility-constraint scripts.
- `scripts/02_validation_sensitivity/`: predictor screening, independent
  validation, model/background sensitivity, hydrology sensitivity, and field
  control validation.
- `scripts/04_data_package/`: helper used to assemble the Science Data Bank
  upload package.
- `docs/script_manifest.csv`: source-to-upload mapping for the curated review
  package.

Manuscript figure and table generation scripts are intentionally omitted from
this GitHub package. Figure and table verification is supported through the
main-figure source tables, table-source data, final evidence files, and
metadata in the associated Science Data Bank package.

## Reproducing the analysis

The full workflow depends on external geospatial products and large derived
intermediate files. For review, start with:

1. Download the Science Data Bank dataset at
   https://doi.org/10.57760/sciencedb.39923.
2. Place the staged data under a local project data directory.
3. Install the Python packages listed in `requirements.txt`.
4. Read `docs/reproducibility.md` for the run order and known path assumptions.

Some scripts still contain local path constants because they were copied from
the working analysis project. The constants are kept visible rather than hidden
so reviewers can inspect the actual processing logic.

## Citation

This repository should be cited through the manuscript Code Availability
statement. Supporting derived data are deposited in Science Data Bank:
https://doi.org/10.57760/sciencedb.39923.
