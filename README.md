# global-oasis-future-suitability

Code accompanying the manuscript "Hydrological constraints and land-cover
filtering reshape the future oasis-suitability envelope across global drylands".

This repository provides the key code needed to inspect the modelling,
validation, sensitivity-analysis, and feasibility-screening workflow. It does
not include raw third-party data, derived rasters, model-ready tables, figure
source data, or table source data. Data will be made available on request, and
third-party products should be obtained from the original data providers listed
in `docs/data_sources.md`.

## What is included

- `scripts/01_core_pipeline/`: sampling, model-table preparation, model
  training, future projection, and feasibility-constraint scripts.
- `scripts/02_validation_sensitivity/`: predictor screening, independent
  validation, model/background sensitivity, hydrology sensitivity, and field
  control validation.
- `docs/script_manifest.csv`: source-to-repository mapping for the curated key-code
  package.

Manuscript figure and table generation scripts are intentionally omitted from
this GitHub package. Figure source data and table source data are not included
in this repository.

## Reproducing the analysis

The full workflow depends on external geospatial products and project-derived
intermediate files that are not included in this repository. For review, start
with:

1. Request the project-derived data from the authors if a full rerun is needed.
2. Obtain third-party products from the original providers listed in
   `docs/data_sources.md`.
3. Place the required input data under a local project data directory.
4. Install the Python packages listed in `requirements.txt`.
5. Read `docs/reproducibility.md` for the run order and known path assumptions.

Some scripts still contain local path constants because they were copied from
the working analysis project. The constants are kept visible rather than hidden
so reviewers can inspect the actual processing logic.

## Citation

This repository should be cited through the manuscript Code Availability
statement. Data will be made available on request.
