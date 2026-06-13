# global-oasis-future-suitability

Code accompanying the manuscript on global oasis-compatible future suitability.

This repository is meant for peer review and reproducibility checking. It does
not include raw third-party data or large derived rasters. Those files are
provided through the associated Science Data Bank record, or should be obtained
from the original data providers listed in `docs/data_sources.md`.

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
this GitHub package. Figure and table reproducibility is supported through the
figure-source data, table-source data, final evidence files, and metadata in
the associated Science Data Bank package.

## Reproducing the analysis

The full workflow depends on external geospatial products and large derived
intermediate files. For review, start with:

1. Download the Science Data Bank dataset once the DOI or reviewer link is
   available.
2. Place the staged data under a local project data directory.
3. Install the Python packages listed in `requirements.txt`.
4. Read `docs/reproducibility.md` for the run order and known path assumptions.

Some scripts still contain local path constants because they were copied from
the working analysis project. The constants are kept visible rather than hidden
so reviewers can inspect the actual processing logic.

## Citation

The final GitHub URL and, if created, the archived software-release DOI should
be cited in the manuscript Code Availability statement.
