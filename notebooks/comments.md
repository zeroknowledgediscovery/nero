# Notes on `paper_results005.ipynb`

## Purpose

`paper_results005.ipynb` is intended to be the analysis in
`paper_results002.ipynb` with the additional datasets introduced in
`paper_results003.ipynb`, while making the data paths repository-relative so
the notebook can run from a clean clone of `nero`.

The guiding rule is:

> Preserve the analysis logic of notebook 002, add the additional data represented
> in notebook 003, and use notebook 003's data-loading behavior for the newer CSVs.

## Where the NERO rates come from

Notebook 003 does **not** read NERO rates from the per-document JSON files and
does **not** invoke the NERO binary.

Its `getData()` function reads four columns directly from each detection CSV:

- `perplexity`
- `percentage`
- `entropy_rate`
- `rates`

The `rates` column is a string representation of a list. Notebook 003:

1. reads that column from the CSV;
2. drops rows where `rates` is missing;
3. converts the remaining strings with `ast.literal_eval`;
4. expands each list into `rates_0`, `rates_1`, ... columns.

`paper_results005.ipynb` now follows the same behavior exactly.

## Missing-rate policy

Rows with `rates == NaN` are excluded.

This is deliberate because it reproduces notebook 003. We do **not** currently:

- infer a missing vector from `entropy_rate`;
- replace a missing vector with zeros;
- recover it from an `entropy_rates/*.json` file;
- rerun `bin/nero` on the corresponding source text.

Those approaches could create a more complete dataset, but they would define a
different analysis cohort than notebook 003.

### Missing-rate audit observed in the current repository

During the 2026-09-22 audit, the API detection CSVs contained the following
numbers of rows with missing `rates`:

| Dataset | CSV rows | Missing `rates` |
| --- | ---: | ---: |
| GPT-4o API | 343 | 61 |
| GPT-4.0 API | 50 | 0 |
| LOC public domain | 101 | 24 |
| US public domain | 99 | 6 |
| GPT-5 | 219 | 15 |
| Claude Sonnet 4 | 1,015 | 35 |
| Gemini 2.5 Pro | 1,045 | 9 |

These counts are an audit of the repository state at that date and may change if
the CSV files are updated.

Some missing entries are recoverable from existing NERO JSONs or source text.
For example, many LOC entries have matching JSON outputs under truncated
filenames, and many AI entries have source text that could be rerun through
NERO. That recovery is **not** performed in notebook 005 because notebook 003
did not perform it.

## Dataset provenance in notebook 005

Notebook 005 combines two sources.

### Legacy data retained from notebook 002

The CSVs needed from the older `recogai_` repository were copied into:

`paper_data/legacy/`

They are used for:

- Project Gutenberg (legacy 002 table)
- arXiv papers
- long-form GPT-4o
- long-form GPT-3.5
- long-form GPT-4.0

Only the compact CSV tables needed by the notebook were copied, not the full
legacy repository.

### API-data tables represented in notebook 003

The following are read directly from the current repository under
`api_data_collection/ai_data/`:

- Library of Congress public-domain text
- U.S. public-domain text
- GPT-4o API
- GPT-4.0 API
- Claude Sonnet 4
- Gemini 2.5 Pro
- GPT-5

The current API Claude/Gemini/GPT-4o/GPT-5 tables are treated as the repository
representations of those model datasets, not as additional independent copies
to concatenate on top of equivalent 002 model tables. This avoids obvious
double-counting.

## Gutenberg choice

Notebook 003 also contains a Gutenberg detection table under
`api_data_collection/ai_data/gutenberg/`.

Notebook 005 keeps the Gutenberg dataset inherited from notebook 002 rather than
adding the 003 Gutenberg table as another human dataset. This follows the
original goal of adding the *new* datasets from 003 to 002 rather than
duplicating an already represented source category.

If the 003 Gutenberg table is later determined to represent a materially
different cohort rather than an updated/recomputed version of the same source,
it should be evaluated explicitly before being added as a separate dataset.

## `entropy_rate` versus the expanded `rates` vector

The notebook retains both quantities.

The scalar `entropy_rate` is taken directly from the CSV.

The expanded `rates_* ` columns come from the CSV `rates` list. Later cells
derive `nz_entropy_rate` by replacing zeros in the expanded rate vector with
missing values and taking the row-wise median. Therefore a row without a
`rates` vector cannot participate in the NERO-vector analyses as currently
defined.

## Reproducibility assumptions

A clean run assumes:

1. the repository is checked out with `api_data_collection/` and
   `paper_data/legacy/` present;
2. the API detection CSV schemas retain the columns expected by `getData()`;
3. `rates` remains a Python-list-like string parseable by
   `ast.literal_eval`;
4. missing `rates` rows continue to be excluded, consistent with notebook 003;
5. no generated/repaired NERO vectors are silently inserted into the analysis.

## Open issues

1. **Missingness may not be random.** Rows without NERO vectors can differ
   systematically by source/model, so dropping them may affect AUC estimates.
   Any manuscript analysis should report the retained sample sizes by dataset.

2. **Recoverable missing rates exist.** A separate sensitivity analysis could
   reconstruct all recoverable rate vectors from the raw JSONs/source text and
   compare results with the notebook-003-compatible complete-case analysis.

3. **Dataset identity/versioning.** Some model datasets moved from older
   `table_data` paths to the newer `api_data_collection` layout. Before a
   final paper freeze, hashes/sample IDs should be compared to document whether
   these are identical, supersets, or regenerated versions.

4. **Perturbation variants.** Some GPT-4o/Gemini rows with names such as
   `_1.05`, `_0.9`, or `_1.1` have missing rates and their transformed
   source text is not committed. These cannot be regenerated from the current
   repository alone.

5. **Row counts should be frozen for publication.** When final results are
   generated, record the Git commit and retained row counts after the
   `dropna(subset=["rates"])` step.

## Recommended paper-facing interpretation

The current notebook should be described as a **complete-case analysis with
respect to the NERO rate vector**. The inclusion criterion is operational:
a sample is included in rate-vector analyses only when the corresponding
detection CSV contains a valid `rates` vector.

A reconstructed-rate analysis, if performed later, should be presented as a
separate sensitivity analysis rather than silently mixed into the primary
notebook.
