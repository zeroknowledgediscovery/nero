# Complete detector data

Goal: produce complete, auditable detector tables for the paper analyses. **Do not impute NERO vectors.** Preserve original CSVs and write completed copies.

## 1. NERO

### Recover from existing JSON
- LOC public domain: **20 rows**

### Run/re-run NERO from existing source text
- API GPT-4o: **49 rows**
- GPT-5: **15 rows**
- Claude 4: **35 rows**
- Gemini 2.5: **4 rows**
- Re-run `gemini_25_pro732.txt` (current vector has 40/41 components)

### Locate source text, then run NERO
- LOC: **4 rows**
- US public domain: **6 rows**
- API GPT-4o: **12 `_1.05` perturbation rows**
- Gemini 2.5: **5 `_0.9`/`_1.1` perturbation rows**
- Legacy GPT-3.5 row **35** has 40/41 components; rerun if source text is found

Already NERO-complete: legacy Gutenberg, legacy arXiv, legacy GPT-4.0, legacy GPT-4o, API GPT-4.0.

## 2. Binoculars

Run Binoculars for all rows in legacy datasets:
- Gutenberg: **406**
- arXiv: **42**
- GPT-3.5: **100**
- GPT-4.0: **41**
- GPT-4o: **98**

Fill missing Binoculars values in newer tables:
- LOC: **6**
- US public domain: **1**
- API GPT-4o: **4**
- GPT-5: **1**
- Claude 4: **4**
- Gemini 2.5: **21**

API GPT-4.0 is complete.

## 3. ZeroGPT

Only fill true blank `percentage` values. **Do not treat 0 as missing.**

- API GPT-4o: **36 rows**
- API GPT-4.0: **1 row** (`gpt_4zero18.txt`)

All other datasets are complete for ZeroGPT.

## Required output

For each dataset:
1. completed CSV;
2. provenance table with columns:

`dataset, filename, field, source`

Use one of:

`original_csv`, `existing_nero_json`, `rerun_nero`, `run_binoculars`, `run_zerogpt`.

Do not overwrite the original CSVs.
