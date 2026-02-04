# Dataset layout and derived metrics

This repository contains AI-generated novels, the scripts and metadata used to generate and analyze them, and derived tables used for figures (including ROC/AUC vs. length).

The top-level structure is organized into three primary folders:

- `generated_txt/`
- `ai_data/`
- `length roc data/`

## `generated_txt/`

This folder contains subdirectories of generated novels. Each subdirectory name corresponds to the model that generated those novels.

If you need details about prompts, model versions, or generation settings, refer to the metadata stored under `ai_data/` (see below).

## `ai_data/`

Each subfolder corresponds to a generation/analysis run and contains:

- the script used to generate the novels
- `skipped_indices/` (indices of novels whose generation stopped early due to refusals)
- CSV table outputs used for analysis/figures
- `metadata/` (prompt metadata, including model/prompt specifics)
- `entropy rates/` (NERO outputs, one JSON per novel)

### `entropy rates/`

Contains the raw JSON outputs produced by NERO for each novel.

### `skipped_indices/`

Contains the index of each novel whose generation was stopped prematurely because the model refused to continue generating text.

### CSV tables

The CSV tables include only texts longer than **150,000 characters**.

When loading these CSVs in pandas, the first column is an index column and should be loaded with `index_col=0`. This index has different meanings depending on the source:

- For AI-generated novels: it matches the index of the topic used to generate that novel.
- For Project Gutenberg texts: it matches the Gutenberg index number for that file.

The topics used for AI generation are stored in:

- `ai_data/topics.json`

#### Column meanings

`perplexity`  
Perplexity predicted by HowkGPT (NYUAD) using the **middle 15,000 characters** of the text. The corresponding web-scraping script is in the `scripts/` folder.  
Source: https://howkgpt.nyuad.nyu.edu/

`prediction`  
Binary prediction produced by HowkGPT for the same input.

`percentage`  
Percentage predicted by ZeroGPT. The corresponding web-scraping script is in the `scripts/` folder.  
Source: https://www.zerogpt.com/

`entropy rate`  
The scalar entropy-rate summary derived from NERO. This value is the **median** of the `entropy rates` list.

`entropy rates`  
The full list of entropy-rate values reported by NERO for the text. The median of this list is reported in `entropy rate`.

Notes on entropy-rate preprocessing:
- Some NERO outputs include zeros; any zeros were removed prior to computing summaries.
- As a result, the stored `entropy rates` lists may differ from the raw NERO output if zeros were present.

Compatibility note:
- Some earlier CSVs may include additional columns (e.g., `confidence level`). These columns were not used for figures and can be ignored.

## `length roc data/`

This folder contains the data used for the figure that reports **AUC as a function of character length**.

It is structured by length bucket. Each length folder contains:
- the truncated/reduced-length generated text used in the AUC computations
- the corresponding NERO JSON outputs for those truncated texts
