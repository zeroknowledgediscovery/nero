# nero

https://zeroknowledgediscovery.github.io/nero/


# NERO: Nonparametric Entropy-Rate Oracle

This repository contains the reference implementation, data pipelines, and analysis workflows for **NERO (Nonparametric Entropy-Rate Oracle)**, as described in the paper:

**Complexity Signature of Generated Text**  
Ross Schmidt, Ishanu Chattopadhyay (2026)

NERO is a **model-agnostic, training-free framework** for characterizing long-form text by estimating its **entropy rate** under a fixed symbolization, and using that estimate as an intrinsic complexity signature to distinguish human-authored prose from LLM-generated text.

This repository is **not** a Python library.  
The **core entropy-rate estimator is provided as a compiled binary**, and all paper results are reproduced via analysis notebooks that invoke this binary and operate on the generated datasets.

---

## What NERO does (conceptual)

NERO maps text to a fixed **27-symbol alphabet** (26 lowercase letters + space) and treats the resulting stream as a realization of a stationary stochastic process. It then estimates the **Shannon entropy rate** of that process using a nonparametric PFSA-based method.

Key properties:

- no access to the generating model,
- no token probabilities or perplexity,
- no supervised training required,
- no calibration to specific LLMs.

For long-form text (> O(10⁴) characters), entropy-rate estimates produced by NERO are **systematically lower for contemporary LLM outputs than for human prose**, as demonstrated in the paper.

---

## Repository structure (important)

```
nero/
├── bin/                       # Compiled NERO binary (core utility)
├── api_data_collection/       # AI text generation pipelines & metadata
├── human_text_categories/     # Human-authored corpora & categorization
├── notebooks/                 # Paper reproduction & analysis notebooks
├── plotdata/                  # Derived tables used for figures
├── tex/                       # Paper / manuscript sources
├── docs/                      # GitHub Pages / rendered documentation
└── README.md
```

### `bin/` — core NERO implementation

This directory contains the **compiled NERO binary** implementing the nonparametric entropy-rate estimator described in the paper.

- This is the *authoritative implementation* of the estimator.
- There is **no Python implementation** of the core algorithm in this repo.
- All analyses call this binary either directly or via notebook wrappers.

If you are looking for “the NERO algorithm,” it lives here.

---

### `api_data_collection/` — AI-generated text

This directory contains the pipelines used to generate **long-form AI text** for the experiments.

As described in the paper:

- Texts were generated via **API and web-interface access** to multiple LLMs.
- Identical topic lists and prompting protocols were used across models.
- Each novel targets ~150k characters using iterative continuation prompts.
- Metadata for each generation (model, date, access modality, refusals, etc.) is stored alongside the text.

This folder corresponds directly to the *AI text generation* sections of the Methods and Supplementary Methods.

---

### `human_text_categories/` — human reference corpora

This directory contains the **human-authored text corpora** and their organization into categories used in the analysis.

As described in the paper, these include:

- Project Gutenberg English novels (after cleaning and filtering),
- arXiv technical manuscripts (converted from LaTeX),
- genre / topic groupings used for stratified analyses.

These data define the **human reference distributions** against which AI-generated text is compared.

---

### `notebooks/` — reproducing the paper

This is the **entry point for reproducing all reported results**.

The notebooks in this directory:

- invoke the NERO binary in `bin/`,
- operate on texts from `api_data_collection/` and `human_text_categories/`,
- generate entropy-rate estimates,
- construct ROC curves, AUC summaries, and length-dependence plots,
- produce the figures and tables reported in the paper.

If you are trying to understand *how the paper was produced*, start here.

---

### `plotdata/` — derived results

This directory contains **intermediate and plot-ready artifacts**, typically CSV or JSON files, produced by the notebooks.

These include:

- entropy-rate summaries,
- detector outputs (NERO and baseline detectors),
- cohort-level tables,
- length-truncation results.

If you want to regenerate figures without recomputing entropy rates from scratch, this is where cached results are read from.

---

### `tex/` — manuscript source

LaTeX sources and related assets for the paper live here.  
This directory is included for transparency and archival completeness.

---

## Reproducing results (summary)

To reproduce the paper results:

1. Ensure the **NERO binary** in `bin/` is executable on your system.
2. Inspect or regenerate datasets in:
   - `api_data_collection/` (AI text)
   - `human_text_categories/` (human text)
3. Open the notebooks in `notebooks/` and execute them in order.
4. Figures and tables are written to or read from `plotdata/`.

No model weights, APIs, or proprietary tools are required to *run NERO itself*—only to regenerate the AI text corpus.

---

## Computational notes

For a text of length `n`, alphabet size `|Σ| = 27`, and `M` substring-frequency cutoffs, the estimator runs in:

```
O(n · M · |Σ|)
```

For fixed `M`, runtime is linear in text length.

---

## License

MIT License.

---

## Citation

If you use this code or data, please cite:

Schmidt R, Chattopadhyay I.  
**Complexity Signature of Generated Text.** 2026.

