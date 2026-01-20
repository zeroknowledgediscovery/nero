# Gutenberg + Entropy-Rate Augmentation Pipeline

This repo is a small ETL pipeline that (1) pulls Project Gutenberg metadata from the Gutendex API, (2) merges that metadata into your existing per-book JSON outputs (the ones containing entropy-rate–related fields), (3) flattens everything into a single CSV for downstream analysis, and (4) explores/aggregates categories in a notebook.

## Execution order

1. `getguteninfo2.py`
2. `mergejsons.py`
3. `processjsonstocsv.py`
4. `textcategories.ipynb`

---

## 1. `getguteninfo2.py` — Fetch Gutendex metadata

**Role.**  
Downloads metadata for Project Gutenberg books from the Gutendex API, one JSON file per Gutenberg ID.

**Inputs.**
- `indexes_.dat`: plain-text file containing Gutenberg numeric IDs, one per line.

**Behavior.**
- Queries `https://gutendex.com/books/<ID>/`.
- Writes results to `Gutenberg_json/gutenberg_<ID>.json`.
- Skips already-downloaded IDs.
- Handles rate limiting (HTTP 429) and transient server errors with retries.
- Writes `.404` marker files for IDs that do not exist.

**Output.**
- `Gutenberg_json/` directory containing one JSON file per valid ID.

**Run.**
```bash
python3 getguteninfo2.py
```

---

## 2. `mergejsons.py` — Augment entropy/index JSONs

**Role.**  
Merges Gutendex metadata into your existing per-book JSON outputs (e.g., those containing entropy rates or posterior probabilities).

**Inputs.**
- `Gutenberg_json/gutenberg_<ID>.json`
- `entropy_rates/Guten_index<ID>.json`

**Behavior.**
- Matches files by Gutenberg ID.
- Inserts Gutendex metadata under the `gutenberg` key.
- Avoids overwriting existing fields.
- Skips files that do not match the expected naming pattern.

**Output.**
- `augmented_jsons/` directory containing merged JSON files.

**Run.**
```bash
python3 mergejsons.py
```

---

## 3. `processjsonstocsv.py` — Flatten to CSV

**Role.**  
Converts the merged JSON files into a single, analysis-ready CSV.

**Inputs.**
- `augmented_jsons/*.json`

**Extracted fields.**
- Gutenberg ID
- Title
- Author(s)
- Subject categories
- Median entropy rate
- Mean entropy rate
- Estimated posterior probability (if present)
- Source JSON filename

**Output.**
- `guten.csv`

**Run.**
```bash
python3 processjsonstocsv.py
```

---

## 4. `textcategories.ipynb` — Analysis and categorization

**Role.**  
Notebook for exploratory analysis and visualization:
- Cleaning and aggregating subject labels
- Grouping texts by genre/category
- Plotting entropy-rate distributions
- Building higher-level category abstractions

**Input.**
- `guten.csv`

---

## Typical directory layout

```
indexes_.dat
entropy_rates/
  Guten_index123.json
Gutenberg_json/
  gutenberg_123.json
augmented_jsons/
  Guten_index123.json
guten.csv
textcategories.ipynb
```

---

## Dependencies

- Python 3
- requests
- numpy
- pandas

Install with:
```bash
pip install requests numpy pandas
```
