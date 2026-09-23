#!/usr/bin/env python3
"""
Compare NERO (zero-shot and trained) with external AI-text detector scores
across older/newer human and AI corpus groupings.

Outputs:
  dataset_coverage.csv
  within_scenario_auc.csv
  within_scenario_trained_repeats.csv
  transfer_auc.csv
  transfer_trained_repeats.csv

Design principles
-----------------
1. NERO zero-shot = negative nz_entropy_rate, where nz_entropy_rate is the
   median of nonzero rates_0..rates_40. AI is the positive class, so lower
   NERO values correspond to larger AI scores after sign inversion.
2. NERO trained = RandomForest on exactly 46 NERO-derived features:
      entropy_rate
      rates_0..rates_40
      nz_entropy_rate, nzmean, nzstd, nzero
3. External scores:
      HowkGPT perplexity : higher => AI
      ZeroGPT percentage : higher => AI
      Binoculars         : lower  => AI
4. Missing detector scores are never imputed for zero-shot detector AUCs.
   Each detector reports the number of human/AI samples actually used.
5. Missing NERO features are imputed with zero, with the imputer fitted on
   training data only.
6. The script deliberately keeps dataset provenance explicit so older/newer
   model and human-source effects can be separated.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    path: str
    label: int
    era: str
    family: str
    display: str


DATASETS = [
    DatasetSpec(
        "human_gutenberg_old",
        "paper_data/legacy/gutenberg/gutenberg_table.csv",
        0, "old", "human", "Gutenberg (legacy)"
    ),
    DatasetSpec(
        "human_arxiv_old",
        "paper_data/legacy/arxiv_entropy/arxiv_table.csv",
        0, "old", "human", "arXiv (legacy)"
    ),
    DatasetSpec(
        "human_loc_new",
        "api_data_collection/ai_data/loc-pd/loc-pd_detection.csv",
        0, "new", "human", "Library of Congress"
    ),
    DatasetSpec(
        "human_uspd_new",
        "api_data_collection/ai_data/us-pd/us-pd_detection.csv",
        0, "new", "human", "US public domain"
    ),
    DatasetSpec(
        "ai_gpt35_old",
        "paper_data/legacy/ai_longform_gpt3.5/gpt_35_table.csv",
        1, "old", "gpt", "GPT-3.5 (legacy longform)"
    ),
    DatasetSpec(
        "ai_gpt40_old",
        "paper_data/legacy/ai_longform_gpt4.0/gpt_40_table.csv",
        1, "old", "gpt", "GPT-4.0 (legacy longform)"
    ),
    DatasetSpec(
        "ai_gpt4o_old",
        "paper_data/legacy/ai_longform_gpt4o/ai_web_longform_table.csv",
        1, "old", "gpt", "GPT-4o (legacy longform)"
    ),
    DatasetSpec(
        "ai_gpt4o_new",
        "api_data_collection/ai_data/openai/gpt4o/gpt_4o_detection.csv",
        1, "new", "gpt", "GPT-4o (API)"
    ),
    DatasetSpec(
        "ai_gpt40_new",
        "api_data_collection/ai_data/openai/gpt4.0/gpt_4zero_detection.csv",
        1, "new", "gpt", "GPT-4.0 (API)"
    ),
    DatasetSpec(
        "ai_gpt5_new",
        "api_data_collection/ai_data/openai/gpt5/gpt_5_detection.csv",
        1, "new", "gpt", "GPT-5"
    ),
    DatasetSpec(
        "ai_claude4_new",
        "api_data_collection/ai_data/claude_sonnet_4/claude_detection.csv",
        1, "new", "non_gpt", "Claude Sonnet 4"
    ),
    DatasetSpec(
        "ai_gemini25_new",
        "api_data_collection/ai_data/gemini_2.5_pro/gemini_25_detection.csv",
        1, "new", "non_gpt", "Gemini 2.5 Pro"
    ),
]


HUMAN_GROUPS = {
    "old_human": ["human_gutenberg_old", "human_arxiv_old"],
    "new_human": ["human_loc_new", "human_uspd_new"],
    "all_human": [
        "human_gutenberg_old", "human_arxiv_old",
        "human_loc_new", "human_uspd_new"
    ],
}


AI_GROUPS = {
    "old_ai": ["ai_gpt35_old", "ai_gpt40_old", "ai_gpt4o_old"],
    "new_ai": [
        "ai_gpt4o_new", "ai_gpt40_new", "ai_gpt5_new",
        "ai_claude4_new", "ai_gemini25_new"
    ],
    "all_ai": [
        "ai_gpt35_old", "ai_gpt40_old", "ai_gpt4o_old",
        "ai_gpt4o_new", "ai_gpt40_new", "ai_gpt5_new",
        "ai_claude4_new", "ai_gemini25_new"
    ],
    "all_gpt": [
        "ai_gpt35_old", "ai_gpt40_old", "ai_gpt4o_old",
        "ai_gpt4o_new", "ai_gpt40_new", "ai_gpt5_new"
    ],
    "new_gpt": ["ai_gpt4o_new", "ai_gpt40_new", "ai_gpt5_new"],
    "new_non_gpt": ["ai_claude4_new", "ai_gemini25_new"],
    "pre5_gpt": [
        "ai_gpt35_old", "ai_gpt40_old", "ai_gpt4o_old",
        "ai_gpt4o_new", "ai_gpt40_new"
    ],
    "new_pre5_gpt": ["ai_gpt4o_new", "ai_gpt40_new"],
    "gpt5": ["ai_gpt5_new"],
    "gpt35_old": ["ai_gpt35_old"],
    "gpt40_old": ["ai_gpt40_old"],
    "gpt4o_old": ["ai_gpt4o_old"],
    "gpt4o_new": ["ai_gpt4o_new"],
    "gpt40_new": ["ai_gpt40_new"],
    "claude4": ["ai_claude4_new"],
    "gemini25": ["ai_gemini25_new"],
}


# Explicit transfer experiments. Human samples are split independently between
# train/test so the same human document never appears in both sides.
TRANSFER_GROUPS = {
    "old_to_new_ai": ("old_ai", "new_ai"),
    "old_to_new_gpt": ("old_ai", "new_gpt"),
    "pre5_gpt_to_gpt5": ("pre5_gpt", "gpt5"),
    "new_pre5_gpt_to_gpt5": ("new_pre5_gpt", "gpt5"),
    "all_pre5_ai_to_gpt5": (
        ["ai_gpt35_old", "ai_gpt40_old", "ai_gpt4o_old",
         "ai_gpt4o_new", "ai_gpt40_new", "ai_claude4_new", "ai_gemini25_new"],
        "gpt5"
    ),
    "new_non_gpt_to_gpt5": ("new_non_gpt", "gpt5"),
}


SCORE_SPECS = {
    # AI is positive class.
    "NERO_zero": ("nz_entropy_rate", -1.0),
    "NERO_entropy_rate": ("entropy_rate", -1.0),
    "HowkGPT_perplexity": ("perplexity", +1.0),
    "ZeroGPT_percentage": ("percentage", +1.0),
    "Binoculars": ("binoculars_score", -1.0),
}


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    cols = []
    seen = {}
    for c in out.columns:
        c = str(c).replace("\ufeff", "").strip().lower().replace(" ", "_")
        c = c.replace("-", "_")
        if c.startswith("unnamed:"):
            c = "index"
        if c in seen:
            seen[c] += 1
            c = f"{c}.{seen[c]}"
        else:
            seen[c] = 0
        cols.append(c)
    out.columns = cols

    # Harmonize known spelling variants.
    rename = {
        "binoculars": "binoculars_score",
        "text_length": "text_length",
        "entropy_rate": "entropy_rate",
    }
    out = out.rename(columns={c: rename.get(c, c) for c in out.columns})
    return out


def _parse_rates(value):
    if pd.isna(value):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        return list(value)
    try:
        x = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return None
    return list(x) if isinstance(x, (list, tuple, np.ndarray)) else None


def load_dataset(root: Path, spec: DatasetSpec) -> pd.DataFrame:
    path = root / spec.path
    df = pd.read_csv(path)
    df = _normalize_columns(df)

    # Keep a stable document identifier when possible.
    id_candidates = [c for c in df.columns if c in {"index", ""}]
    if id_candidates:
        doc_id = df[id_candidates[0]].astype(str)
    else:
        doc_id = pd.Series(np.arange(len(df)), index=df.index).astype(str)

    parsed = df["rates"].apply(_parse_rates) if "rates" in df.columns else pd.Series([None] * len(df))

    rate_mat = np.full((len(df), 41), np.nan, dtype=float)
    for i, rates in enumerate(parsed):
        if rates is None:
            continue
        vals = np.asarray(rates[:41], dtype=float)
        rate_mat[i, : len(vals)] = vals

    out = pd.DataFrame(index=np.arange(len(df)))
    for c in ["perplexity", "percentage", "entropy_rate", "binoculars_score", "text_length"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").values if c in df.columns else np.nan

    for j in range(41):
        out[f"rates_{j}"] = rate_mat[:, j]

    rate_cols = [f"rates_{j}" for j in range(41)]
    rates_df = out[rate_cols]

    nonzero = rates_df.mask(rates_df == 0)
    out["nz_entropy_rate"] = nonzero.median(axis=1)

    nz_mask = rates_df.ne(0) & rates_df.notna()
    out["nzmean"] = nz_mask.mean(axis=1)
    out["nzstd"] = nz_mask.astype(float).std(axis=1)

    base_nero = ["entropy_rate"] + rate_cols
    out["nzero"] = out[base_nero].eq(0).sum(axis=1)

    out["label"] = int(spec.label)
    out["dataset"] = spec.key
    out["display"] = spec.display
    out["era"] = spec.era
    out["family"] = spec.family
    out["doc_id"] = doc_id.values
    out["has_rates"] = parsed.notna().values
    return out


def load_all(root: Path) -> dict[str, pd.DataFrame]:
    return {spec.key: load_dataset(root, spec) for spec in DATASETS}


def _concat(data: dict[str, pd.DataFrame], keys: Iterable[str]) -> pd.DataFrame:
    return pd.concat([data[k] for k in keys], axis=0, ignore_index=True)


def _resolve_ai_group(group) -> list[str]:
    if isinstance(group, str):
        return list(AI_GROUPS[group])
    return list(group)


def nero_feature_cols() -> list[str]:
    return (
        ["entropy_rate"]
        + [f"rates_{j}" for j in range(41)]
        + ["nz_entropy_rate", "nzmean", "nzstd", "nzero"]
    )


def detector_auc(df: pd.DataFrame, metric: str) -> dict:
    col, sign = SCORE_SPECS[metric]
    use = df[["label", col]].copy()
    use[col] = pd.to_numeric(use[col], errors="coerce")
    use = use.replace([np.inf, -np.inf], np.nan).dropna()

    nh = int((use.label == 0).sum())
    na = int((use.label == 1).sum())
    if nh == 0 or na == 0:
        return {"auc": np.nan, "n_human": nh, "n_ai": na, "n_total": len(use)}

    score = sign * use[col].to_numpy(dtype=float)
    return {
        "auc": float(roc_auc_score(use.label.to_numpy(dtype=int), score)),
        "n_human": nh,
        "n_ai": na,
        "n_total": int(len(use)),
    }


def train_nero_rf(
    df: pd.DataFrame,
    *,
    seed: int,
    test_size: float,
    trees: int,
) -> dict:
    # Match the notebook convention: trained NERO uses only rows that actually
    # contain a rate vector. Do not let the classifier learn missingness.
    df = df.loc[df["has_rates"]].copy()

    cols = nero_feature_cols()
    X = df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y = df["label"].to_numpy(dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    imp = SimpleImputer(strategy="constant", fill_value=0)
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, score)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_human": int((y_test == 0).sum()),
        "n_test_ai": int((y_test == 1).sum()),
    }


def run_within_scenarios(
    data: dict[str, pd.DataFrame],
    *,
    repeats: int = 5,
    test_size: float = 0.5,
    trees: int = 750,
):
    summary_rows = []
    repeat_rows = []

    for hname, hkeys in HUMAN_GROUPS.items():
        human = _concat(data, hkeys)

        for aname, akeys in AI_GROUPS.items():
            ai = _concat(data, akeys)
            frame = pd.concat([human, ai], ignore_index=True)

            for metric in SCORE_SPECS:
                r = detector_auc(frame, metric)
                summary_rows.append({
                    "analysis": "within",
                    "human_group": hname,
                    "ai_group": aname,
                    "detector": metric,
                    **r,
                })

            trained = []
            for seed in range(repeats):
                r = train_nero_rf(
                    frame,
                    seed=seed,
                    test_size=test_size,
                    trees=trees,
                )
                trained.append(r["auc"])
                repeat_rows.append({
                    "analysis": "within",
                    "human_group": hname,
                    "ai_group": aname,
                    "detector": "NERO_RF",
                    "repeat": seed,
                    **r,
                })

            nero_frame = frame.loc[frame["has_rates"]]
            summary_rows.append({
                "analysis": "within",
                "human_group": hname,
                "ai_group": aname,
                "detector": "NERO_RF",
                "auc": float(np.mean(trained)),
                "auc_sd": float(np.std(trained, ddof=1)) if len(trained) > 1 else 0.0,
                "n_human": int((nero_frame.label == 0).sum()),
                "n_ai": int((nero_frame.label == 1).sum()),
                "n_total": int(len(nero_frame)),
            })

    return pd.DataFrame(summary_rows), pd.DataFrame(repeat_rows)


def _split_humans_for_transfer(
    human: pd.DataFrame,
    *,
    seed: int,
):
    # Split within each human source so each source is represented on both sides.
    train_parts, test_parts = [], []
    for _, g in human.groupby("dataset", sort=False):
        if len(g) < 2:
            train_parts.append(g)
            continue
        tr, te = train_test_split(g, test_size=0.5, random_state=seed)
        train_parts.append(tr)
        test_parts.append(te)
    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(test_parts, ignore_index=True),
    )


def train_transfer_rf(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    seed: int,
    trees: int,
) -> dict:
    # Complete-case NERO evaluation: require a real rates vector on both sides.
    train_df = train_df.loc[train_df["has_rates"]].copy()
    test_df = test_df.loc[test_df["has_rates"]].copy()

    cols = nero_feature_cols()
    X_train = train_df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y_train = train_df["label"].to_numpy(dtype=int)
    X_test = test_df[cols].replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
    y_test = test_df["label"].to_numpy(dtype=int)

    imp = SimpleImputer(strategy="constant", fill_value=0)
    X_train = imp.fit_transform(X_train)
    X_test = imp.transform(X_test)

    model = RandomForestClassifier(
        n_estimators=trees,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    score = model.predict_proba(X_test)[:, 1]
    return {
        "auc": float(roc_auc_score(y_test, score)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_test_human": int((y_test == 0).sum()),
        "n_test_ai": int((y_test == 1).sum()),
    }


def run_transfer_scenarios(
    data: dict[str, pd.DataFrame],
    *,
    repeats: int = 5,
    trees: int = 750,
):
    summary_rows = []
    repeat_rows = []

    for hname, hkeys in HUMAN_GROUPS.items():
        human = _concat(data, hkeys)

        for tname, (train_group, test_group) in TRANSFER_GROUPS.items():
            train_ai_keys = _resolve_ai_group(train_group)
            test_ai_keys = _resolve_ai_group(test_group)
            train_ai = _concat(data, train_ai_keys)
            test_ai = _concat(data, test_ai_keys)

            trained_aucs = []
            # Zero-shot/external detectors are evaluated on each repeat's human
            # test half plus the fixed target AI cohort, then averaged.
            per_metric = {m: [] for m in SCORE_SPECS}
            per_metric_ns = {m: [] for m in SCORE_SPECS}

            for seed in range(repeats):
                human_train, human_test = _split_humans_for_transfer(human, seed=seed)
                train_df = pd.concat([human_train, train_ai], ignore_index=True)
                test_df = pd.concat([human_test, test_ai], ignore_index=True)

                r = train_transfer_rf(
                    train_df, test_df,
                    seed=seed,
                    trees=trees,
                )
                trained_aucs.append(r["auc"])
                repeat_rows.append({
                    "analysis": "transfer",
                    "human_group": hname,
                    "transfer": tname,
                    "train_ai_group": str(train_group),
                    "test_ai_group": str(test_group),
                    "detector": "NERO_RF",
                    "repeat": seed,
                    **r,
                })

                for metric in SCORE_SPECS:
                    z = detector_auc(test_df, metric)
                    per_metric[metric].append(z["auc"])
                    per_metric_ns[metric].append(z)

            summary_rows.append({
                "analysis": "transfer",
                "human_group": hname,
                "transfer": tname,
                "detector": "NERO_RF",
                "auc": float(np.nanmean(trained_aucs)),
                "auc_sd": float(np.nanstd(trained_aucs, ddof=1)) if repeats > 1 else 0.0,
            })

            for metric in SCORE_SPECS:
                vals = np.asarray(per_metric[metric], dtype=float)
                valid = vals[np.isfinite(vals)]
                if valid.size:
                    auc_mean = float(np.mean(valid))
                    auc_sd = float(np.std(valid, ddof=1)) if valid.size > 1 else 0.0
                else:
                    auc_mean = np.nan
                    auc_sd = np.nan
                nlast = per_metric_ns[metric][-1]
                summary_rows.append({
                    "analysis": "transfer",
                    "human_group": hname,
                    "transfer": tname,
                    "detector": metric,
                    "auc": auc_mean,
                    "auc_sd": auc_sd,
                    "n_human": nlast["n_human"],
                    "n_ai": nlast["n_ai"],
                    "n_total": nlast["n_total"],
                })

    return pd.DataFrame(summary_rows), pd.DataFrame(repeat_rows)


def dataset_coverage(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for key, df in data.items():
        base = {
            "dataset": key,
            "display": df["display"].iloc[0],
            "label": int(df["label"].iloc[0]),
            "era": df["era"].iloc[0],
            "family": df["family"].iloc[0],
            "n": int(len(df)),
            "rates_available": int(df["has_rates"].sum()),
        }
        for metric, (col, _) in SCORE_SPECS.items():
            vals = pd.to_numeric(df[col], errors="coerce")
            base[f"{metric}_available"] = int(vals.notna().sum())
            base[f"{metric}_zero_fraction"] = (
                float((vals.dropna() == 0).mean()) if vals.notna().any() else np.nan
            )
            base[f"{metric}_median"] = (
                float(vals.median()) if vals.notna().any() else np.nan
            )
        rows.append(base)
    return pd.DataFrame(rows)


def run_all(
    root: Path,
    output_dir: Path,
    *,
    repeats: int,
    test_size: float,
    trees: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_all(root)

    coverage = dataset_coverage(data)
    within, within_rep = run_within_scenarios(
        data,
        repeats=repeats,
        test_size=test_size,
        trees=trees,
    )
    transfer, transfer_rep = run_transfer_scenarios(
        data,
        repeats=repeats,
        trees=trees,
    )

    coverage.to_csv(output_dir / "dataset_coverage.csv", index=False)
    within.to_csv(output_dir / "within_scenario_auc.csv", index=False)
    within_rep.to_csv(output_dir / "within_scenario_trained_repeats.csv", index=False)
    transfer.to_csv(output_dir / "transfer_auc.csv", index=False)
    transfer_rep.to_csv(output_dir / "transfer_trained_repeats.csv", index=False)

    print("Wrote:")
    for name in [
        "dataset_coverage.csv",
        "within_scenario_auc.csv",
        "within_scenario_trained_repeats.csv",
        "transfer_auc.csv",
        "transfer_trained_repeats.csv",
    ]:
        print(" ", output_dir / name)

    return {
        "coverage": coverage,
        "within": within,
        "within_repeats": within_rep,
        "transfer": transfer,
        "transfer_repeats": transfer_rep,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.5)
    ap.add_argument("--trees", type=int, default=750)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    out = args.output_dir
    if out is None:
        out = root / "results" / "detector_comparison"
    elif not out.is_absolute():
        out = root / out

    run_all(
        root,
        out,
        repeats=args.repeats,
        test_size=args.test_size,
        trees=args.trees,
    )


if __name__ == "__main__":
    main()
