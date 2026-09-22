#!/usr/bin/env python3
"""Repair missing NERO rate vectors used by paper_results005.

Recovery order:
1. Use an existing sibling entropy_rates JSON (exact stem match).
2. For LOC public-domain files, use a unique truncated-prefix JSON match.
3. If the source text is committed, rerun bin/nero on the first 80,000
   symbols, matching the process_length used by the existing JSON outputs.
4. Leave genuinely unrecoverable rows missing; paper_results005 will report
   and exclude only those rows.

The original source CSVs are never modified. Repaired copies are written to
paper_data/repaired/.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    csv: str
    entropy_dir: str
    output: str
    source_dir: str | None = None
    source_name: Callable[[str], str] | None = None
    allow_prefix_json: bool = False


def _identity(name: str) -> str:
    return name


def _claude_source(name: str) -> str:
    m = re.fullmatch(r"test(\d+)\.txt", name)
    return f"claude_sonnet{m.group(1)}.txt" if m else name


SPECS = [
    DatasetSpec(
        "gpt4o",
        "api_data_collection/ai_data/openai/gpt4o/gpt_4o_detection.csv",
        "api_data_collection/ai_data/openai/gpt4o/entropy_rates",
        "paper_data/repaired/gpt4o.csv",
        "api_data_collection/generated_txt/api_gpt4o",
        _identity,
    ),
    DatasetSpec(
        "gpt4.0",
        "api_data_collection/ai_data/openai/gpt4.0/gpt_4zero_detection.csv",
        "api_data_collection/ai_data/openai/gpt4.0/entropy_rates",
        "paper_data/repaired/gpt4.0.csv",
        "api_data_collection/generated_txt/api_gpt4.0",
        _identity,
    ),
    DatasetSpec(
        "loc-pd",
        "api_data_collection/ai_data/loc-pd/loc-pd_detection.csv",
        "api_data_collection/ai_data/loc-pd/entropy_rates",
        "paper_data/repaired/loc-pd.csv",
        allow_prefix_json=True,
    ),
    DatasetSpec(
        "us-pd",
        "api_data_collection/ai_data/us-pd/us-pd_detection.csv",
        "api_data_collection/ai_data/us-pd/entropy_rates",
        "paper_data/repaired/us-pd.csv",
    ),
    DatasetSpec(
        "gpt5",
        "api_data_collection/ai_data/openai/gpt5/gpt_5_detection.csv",
        "api_data_collection/ai_data/openai/gpt5/entropy_rates",
        "paper_data/repaired/gpt5.csv",
        "api_data_collection/generated_txt/api_gpt5.0",
        _identity,
    ),
    DatasetSpec(
        "claude",
        "api_data_collection/ai_data/claude_sonnet_4/claude_detection.csv",
        "api_data_collection/ai_data/claude_sonnet_4/entropy_rates",
        "paper_data/repaired/claude.csv",
        "api_data_collection/generated_txt/api_claude_sonnet_4.0",
        _claude_source,
    ),
    DatasetSpec(
        "gemini",
        "api_data_collection/ai_data/gemini_2.5_pro/gemini_25_detection.csv",
        "api_data_collection/ai_data/gemini_2.5_pro/entropy_rates",
        "paper_data/repaired/gemini.csv",
        "api_data_collection/generated_txt/api_gemini_2.5_pro",
        _identity,
    ),
]


def _missing_rate(value) -> bool:
    return pd.isna(value) or not str(value).strip()


def _load_nero_json(path: Path):
    try:
        data = json.loads(path.read_text())
    except Exception:
        return None
    rates = data.get("entropy_rate")
    scalar = data.get("average_entropy_rate")
    if not isinstance(rates, list) or scalar is None:
        return None
    return float(scalar), rates


def _json_for_row(row_name: str, entropy_dir: Path, allow_prefix: bool) -> Path | None:
    stem = Path(row_name).stem
    exact = entropy_dir / f"{stem}.json"
    if exact.exists():
        return exact
    if not allow_prefix or not entropy_dir.exists():
        return None

    matches = []
    for p in entropy_dir.glob("*.json"):
        s = p.stem
        if stem.startswith(s) or s.startswith(stem):
            matches.append(p)
    return matches[0] if len(matches) == 1 else None


def _run_nero(
    *,
    nero: Path,
    source: Path,
    output_json: Path,
    process_length: int,
    timeout: int,
):
    output_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(nero),
        "-f", str(source),
        "-x", str(process_length),
        "-o", str(output_json),
    ]
    proc = subprocess.run(
        cmd,
        cwd=nero.parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"NERO failed for {source} (exit {proc.returncode}):\n{proc.stdout[-4000:]}"
        )
    parsed = _load_nero_json(output_json)
    if parsed is None:
        raise RuntimeError(f"NERO did not produce a valid JSON result for {source}")
    return parsed


def repair_dataset(
    root: Path,
    spec: DatasetSpec,
    *,
    nero: Path,
    process_length: int,
    timeout: int,
    workers: int,
):
    src_csv = root / spec.csv
    entropy_dir = root / spec.entropy_dir
    out_csv = root / spec.output
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(src_csv)
    if "rates" not in df.columns or "entropy_rate" not in df.columns:
        raise ValueError(f"{src_csv} does not have entropy_rate/rates columns")

    id_col = df.columns[0]
    missing_idx = [i for i, v in df["rates"].items() if _missing_rate(v)]

    summary = {
        "dataset": spec.name,
        "rows": int(len(df)),
        "missing_initial": int(len(missing_idx)),
        "recovered_json": 0,
        "recovered_nero": 0,
        "unresolved": [],
    }

    # First use already committed NERO outputs.
    remaining = []
    for i in missing_idx:
        row_name = str(df.at[i, id_col])
        p = _json_for_row(row_name, entropy_dir, spec.allow_prefix_json)
        parsed = _load_nero_json(p) if p else None
        if parsed is None:
            remaining.append(i)
            continue
        scalar, rates = parsed
        df.at[i, "entropy_rate"] = scalar
        df.at[i, "rates"] = repr(rates)
        summary["recovered_json"] += 1

    # Then regenerate when the exact source text exists.
    tasks = {}
    if spec.source_dir and spec.source_name:
        source_dir = root / spec.source_dir
        generated_dir = root / "paper_data/repaired/generated_entropy" / spec.name
        for i in remaining:
            row_name = str(df.at[i, id_col])
            source_name = spec.source_name(row_name)
            source = source_dir / source_name
            if source.exists():
                output_json = generated_dir / f"row_{i}.json"
                tasks[i] = (row_name, source, output_json)

    recovered_by_i = {}
    if tasks:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            future_map = {
                ex.submit(
                    _run_nero,
                    nero=nero,
                    source=source,
                    output_json=output_json,
                    process_length=process_length,
                    timeout=timeout,
                ): i
                for i, (_, source, output_json) in tasks.items()
            }
            for fut in as_completed(future_map):
                i = future_map[fut]
                try:
                    recovered_by_i[i] = fut.result()
                except Exception as e:
                    print(f"[repair] {spec.name} row {i}: {e}")

    for i, parsed in recovered_by_i.items():
        scalar, rates = parsed
        df.at[i, "entropy_rate"] = scalar
        df.at[i, "rates"] = repr(rates)
        summary["recovered_nero"] += 1

    for i in missing_idx:
        if _missing_rate(df.at[i, "rates"]):
            summary["unresolved"].append(str(df.at[i, id_col]))

    df.to_csv(out_csv, index=False)
    summary["missing_final"] = len(summary["unresolved"])
    print(
        f"[repair] {spec.name}: rows={summary['rows']} "
        f"missing={summary['missing_initial']} "
        f"json={summary['recovered_json']} "
        f"nero={summary['recovered_nero']} "
        f"unresolved={summary['missing_final']}"
    )
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--process-length", type=int, default=80000)
    ap.add_argument("--nero-timeout", type=int, default=900)
    args = ap.parse_args()

    root = args.repo_root.resolve()
    nero = root / "bin" / "nero"
    if not nero.exists():
        raise FileNotFoundError(nero)
    nero.chmod(nero.stat().st_mode | 0o111)

    summaries = []
    for spec in SPECS:
        summaries.append(
            repair_dataset(
                root,
                spec,
                nero=nero,
                process_length=args.process_length,
                timeout=args.nero_timeout,
                workers=args.workers,
            )
        )

    out = root / "paper_data" / "repaired" / "repair_summary.json"
    out.write_text(json.dumps(summaries, indent=2) + "\n")
    print(f"[repair] summary: {out}")


if __name__ == "__main__":
    main()
