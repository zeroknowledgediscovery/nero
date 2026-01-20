import json
import numpy as np
import pandas as pd
from pathlib import Path

def _summarize_entropy(entropy_rate):
    if entropy_rate is None:
        return None, None
    if not isinstance(entropy_rate, (list, tuple, np.ndarray)):
        return None, None

    x = np.asarray(entropy_rate, dtype=float)
    x = x[np.isfinite(x) & (x > 0)]
    if x.size == 0:
        return None, None

    return float(np.median(x)), float(np.mean(x))

def process_json_dir_to_csv(json_dir, csv_file_path):
    processed_data = []
    json_dir = Path(json_dir)

    for json_file in sorted(json_dir.glob("*.json")):
        with open(json_file, 'r', encoding='utf-8') as f:
            try:
                obj = json.load(f)
            except json.JSONDecodeError:
                continue

        # Support either single dict per file, or list of dicts per file
        books = obj if isinstance(obj, list) else [obj]

        for book in books:
            if not isinstance(book, dict):
                continue

            # Prefer nested gutenberg metadata if present
            g = book.get("gutenberg")
            meta = g if isinstance(g, dict) else book

            title = meta.get("title", "Unknown")
            authors_list = meta.get("authors") or []
            author = ", ".join(a.get("name", "") for a in authors_list if isinstance(a, dict) and a.get("name")) or "Unknown"

            book_id = meta.get("id")  # NOTE: this is under gutenberg in your augmented files
            subjects_list = meta.get("subjects") or ["Unknown"]
            subject = " ".join(subjects_list) if isinstance(subjects_list, list) and subjects_list else "Unknown"

            median_entropy_rate, mean_entropy_rate = _summarize_entropy(book.get("entropy_rate"))

            estimated_posterior_probability = book.get("estimated_posterior_probability", None)

            processed_data.append({
                "id": book_id,
                "Title": title,
                "Author": author,
                "Subjects": subject,
                "Median Entropy Rate": median_entropy_rate,
                "Mean Entropy Rate": mean_entropy_rate,
                "Estimated Posterior Probability": estimated_posterior_probability,
                "Source File": json_file.name
            })

    df = pd.DataFrame(processed_data)
    df.to_csv(csv_file_path, index=False)
    print(f"Wrote {len(df)} rows to {csv_file_path}")

# Usage
json_dir = "augmented_jsons/"
csv_file_path = "guten.csv"
process_json_dir_to_csv(json_dir, csv_file_path)
