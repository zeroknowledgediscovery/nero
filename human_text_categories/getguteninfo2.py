import requests
import json
import time
import random
from pathlib import Path

def read_indexes(file_path):
    with open(file_path) as f:
        return [line.strip() for line in f if line.strip().isdigit()]

def save_json(data, filename):
    with open(filename, "w") as f:
        json.dump(data, f, indent=4)

def fetch_one_with_backoff(session, book_id, max_retries=8, base_delay=0.5):
    url = f"https://gutendex.com/books/{book_id}/"

    for attempt in range(max_retries + 1):
        r = session.get(url, timeout=30)

        if r.status_code == 200:
            return r.json()

        if r.status_code == 404:
            raise FileNotFoundError(f"404 Not Found for id={book_id}")

        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            if retry_after is not None:
                sleep_s = float(retry_after)
            else:
                sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
            continue

        if r.status_code in (500, 502, 503, 504):
            sleep_s = base_delay * (2 ** attempt) + random.uniform(0, 0.25)
            time.sleep(sleep_s)
            continue

        r.raise_for_status()

    raise RuntimeError(f"Exceeded retries for id={book_id} (last status={r.status_code})")

def main(file_path, out_dir="."):
    ids = read_indexes(file_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Total IDs: {len(ids)}")

    with requests.Session() as session:
        session.headers.update({"User-Agent": "gutendex-crawler/1.0"})

        for i, book_id in enumerate(ids, start=1):
            outfile = out_dir / f"gutenberg_{book_id}.json"

            # ---- SKIP IF ALREADY PRESENT ----
            if outfile.exists():
                continue

            try:
                data = fetch_one_with_backoff(session, book_id)
                save_json(data, outfile)
            except FileNotFoundError:
                # optional: create a marker file so we don't retry missing IDs
                (out_dir / f"gutenberg_{book_id}.404").touch()
            except Exception as e:
                print(f"[FAIL] id={book_id}: {e}")

            time.sleep(0.25)  # baseline rate limit

            if i % 50 == 0:
                print(f"Processed {i}/{len(ids)}")

if __name__ == "__main__":
    main("indexes_.dat", out_dir="Gutenberg_json")
