#!/usr/bin/env python3
import argparse
import csv
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Build img_path manifest from replay jsonl")
    p.add_argument("--input-jsonl", required=True)
    p.add_argument("--output-csv", required=True)
    p.add_argument("--source-key", default="source_image")
    p.add_argument("--require-ok", action="store_true", default=True)
    p.add_argument("--no-require-ok", action="store_true", default=False)
    return p.parse_args()


def main():
    args = parse_args()
    if args.no_require_ok:
        args.require_ok = False

    in_path = Path(args.input_jsonl)
    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    uniq = []
    seen = set()
    with in_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if args.require_ok and (not rec.get("ok", False)):
                continue
            src = rec.get(args.source_key, None)
            if not src:
                continue
            if src in seen:
                continue
            seen.add(src)
            uniq.append(src)

    with out_path.open("w", encoding="utf-8", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["img_path"])
        for p in uniq:
            wr.writerow([p])

    print(f"[done] input={in_path} output={out_path} count={len(uniq)}")


if __name__ == "__main__":
    main()
