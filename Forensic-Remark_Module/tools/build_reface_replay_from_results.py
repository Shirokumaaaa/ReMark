#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description='Build REFace replay jsonl from result dirs')
    p.add_argument('--results-root', required=True, help='.../Attack-REFace/data/faceswap_outputs/results')
    p.add_argument('--base-jsonl', required=True, help='existing generation_results.jsonl used to map idx->source_image')
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--variants', default='inpaint,ref', help='comma separated: inpaint,ref,GT,mask')
    p.add_argument('--buckets', default='0,1,2,3', help='comma separated folder ids')
    p.add_argument('--max-records', type=int, default=0)
    return p.parse_args()


def load_idx_to_source(base_jsonl: Path):
    mp = {}
    with base_jsonl.open('r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if not rec.get('ok', False):
                continue
            outs = rec.get('outputs', [])
            if not outs:
                continue
            stem = Path(outs[0]).stem
            # expect 12-digit index, e.g. 000000000123
            key = stem[:12]
            mp[key] = rec.get('source_image', '')
    return mp


def main():
    args = parse_args()
    results_root = Path(args.results_root)
    base_jsonl = Path(args.base_jsonl)
    out_jsonl = Path(args.output_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)

    variants = [v.strip() for v in args.variants.split(',') if v.strip()]
    buckets = [b.strip() for b in args.buckets.split(',') if b.strip()]

    idx2src = load_idx_to_source(base_jsonl)

    records = []
    for b in buckets:
        d = results_root / b
        if not d.exists():
            continue
        for v in variants:
            patt = f'*_{v}.png'
            for p in sorted(d.glob(patt)):
                key = p.stem[:12]
                src = idx2src.get(key, '')
                if not src:
                    continue
                records.append((src, p, b, v, key))

    # Keep deterministic order and optional cap
    if args.max_records > 0:
        records = records[:args.max_records]

    with out_jsonl.open('w', encoding='utf-8') as fo:
        for i, (src, p, b, v, key) in enumerate(records):
            rec = {
                'index': i,
                'source_image': str(src),
                'ok': True,
                'outputs': [str(p.resolve())],
                'error': None,
                'replay_meta': {
                    'attack': 'reface',
                    'bucket': b,
                    'variant': v,
                    'pair_index': key,
                },
            }
            fo.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'[done] records={len(records)} output={out_jsonl}')


if __name__ == '__main__':
    main()
