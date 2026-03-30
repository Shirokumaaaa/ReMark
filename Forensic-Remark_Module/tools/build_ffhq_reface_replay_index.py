#!/usr/bin/env python3
import argparse
import csv
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


def _norm(p: str) -> str:
    return str(Path(p).resolve())


def _read_manifest(csv_path: str):
    rows = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            p = (r.get('img_path') or r.get('image_path') or '').strip()
            if not p:
                continue
            rows.append(_norm(p))
    return rows


def _pick_first_source(csv_path: str) -> str:
    with open(csv_path, 'r', encoding='utf-8') as f:
        rd = csv.DictReader(f)
        for r in rd:
            p = (r.get('img_path') or r.get('image_path') or '').strip()
            if p:
                return _norm(p)
    return ''


def _build_cfg(args):
    fixed_source = (args.fixed_source or '').strip() or _pick_first_source(args.train_csv)
    opts = SimpleNamespace(
        reface_mode='online',
        reface_online_fallback_replay=False,
        reface_allow_missing=False,
        reface_python_bin=args.reface_python,
        reface_repo_root=args.reface_repo,
        reface_online_config=args.reface_config,
        reface_online_ckpt=args.reface_ckpt,
        reface_online_timeout_sec=args.timeout_sec,
        reface_online_ddim_steps=args.ddim_steps,
        reface_online_scale=args.scale,
        reface_online_batch_size=max(int(args.online_batch_size), 1),
        reface_source_csv='',
        reface_fixed_source_path=fixed_source,
        reface_blend_alpha=1.0,
        enforce_nontrivial_swap=False,
        nontrivial_swap_eps=1.0e-4,
    )
    data = SimpleNamespace(train_csv=args.train_csv)
    return SimpleNamespace(attack_options=opts, data=data)


def main():
    p = argparse.ArgumentParser(description='Build FFHQ ReFace replay jsonl by running online generation once per target image.')
    p.add_argument('--train-csv', required=True)
    p.add_argument('--val-csv', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--reface-repo', default='/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace')
    p.add_argument('--reface-python', default='/home/ldy/miniconda3/envs/REFace/bin/python')
    p.add_argument('--reface-config', default='/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/models/REFace/configs/project_ffhq.yaml')
    p.add_argument('--reface-ckpt', default='/mnt/personal_workspace/chenkeyu/ReMark/Attack-REFace/models/REFace/checkpoints/last.ckpt')
    p.add_argument('--fixed-source', default='')
    p.add_argument('--timeout-sec', type=int, default=1200)
    p.add_argument('--ddim-steps', type=int, default=40)
    p.add_argument('--scale', type=float, default=3.5)
    p.add_argument('--online-batch-size', type=int, default=1)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--limit', type=int, default=1000000000)
    p.add_argument('--progress-every', type=int, default=20)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from attacks.reface import ReFaceAttack  # pylint: disable=import-outside-toplevel

    all_paths = []
    seen = set()
    for c in (args.train_csv, args.val_csv):
        for ip in _read_manifest(c):
            if ip in seen:
                continue
            seen.add(ip)
            all_paths.append(ip)

    end = min(len(all_paths), args.start_index + max(int(args.limit), 0))
    subset = all_paths[args.start_index:end]

    os.makedirs(os.path.dirname(args.output_jsonl), exist_ok=True)

    cfg = _build_cfg(args)
    attack = ReFaceAttack(cfg)

    done_map = {}
    if os.path.exists(args.output_jsonl):
        with open(args.output_jsonl, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                src = str(obj.get('source_image', '')).strip()
                outs = obj.get('outputs', [])
                if src and outs:
                    done_map[_norm(src)] = outs[0]

    ok = 0
    fail = 0
    with open(args.output_jsonl, 'a', encoding='utf-8') as wf:
        for idx, img_path in enumerate(subset, start=1):
            if img_path in done_map:
                ok += 1
                if idx % args.progress_every == 0:
                    print(f'[resume] {idx}/{len(subset)} ok={ok} fail={fail}')
                continue
            try:
                attack._generate_online_for_targets([img_path], target_images=None)
                cache_path = attack._cache_path(img_path)
                if not os.path.exists(cache_path):
                    raise RuntimeError(f'cache missing: {cache_path}')
                rec = {
                    'ok': True,
                    'source_image': img_path,
                    'outputs': [cache_path],
                    'replay_meta': {
                        'method': 'reface_online_to_replay',
                    },
                }
                wf.write(json.dumps(rec, ensure_ascii=False) + '\n')
                wf.flush()
                ok += 1
            except Exception as e:
                fail += 1
                print(f'[warn] fail {img_path}: {e}')
            if idx % args.progress_every == 0:
                print(f'[progress] {idx}/{len(subset)} ok={ok} fail={fail}')

    print(f'[done] total={len(subset)} ok={ok} fail={fail} jsonl={args.output_jsonl}')


if __name__ == '__main__':
    main()
