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
            ap = _norm(p)
            if os.path.exists(ap):
                rows.append(ap)
    return rows


def _build_cfg(args):
    opts = SimpleNamespace(
        face_adapter_mode='online',
        face_adapter_online_fallback_replay=False,
        face_adapter_allow_missing=False,
        face_adapter_python_bin=args.face_python,
        face_adapter_repo_root=args.face_repo,
        face_adapter_online_cache_dir=args.cache_dir,
        face_adapter_online_hf_cache=args.hf_cache,
        face_adapter_online_local_files_only=True,
        face_adapter_online_timeout_sec=args.timeout_sec,
        face_adapter_online_num_inference_steps=args.num_steps,
        face_adapter_online_guidance_scale=args.guidance_scale,
        face_adapter_online_crop_ratio=args.crop_ratio,
        face_adapter_source_csv=args.source_csv,
        face_adapter_fixed_source_path=args.fixed_source if args.fixed_source else '',
        face_adapter_blend_alpha=1.0,
        enforce_nontrivial_swap=False,
        nontrivial_swap_eps=1.0e-4,
    )
    data = SimpleNamespace(train_csv=args.source_csv)
    return SimpleNamespace(attack_options=opts, data=data)


def main():
    p = argparse.ArgumentParser(description='Build FFHQ Face-Adapter replay jsonl by running online generation once per target image.')
    p.add_argument('--train-csv', required=True)
    p.add_argument('--val-csv', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--face-repo', default='/mnt/personal_workspace/chenkeyu/ReMark/Attack-Face-Adapter')
    p.add_argument('--face-python', default='/home/ldy/miniconda3/envs/FaceAdapter/bin/python')
    p.add_argument('--cache-dir', default='/tmp/remark_face_adapter_online_cache')
    p.add_argument('--hf-cache', default='/home/ldy/.cache/huggingface/hub')
    p.add_argument('--source-csv', default='')
    p.add_argument('--fixed-source', default='')
    p.add_argument('--timeout-sec', type=int, default=600)
    p.add_argument('--num-steps', type=int, default=12)
    p.add_argument('--guidance-scale', type=float, default=3.5)
    p.add_argument('--crop-ratio', type=float, default=0.81)
    p.add_argument('--start-index', type=int, default=0)
    p.add_argument('--limit', type=int, default=1000000000)
    p.add_argument('--progress-every', type=int, default=20)
    args = p.parse_args()

    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from attacks.face_adapter import FaceAdapterAttack  # pylint: disable=import-outside-toplevel

    source_csv = args.source_csv.strip() or args.train_csv
    args.source_csv = source_csv

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
    os.makedirs(args.cache_dir, exist_ok=True)

    cfg = _build_cfg(args)
    attack = FaceAdapterAttack(cfg)

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
                        'method': 'face_adapter_online_to_replay',
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
