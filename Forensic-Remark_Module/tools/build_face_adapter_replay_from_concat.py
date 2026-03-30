#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser(description='Build Face-Adapter replay jsonl from concat images')
    p.add_argument('--concat-dir', required=True)
    p.add_argument('--output-jsonl', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--source-root', required=True, help='e.g. /.../Dataset-CelebA_HQ/test')
    p.add_argument('--max-records', type=int, default=0)
    p.add_argument('--jpeg-quality', type=int, default=95)
    p.add_argument('--save-format', choices=['jpg', 'png'], default='jpg')
    return p.parse_args()


def parse_pair(stem: str):
    if '_' not in stem:
        return None, None
    a, b = stem.split('_', 1)
    return a.strip(), b.strip()


def _infer_grid_layout(w: int, h: int, nrow: int = 4):
    """Infer torchvision.make_grid layout (padding + cell size)."""
    # prefer common paddings used by make_grid
    for pad in (2, 0, 1, 4, 8):
        inner_w = w - pad * (nrow + 1)
        inner_h = h - pad * 2
        if inner_w <= 0 or inner_h <= 0:
            continue
        if inner_w % nrow != 0:
            continue
        cell_w = inner_w // nrow
        cell_h = inner_h
        # Face-Adapter concat should be square tiles; tolerate tiny mismatch
        if abs(cell_w - cell_h) <= 2:
            return pad, cell_w, cell_h
    # Fallback: equal-width split without padding
    return 0, w // nrow, h


def _crop_grid_cell(im: Image.Image, idx: int, nrow: int = 4) -> Image.Image:
    w, h = im.size
    pad, cell_w, cell_h = _infer_grid_layout(w, h, nrow=nrow)
    x0 = pad + idx * (cell_w + pad)
    y0 = pad
    x1 = x0 + cell_w
    y1 = y0 + cell_h
    return im.crop((x0, y0, x1, y1))


def main():
    args = parse_args()
    concat_dir = Path(args.concat_dir)
    out_jsonl = Path(args.output_jsonl)
    out_dir = Path(args.output_dir)
    source_root = Path(args.source_root)

    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted([p for p in concat_dir.glob('*.jpg')]) + sorted([p for p in concat_dir.glob('*.png')])
    total = ok = 0

    with out_jsonl.open('w', encoding='utf-8') as fo:
        for p in files:
            if args.max_records > 0 and ok >= args.max_records:
                break
            total += 1
            src_id, tgt_id = parse_pair(p.stem)
            if not src_id:
                continue
            src_img = source_root / f'{src_id}.jpg'
            if not src_img.exists():
                continue

            try:
                im = Image.open(p).convert('RGB')
                # concat columns: [source, target, reenact, swap]
                swap = _crop_grid_cell(im, idx=3, nrow=4)

                ext = 'png' if args.save_format == 'png' else 'jpg'
                out_img = out_dir / f'{src_id}_{tgt_id}_swap.{ext}'
                if ext == 'jpg':
                    swap.save(out_img, quality=int(args.jpeg_quality))
                else:
                    swap.save(out_img)

                rec = {
                    'index': ok,
                    'source_image': str(src_img.resolve()),
                    'ok': True,
                    'outputs': [str(out_img.resolve())],
                    'error': None,
                    'replay_meta': {
                        'attack': 'face_adapter',
                        'from': 'concat',
                        'pair': f'{src_id}_{tgt_id}',
                    },
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')
                ok += 1
            except Exception as e:  # noqa: BLE001
                rec = {
                    'index': ok,
                    'source_image': str(src_img.resolve()),
                    'ok': False,
                    'outputs': [],
                    'error': str(e),
                }
                fo.write(json.dumps(rec, ensure_ascii=False) + '\n')

    print(f'[done] concat={concat_dir} total_files={total} written={ok} output={out_jsonl}')


if __name__ == '__main__':
    main()
