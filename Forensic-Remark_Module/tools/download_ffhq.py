"""
FFHQ 下载脚本（HuggingFace → 原始 1024×1024 PNG）

数据源：Isamu136/ffhq_indexed（HuggingFace，parquet 格式）
策略：逐 shard 下载 → 提取原始 PNG → 存盘 → 删 shard
磁盘峰值：~500MB（单 shard） + 已提取图像（~1.3MB/张）

用法：
    python tools/download_ffhq.py                          # 默认 10k+2k
    python tools/download_ffhq.py --all                    # 下载全部 47k 张
    python tools/download_ffhq.py --n_train 10000 --n_val 2000
"""

import argparse
import io
import random
import sys
from pathlib import Path

import pyarrow.parquet as pq
import requests
from PIL import Image
from tqdm import tqdm
from huggingface_hub import HfApi

# ── 路径 ──────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parents[1]
REPO_ROOT    = ROOT.parent
DEFAULT_OUT  = REPO_ROOT / 'Dataset-FFHQ'
MANIFEST_DIR = ROOT / 'data_manifests'

HF_REPO = 'Isamu136/ffhq_indexed'
HF_BASE = f'https://huggingface.co/datasets/{HF_REPO}/resolve/main'


# ── shard 列表 ────────────────────────────────────────────────────────────────

def get_shard_names() -> list[str]:
    """从 HuggingFace API 获取所有可用 parquet shard 名"""
    print('获取 shard 列表...')
    api = HfApi()
    files = [
        f for f in api.list_repo_files(HF_REPO, repo_type='dataset')
        if f.startswith('data/') and f.endswith('.parquet')
    ]
    files.sort()
    print(f'  共 {len(files)} 个 shard 可用')
    return files


# ── 下载单个 shard ────────────────────────────────────────────────────────────

def download_shard(shard_path: str, tmp_file: Path) -> bool:
    url = f'{HF_BASE}/{shard_path}'
    try:
        r = requests.get(url, stream=True, timeout=60)
        r.raise_for_status()
        total = int(r.headers.get('content-length', 0))
        with open(tmp_file, 'wb') as f, tqdm(
            total=total, unit='B', unit_scale=True,
            desc=f'  {Path(shard_path).name[:45]}', leave=False,
        ) as bar:
            for chunk in r.iter_content(chunk_size=4 << 20):  # 4MB chunks
                f.write(chunk)
                bar.update(len(chunk))
        return True
    except Exception as e:
        print(f'  [ERROR] {e}')
        if tmp_file.exists():
            tmp_file.unlink()
        return False


# ── 提取图像 ──────────────────────────────────────────────────────────────────

def extract_images(shard_file: Path, img_dir: Path, start_idx: int) -> list[str]:
    """从 parquet shard 提取原始 PNG，保存到 img_dir，返回路径列表"""
    pf   = pq.ParquetFile(shard_file)
    paths = []
    for batch in pf.iter_batches(batch_size=32, columns=['image']):
        for img_dict in batch.to_pydict()['image']:
            png_bytes = img_dict['bytes']
            filename  = img_dict.get('path', f'{start_idx + len(paths):05d}.png')
            dest = img_dir / Path(filename).name
            # 直接写原始 bytes，不解压再压缩，速度更快
            dest.write_bytes(png_bytes)
            paths.append(str(dest))
    return paths


# ── manifest ──────────────────────────────────────────────────────────────────

def write_csv(paths: list[str], csv_path: Path):
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('img_path\n')
        f.writelines(p + '\n' for p in paths)
    print(f'  → {csv_path.name}  ({len(paths)} 张)')


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--out_dir',  default=str(DEFAULT_OUT))
    p.add_argument('--n_train',  type=int, default=10000)
    p.add_argument('--n_val',    type=int, default=2000)
    p.add_argument('--all',      action='store_true', help='下载全部可用图像')
    p.add_argument('--seed',     type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / 'images'
    img_dir.mkdir(parents=True, exist_ok=True)
    tmp_shard = out_dir / '_tmp.parquet'

    n_needed = 999_999 if args.all else (args.n_train + args.n_val)

    shards = get_shard_names()
    all_paths: list[str] = []

    for i, shard in enumerate(shards):
        if not args.all and len(all_paths) >= n_needed:
            break

        print(f'\n[shard {i+1}/{len(shards)}  已收集 {len(all_paths)}/{n_needed}]')
        if not download_shard(shard, tmp_shard):
            continue

        new = extract_images(tmp_shard, img_dir, start_idx=len(all_paths))
        all_paths.extend(new)
        tmp_shard.unlink(missing_ok=True)
        print(f'  提取 {len(new)} 张  累计 {len(all_paths)}')

    if len(all_paths) < args.n_train + args.n_val:
        print(f'WARNING: 仅 {len(all_paths)} 张，不足 {args.n_train+args.n_val}')

    # 打乱 & 切分
    rng = random.Random(args.seed)
    rng.shuffle(all_paths)
    train = all_paths[:args.n_train]
    val   = all_paths[args.n_train:args.n_train + args.n_val]
    rest  = all_paths[args.n_train + args.n_val:]

    print('\n写入 manifests ...')
    write_csv(train, MANIFEST_DIR / 'ffhq_train.csv')
    write_csv(val,   MANIFEST_DIR / 'ffhq_val.csv')
    if rest:
        write_csv(rest, MANIFEST_DIR / 'ffhq_rest.csv')

    print('\n完成！stage1_vae.yaml 中更新：')
    print('  train_csv: data_manifests/ffhq_train.csv')
    print('  val_csv:   data_manifests/ffhq_val.csv')


if __name__ == '__main__':
    main()
