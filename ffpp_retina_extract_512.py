from glob import glob
import argparse
import os
import cv2
from tqdm import tqdm
import numpy as np
from retinaface.pre_trained_models import get_model
import torch


def pick_largest_face(faces):
    if not faces:
        return None
    valid = []
    for face in faces:
        bbox = face.get('bbox', None) if isinstance(face, dict) else None
        if bbox is None:
            continue
        if len(bbox) < 4:
            continue
        valid.append(face)
    if not valid:
        return None

    def area(face):
        x0, y0, x1, y1 = face['bbox']
        return max(0.0, (x1 - x0)) * max(0.0, (y1 - y0))
    return max(valid, key=area)


def square_crop_with_padding(img_bgr, bbox, scale=2.2):
    h, w = img_bgr.shape[:2]
    x0, y0, x1, y1 = bbox

    cx = 0.5 * (x0 + x1)
    cy = 0.5 * (y0 + y1)
    bw = max(1.0, x1 - x0)
    bh = max(1.0, y1 - y0)
    side = max(bw, bh) * float(scale)

    sx0 = int(round(cx - side / 2.0))
    sy0 = int(round(cy - side / 2.0))
    sx1 = int(round(cx + side / 2.0))
    sy1 = int(round(cy + side / 2.0))

    pad_l = max(0, -sx0)
    pad_t = max(0, -sy0)
    pad_r = max(0, sx1 - w)
    pad_b = max(0, sy1 - h)

    if pad_l or pad_t or pad_r or pad_b:
        img_bgr = cv2.copyMakeBorder(
            img_bgr,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            borderType=cv2.BORDER_REFLECT_101,
        )
        sx0 += pad_l
        sx1 += pad_l
        sy0 += pad_t
        sy1 += pad_t

    crop = img_bgr[sy0:sy1, sx0:sx1]
    return crop


def process_video(model, mp4_path, dataset_root, out_subdir='frames_retina_512', num_frames=20, out_size=512, scale=2.2):
    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return 0, 0

    frame_idxs = np.linspace(0, total - 1, num_frames, endpoint=True, dtype=int)
    frame_idxs = sorted(set(int(x) for x in frame_idxs))

    stem = os.path.basename(mp4_path).replace('.mp4', '')
    out_dir = os.path.join(dataset_root, out_subdir, stem)
    os.makedirs(out_dir, exist_ok=True)

    saved = 0
    detected = 0

    for idx in frame_idxs:
        out_file = os.path.join(out_dir, f'frame_{idx:06d}.png')
        if os.path.exists(out_file):
            continue

        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame_bgr = cap.read()
        if not ok or frame_bgr is None:
            continue

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        faces = model.predict_jsons(frame_rgb)
        face = pick_largest_face(faces)
        if face is None:
            continue

        detected += 1
        x0, y0, x1, y1 = face['bbox']
        crop = square_crop_with_padding(frame_bgr, (x0, y0, x1, y1), scale=scale)
        if crop is None or crop.size == 0:
            continue

        crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_CUBIC)
        if cv2.imwrite(out_file, crop):
            saved += 1

    cap.release()
    return saved, detected


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/mnt/personal_workspace/chenkeyu/ReMark/Dataset-FF++')
    parser.add_argument('--comp', default='c23')
    parser.add_argument('--num_frames', type=int, default=20)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--scale', type=float, default=2.2)
    parser.add_argument('--out_subdir', default='frames_retina_512')
    parser.add_argument('--max_videos', type=int, default=0)
    args = parser.parse_args()

    datasets = [
        os.path.join(args.root, 'original_sequences', 'youtube', args.comp),
        os.path.join(args.root, 'original_sequences', 'actors', args.comp),
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[retina512] device={device}', flush=True)
    model = get_model('resnet50_2020-07-20', max_size=2048, device=device)
    model.eval()

    all_videos = []
    for ds in datasets:
        videos = sorted(glob(os.path.join(ds, 'videos', '*.mp4')))
        print(f'[retina512] {ds} videos={len(videos)}', flush=True)
        all_videos.extend((ds, v) for v in videos)

    if args.max_videos > 0:
        all_videos = all_videos[:args.max_videos]
    print(f'[retina512] total_videos={len(all_videos)}', flush=True)

    total_saved = 0
    total_detected = 0
    for ds, mp4 in tqdm(all_videos, desc='Retina512 crop'):
        s, d = process_video(
            model,
            mp4,
            dataset_root=ds,
            out_subdir=args.out_subdir,
            num_frames=args.num_frames,
            out_size=args.size,
            scale=args.scale,
        )
        total_saved += s
        total_detected += d

    count_files = 0
    for ds in datasets:
        count_files += len(glob(os.path.join(ds, args.out_subdir, '*', '*.png')))

    print(f'[retina512] write_calls_saved={total_saved}', flush=True)
    print(f'[retina512] detected_frames={total_detected}', flush=True)
    print(f'[retina512] actual_unique_png_files={count_files}', flush=True)


if __name__ == '__main__':
    main()
