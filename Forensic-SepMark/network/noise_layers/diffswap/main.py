import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


class DiffSwap(nn.Module):
    """
    Lightweight DiffSwap noise layer for SepMark.
    For each sample:
    - target: encoded/watermarked image (keeps background/context)
    - source: random face image from sibling Attack-DiffSwap repo
    - blend by face mask to mimic swap while preserving target background
    """

    def __init__(self, source_root=None):
        super(DiffSwap, self).__init__()

        if source_root is None:
            source_root = (
                Path(__file__).resolve().parents[4]
                / "Attack-DiffSwap"
                / "data"
                / "portrait"
                / "source"
            )
        self.source_root = Path(source_root)
        self.source_images = self._collect_images(self.source_root)

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ])

        if len(self.source_images) == 0:
            print(f"[DiffSwap] No source images found under: {self.source_root}")
        else:
            print(f"[DiffSwap] Loaded {len(self.source_images)} source images from: {self.source_root}")

    @staticmethod
    def _collect_images(root: Path):
        if not root.exists():
            return []
        out = []
        for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp"):
            out.extend(root.rglob(ext))
        return sorted(out)

    def _fallback_face_mask(self, h, w, device):
        yy = torch.linspace(-1.0, 1.0, h, device=device).view(h, 1).expand(h, w)
        xx = torch.linspace(-1.0, 1.0, w, device=device).view(1, w).expand(h, w)
        # Ellipse centered slightly above middle, roughly covering the face area.
        ellipse = ((xx / 0.55) ** 2 + ((yy + 0.1) / 0.75) ** 2) <= 1.0
        m = ellipse.float().unsqueeze(0).unsqueeze(0)
        # Smooth boundary to avoid hard seams.
        m = F.avg_pool2d(m, kernel_size=15, stride=1, padding=7)
        return m.clamp(0.0, 1.0)

    def _build_face_mask(self, mask_item, h, w, device):
        # Expected shapes from SepMark datasets are either [H, W] or [1, H, W].
        if isinstance(mask_item, torch.Tensor) and mask_item.dim() in (2, 3):
            if mask_item.dim() == 2:
                m = mask_item.unsqueeze(0).unsqueeze(0)
            else:
                m = mask_item.unsqueeze(0)
                if m.shape[1] != 1:
                    # Merge all semantic channels to a single face-region mask.
                    m = m.max(dim=1, keepdim=True).values
            m = F.interpolate(m.float(), size=(h, w), mode="nearest")
            if m.max() > 1.0:
                m = (m > 0.5).float()
            else:
                m = (m > 0.2).float()
            # When dataset masks are missing/empty, fallback to an ellipse region.
            if float(m.sum().item()) < 1.0:
                return self._fallback_face_mask(h, w, device)
            m = F.avg_pool2d(m, kernel_size=15, stride=1, padding=7).clamp(0.0, 1.0)
            return m.to(device)
        return self._fallback_face_mask(h, w, device)

    def forward(self, image_cover_mask):
        # target: encoded image with watermark
        image = image_cover_mask[0]
        # mask: face region mask when available
        mask = image_cover_mask[2] if len(image_cover_mask) > 2 else None

        if len(self.source_images) == 0:
            # Graceful fallback keeps original pipeline usable.
            return image

        noised_image = torch.zeros_like(image)
        h, w = image.shape[2], image.shape[3]

        for i in range(image.shape[0]):
            source_path = random.choice(self.source_images)
            source = Image.open(source_path).convert("RGB").resize((w, h), Image.BICUBIC)
            source_tensor = self.transform(source).to(image.device)

            target_tensor = image[i]
            mask_item = mask[i] if isinstance(mask, torch.Tensor) and i < mask.shape[0] else None
            face_mask = self._build_face_mask(mask_item, h, w, image.device).squeeze(0)

            # Keep target background, replace mainly face region with random source.
            blended = target_tensor * (1.0 - face_mask) + source_tensor * face_mask
            noised_image[i] = blended.clamp(-1.0, 1.0)

        return noised_image
