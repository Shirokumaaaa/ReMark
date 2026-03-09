import os
import random
import sys
from pathlib import Path

import cv2
import dlib
import numpy as np
import torch
import torch.nn as nn
from scipy.spatial import ConvexHull


class _AttackDiffSwapEngine:
    def __init__(self, attack_root: str, checkpoint: str, config: str, tgt_scale: float, ddim_steps: int, device: torch.device):
        self.attack_root = Path(attack_root).resolve()
        self.device = device
        self.ddim_steps = int(ddim_steps)
        self.tgt_scale = float(tgt_scale)

        if str(self.attack_root) not in sys.path:
            sys.path.insert(0, str(self.attack_root))
        pyver = f"python{sys.version_info.major}.{sys.version_info.minor}"
        env_site = Path(sys.prefix) / "lib" / pyver / "site-packages"
        if env_site.exists():
            env_site_str = str(env_site)
            if env_site_str in sys.path:
                sys.path.remove(env_site_str)
            sys.path.insert(0, env_site_str)

        from omegaconf import OmegaConf
        from ldm.util import instantiate_from_config
        from ldm.models.diffusion.ddim import DDIMSampler
        from ldm.data.portrait import Portrait
        from mtcnn import MTCNN
        from data_preprocessing.align.align_trans import get_reference_facial_points, warp_and_crop_face
        self._warp_and_crop_face = warp_and_crop_face

        config_path = Path(config)
        if not config_path.is_absolute():
            config_path = self.attack_root / config_path
        ckpt_path = Path(checkpoint)
        if not ckpt_path.is_absolute():
            ckpt_path = self.attack_root / ckpt_path

        old_cwd = os.getcwd()
        try:
            os.chdir(str(self.attack_root))
            cfg = OmegaConf.load(str(config_path))
            self.model = instantiate_from_config(cfg.model)
            self.model.init_from_ckpt(str(ckpt_path))
            self.model = self.model.to(self.device).eval()
            self.model.cond_stage_model.affine_crop = True
            self.model.cond_stage_model.swap = True
            self.sampler = DDIMSampler(self.model, target_preserve_scale=self.tgt_scale)
            self.portrait = Portrait(str(self.attack_root / "data" / "portrait"))
        finally:
            os.chdir(old_cwd)

        self.src_n = len(self.portrait.src_list)
        self.tgt_n = len(self.portrait.tgt_list)
        self.mtcnn = MTCNN()
        self.dlib_detector = dlib.get_frontal_face_detector()
        self.dlib_predictor = dlib.shape_predictor(str(self.attack_root / "checkpoints/shape_predictor_68_face_landmarks.dat"))
        self.ref5 = get_reference_facial_points(default_square=True) * (112.0 / 112.0)
        self.organ_indices = {
            "l_eye": list(range(36, 42)),
            "r_eye": list(range(42, 48)),
            "nose": list(range(27, 36)),
            "mouth": list(range(48, 68)),
        }

    @staticmethod
    def _to_tensor_batch(sample, device):
        out = {}
        for k, v in sample.items():
            if isinstance(v, np.ndarray):
                t = torch.from_numpy(v)
                if t.dtype == torch.bool:
                    t = t.float()
                out[k] = t.unsqueeze(0).to(device)
            elif isinstance(v, torch.Tensor):
                out[k] = v.unsqueeze(0).to(device)
            else:
                out[k] = [v]
        return out

    @torch.no_grad()
    def _perform_swap(self, batch):
        z, c, x, _, _ = self.model.get_input(
            batch,
            self.model.first_stage_key,
            return_first_stage_outputs=True,
            force_c_encode=True,
            return_original_cond=True,
            swap=True,
        )
        n = x.size(0)
        h, w = z.shape[2], z.shape[3]
        mask = (1 - batch["mask"].float())[:, None]
        mask = torch.nn.functional.interpolate(mask, size=(h, w), mode="nearest")
        mask[mask > 0] = 1
        mask[mask <= 0] = 0
        shape = (self.model.channels, self.model.image_size, self.model.image_size)
        with self.model.ema_scope("Plotting Inpaint"):
            samples, _ = self.sampler.sample(
                self.ddim_steps, n, shape, c, eta=0.0, x0=z[:n], mask=mask, verbose=False
            )
        out = self.model.decode_first_stage(samples.to(self.model.device))
        return torch.clamp(out, -1.0, 1.0)

    @torch.no_grad()
    def swap_one(self, encoded_chw: torch.Tensor) -> torch.Tensor:
        src_idx = random.randrange(self.src_n)
        tgt_idx = random.randrange(self.tgt_n)
        sample = self.portrait[src_idx * self.tgt_n + tgt_idx]
        encoded_hwc = encoded_chw.detach().permute(1, 2, 0).cpu().numpy().astype(np.float32)
        encoded_u8 = np.clip((encoded_hwc + 1.0) * 127.5, 0, 255).astype(np.uint8)
        geom = self._build_target_geometry(encoded_u8)
        if geom is None:
            # Keep compatibility when detection fails on edge cases.
            sample["image"] = encoded_hwc
        else:
            sample["image"] = encoded_hwc
            sample["landmark"] = geom["landmark"]
            sample["mask"] = geom["mask"]
            sample["mask_organ"] = geom["mask_organ"]
            sample["affine_theta"] = geom["affine_theta"]
            sample["target"] = "__encoded__.png"
        batch = self._to_tensor_batch(sample, self.device)
        attacked = self._perform_swap(batch)[0]
        return attacked

    def _build_target_geometry(self, image_u8: np.ndarray):
        lm = self._detect_landmark68(image_u8)
        theta = self._compute_affine_theta_from_mtcnn(image_u8)
        if lm is None or theta is None:
            return None
        landmark = (lm.astype(np.float32) / 256.0).astype(np.float32)
        mask, mask_organ = self._build_masks_from_landmark(landmark)
        return {
            "landmark": landmark,
            "mask": mask,
            "mask_organ": mask_organ,
            "affine_theta": theta.astype(np.float32),
        }

    def _detect_landmark68(self, image_u8: np.ndarray):
        gray = cv2.cvtColor(image_u8, cv2.COLOR_RGB2GRAY)
        faces = self.dlib_detector(gray, 1)
        if len(faces) == 0:
            return None
        face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        shape = self.dlib_predictor(image_u8, face)
        pts = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            pts[i, 0] = shape.part(i).x
            pts[i, 1] = shape.part(i).y
        return pts

    def _compute_affine_theta_from_mtcnn(self, image_u8: np.ndarray):
        def compute_area(item):
            return -float(item["box"][2] * item["box"][3])

        keys = ["left_eye", "right_eye", "nose", "mouth_left", "mouth_right"]
        det = self.mtcnn.detect_faces(image_u8)
        if len(det) == 0:
            return None
        item = sorted(det, key=compute_area)[0]
        facial5points = [item["keypoints"][k] for k in keys]
        tfm = self._warp_and_crop_face(None, facial5points, self.ref5, crop_size=(112, 112), return_tfm=True)
        return self._tfm2theta(tfm).astype(np.float32)

    @staticmethod
    def _tfm2theta(tfm):
        h1 = w1 = 256
        h2 = w2 = 112
        a = np.array([[2 / (w1 - 1), 0, -1], [0, 2 / (h1 - 1), -1], [0, 0, 1]], dtype=np.float32)
        b = np.linalg.inv(np.array([[2 / (w2 - 1), 0, -1], [0, 2 / (h2 - 1), -1], [0, 0, 1]], dtype=np.float32))
        c = np.array([[0, 0, 1]], dtype=np.float32)
        ttt = np.concatenate([tfm, c], axis=0)
        ttt = np.linalg.inv(ttt)
        theta = a @ ttt @ b
        return theta[:2]

    def _build_masks_from_landmark(self, landmark_norm: np.ndarray):
        mask = self._extract_convex_hull(landmark_norm)
        organ_masks = []
        for key in ["l_eye", "r_eye", "nose", "mouth"]:
            organ_masks.append(self._extract_convex_hull(landmark_norm[self.organ_indices[key]]))
        return mask, np.stack(organ_masks, axis=0)

    @staticmethod
    def _extract_convex_hull(landmark_norm: np.ndarray, size: int = 256):
        points = (landmark_norm * float(size)).astype(np.float32)
        hull = ConvexHull(points)
        image = np.zeros((size, size), dtype=np.uint8)
        poly = np.concatenate([points[hull.vertices, :1], points[hull.vertices, 1:]], axis=-1).astype(np.int32)
        mask = cv2.fillPoly(image, pts=[poly], color=255)
        return (mask > 0)


_ENGINE = None
_ENGINE_KEY = None


class AttackDiffSwap(nn.Module):
    def __init__(self, attack_root=None, checkpoint=None, config=None, tgt_scale=None, ddim_steps=None):
        super(AttackDiffSwap, self).__init__()
        default_attack_root = Path(__file__).resolve().parents[4] / "Attack-DiffSwap"
        attack_root = attack_root or os.environ.get(
            "SEPMARK_ATTACK_DIFFSWAP_ROOT", str(default_attack_root)
        )
        checkpoint = checkpoint or os.environ.get("SEPMARK_ATTACK_DIFFSWAP_CKPT", "checkpoints/diffswap.pth")
        config = config or os.environ.get("SEPMARK_ATTACK_DIFFSWAP_CONFIG", "configs/diffswap/default-project.yaml")
        tgt_scale = float(tgt_scale if tgt_scale is not None else os.environ.get("SEPMARK_ATTACK_DIFFSWAP_TGT_SCALE", "0.01"))
        ddim_steps = int(ddim_steps if ddim_steps is not None else os.environ.get("SEPMARK_ATTACK_DIFFSWAP_STEPS", "200"))

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_key = (str(Path(attack_root).resolve()), str(checkpoint), str(config), float(tgt_scale), int(ddim_steps), str(self.device))
        global _ENGINE, _ENGINE_KEY
        if _ENGINE is None or _ENGINE_KEY != self._init_key:
            _ENGINE = _AttackDiffSwapEngine(
                attack_root=attack_root,
                checkpoint=checkpoint,
                config=config,
                tgt_scale=tgt_scale,
                ddim_steps=ddim_steps,
                device=self.device,
            )
            _ENGINE_KEY = self._init_key
        self.engine = _ENGINE

    @torch.no_grad()
    def forward(self, image_cover_mask):
        image = image_cover_mask[0]
        out = torch.zeros_like(image)
        for i in range(image.shape[0]):
            out[i] = self.engine.swap_one(image[i].to(self.device)).to(image.device)
        return out
