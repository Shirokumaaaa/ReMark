import hashlib
import os
from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence, Tuple

import dlib
import numpy as np
import torch
from PIL import Image


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DEFAULT_PREDICTOR = os.path.join(
    _REPO_ROOT,
    "Attack-DiffSwap",
    "checkpoints",
    "shape_predictor_68_face_landmarks.dat",
)


def _normalize_path(path: str) -> str:
    return os.path.abspath(os.path.expanduser(str(path)))


def _sha1_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _ensure_parent(path: str):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)


def _safe_load_npy(path: str, expected_shape: Optional[Tuple[int, ...]] = None) -> np.ndarray:
    arr = np.load(path)
    if expected_shape is not None and tuple(arr.shape) != tuple(expected_shape):
        raise ValueError(f"Unexpected npy shape at {path}: got {tuple(arr.shape)}, expected {expected_shape}")
    if arr.size == 0:
        raise ValueError(f"Empty npy array at {path}")
    return arr


def _safe_cache_load(path: str, expected_shape: Optional[Tuple[int, ...]] = None) -> Optional[np.ndarray]:
    if not os.path.isfile(path):
        return None
    try:
        return _safe_load_npy(path, expected_shape=expected_shape)
    except Exception:
        try:
            os.remove(path)
        except OSError:
            pass
        return None


def _pack_uint_bits(values: np.ndarray, bits_per_value: int) -> np.ndarray:
    """
    values: uint array in [0, 2^bits_per_value - 1], shape [N]
    returns: float32 {0,1} bits, shape [N * bits_per_value]
    """
    values = values.astype(np.uint16, copy=False).reshape(-1)
    out = np.empty((values.shape[0], bits_per_value), dtype=np.float32)
    for i in range(bits_per_value):
        shift = bits_per_value - 1 - i
        out[:, i] = ((values >> shift) & 1).astype(np.float32)
    return out.reshape(-1)


def _resample_1d(values: np.ndarray, num_values: int) -> np.ndarray:
    if values.ndim != 1:
        raise ValueError(f"Expected 1D values, got shape={values.shape}")
    if num_values <= 0:
        return np.zeros((0,), dtype=np.float32)
    if values.shape[0] == num_values:
        return values.astype(np.float32, copy=False)
    x_src = np.linspace(0.0, 1.0, num=values.shape[0], dtype=np.float32)
    x_dst = np.linspace(0.0, 1.0, num=num_values, dtype=np.float32)
    return np.interp(x_dst, x_src, values).astype(np.float32, copy=False)


def _encode_landmarks_to_bits(
    landmarks_xy: np.ndarray,
    num_bits: int,
    bits_per_value: int = 4,
) -> np.ndarray:
    """
    landmarks_xy: [68, 2], normalized to [0,1]
    num_bits: target bit length (supports multiples of bits_per_value)

    Encoding rule:
      1) flatten 68x2 normalized coordinates -> 136 scalars
      2) linearly resample to num_bits / bits_per_value scalars
      3) quantize each scalar to 4-bit unsigned integer
      4) pack to binary bits

    This keeps all methods on one canonical landmark descriptor family while
    still allowing LampMark to consume the prefix 64 bits natively.
    """
    if num_bits <= 0:
        return np.zeros((0,), dtype=np.float32)
    if num_bits % bits_per_value != 0:
        raise ValueError(
            f"num_bits must be divisible by bits_per_value={bits_per_value}, got {num_bits}"
        )
    flat = landmarks_xy.astype(np.float32, copy=False).reshape(-1)
    flat = np.clip(flat, 0.0, 1.0)
    num_values = num_bits // bits_per_value
    desc = _resample_1d(flat, num_values)
    levels = (1 << bits_per_value) - 1
    quantized = np.rint(desc * levels).astype(np.uint16)
    return _pack_uint_bits(quantized, bits_per_value=bits_per_value)


def _tensor_image_to_rgb_u8(image: torch.Tensor) -> np.ndarray:
    """
    image: [3,H,W], either in [-1,1] or [0,1]
    """
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"Expected image shape [3,H,W], got {tuple(image.shape)}")
    x = image.detach().float().cpu()
    if x.min().item() < -0.01:
        x = (x.clamp(-1.0, 1.0) + 1.0) * 0.5
    else:
        x = x.clamp(0.0, 1.0)
    rgb = (x.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return rgb


@dataclass
class LandmarkBitEncoder:
    predictor_path: str = _DEFAULT_PREDICTOR
    cache_dir: str = ""
    canonical_bits: int = 128
    bits_per_value: int = 4

    def __post_init__(self):
        self.predictor_path = _normalize_path(self.predictor_path)
        if not os.path.isfile(self.predictor_path):
            raise FileNotFoundError(f"landmark predictor not found: {self.predictor_path}")
        self.cache_dir = _normalize_path(self.cache_dir) if self.cache_dir else ""
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self.predictor_path)

    def _cache_key(self, image_path: str) -> str:
        key = f"{_normalize_path(image_path)}|{self.predictor_path}|{self.canonical_bits}|{self.bits_per_value}"
        return _sha1_text(key)

    def _cache_landmark_path(self, image_path: str) -> str:
        return os.path.join(self.cache_dir, "landmarks68", self._cache_key(image_path) + ".npy")

    def _cache_bit_path(self, image_path: str) -> str:
        return os.path.join(self.cache_dir, "bits128", self._cache_key(image_path) + ".npy")

    def detect_landmarks68(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        if image_rgb_u8.ndim != 3 or image_rgb_u8.shape[2] != 3:
            raise ValueError(f"Expected RGB image array with shape [H,W,3], got {image_rgb_u8.shape}")
        gray = np.round(
            0.299 * image_rgb_u8[:, :, 0]
            + 0.587 * image_rgb_u8[:, :, 1]
            + 0.114 * image_rgb_u8[:, :, 2]
        ).astype(np.uint8)
        faces = self._detector(gray, 1)
        if len(faces) == 0:
            raise RuntimeError("No face detected for dlib68 landmark extraction.")
        face = max(faces, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        shape = self._predictor(gray, face)
        pts = np.zeros((68, 2), dtype=np.float32)
        for i in range(68):
            pts[i, 0] = float(shape.part(i).x)
            pts[i, 1] = float(shape.part(i).y)
        return pts

    def load_image_rgb_u8(self, image_path: str) -> np.ndarray:
        try:
            with Image.open(image_path) as img:
                return np.asarray(img.convert("RGB"), dtype=np.uint8)
        except FileNotFoundError:
            raise
        except Exception as exc:
            raise RuntimeError(f"Failed to read image: {image_path}") from exc

    def extract_landmarks68_from_path(self, image_path: str) -> np.ndarray:
        image_path = _normalize_path(image_path)
        if self.cache_dir:
            cached = self._cache_landmark_path(image_path)
            cached_arr = _safe_cache_load(cached, expected_shape=(68, 2))
            if cached_arr is not None:
                return cached_arr.astype(np.float32, copy=False)
        image_rgb = self.load_image_rgb_u8(image_path)
        landmarks = self.detect_landmarks68(image_rgb)
        if self.cache_dir:
            cached = self._cache_landmark_path(image_path)
            _ensure_parent(cached)
            np.save(cached, landmarks.astype(np.float32))
        return landmarks

    def canonical_bits_from_path(self, image_path: str) -> np.ndarray:
        image_path = _normalize_path(image_path)
        if self.cache_dir:
            cached = self._cache_bit_path(image_path)
            cached_arr = _safe_cache_load(cached, expected_shape=(self.canonical_bits,))
            if cached_arr is not None:
                return cached_arr.astype(np.float32, copy=False)
        landmarks = self.extract_landmarks68_from_path(image_path)
        image_rgb = self.load_image_rgb_u8(image_path)
        bits = self.canonical_bits_from_landmarks(landmarks=landmarks, image_rgb_u8=image_rgb)
        if self.cache_dir:
            cached = self._cache_bit_path(image_path)
            _ensure_parent(cached)
            np.save(cached, bits.astype(np.float32))
        return bits

    def canonical_bits_from_landmarks(self, landmarks: np.ndarray, image_rgb_u8: np.ndarray) -> np.ndarray:
        h, w = image_rgb_u8.shape[:2]
        if h <= 0 or w <= 0:
            raise RuntimeError("Invalid image size for landmark bits.")
        landmarks_norm = landmarks.copy()
        landmarks_norm[:, 0] /= float(w)
        landmarks_norm[:, 1] /= float(h)
        return _encode_landmarks_to_bits(
            landmarks_norm,
            num_bits=self.canonical_bits,
            bits_per_value=self.bits_per_value,
        )

    def canonical_bits_from_rgb_u8(self, image_rgb_u8: np.ndarray) -> np.ndarray:
        landmarks = self.detect_landmarks68(image_rgb_u8)
        return self.canonical_bits_from_landmarks(landmarks=landmarks, image_rgb_u8=image_rgb_u8)

    def bits_from_path(self, image_path: str, num_bits: int) -> np.ndarray:
        base = self.canonical_bits_from_path(image_path)
        if num_bits > base.shape[0]:
            raise ValueError(
                f"Requested num_bits={num_bits}, but canonical_bits={base.shape[0]}"
            )
        return base[:num_bits].astype(np.float32, copy=False)

    def bits_from_rgb_u8(self, image_rgb_u8: np.ndarray, num_bits: int) -> np.ndarray:
        base = self.canonical_bits_from_rgb_u8(image_rgb_u8)
        if num_bits > base.shape[0]:
            raise ValueError(
                f"Requested num_bits={num_bits}, but canonical_bits={base.shape[0]}"
            )
        return base[:num_bits].astype(np.float32, copy=False)


def build_landmark_bit_encoder(
    predictor_path: Optional[str] = None,
    cache_dir: str = "",
    canonical_bits: int = 128,
    bits_per_value: int = 4,
) -> LandmarkBitEncoder:
    return LandmarkBitEncoder(
        predictor_path=predictor_path or _DEFAULT_PREDICTOR,
        cache_dir=cache_dir,
        canonical_bits=canonical_bits,
        bits_per_value=bits_per_value,
    )


def bits_from_image_path(
    image_path: str,
    num_bits: int,
    predictor_path: Optional[str] = None,
    cache_dir: str = "",
    canonical_bits: int = 128,
    bits_per_value: int = 4,
) -> np.ndarray:
    encoder = build_landmark_bit_encoder(
        predictor_path=predictor_path,
        cache_dir=cache_dir,
        canonical_bits=canonical_bits,
        bits_per_value=bits_per_value,
    )
    return encoder.bits_from_path(image_path=image_path, num_bits=num_bits)


def bits_from_image_paths(
    image_paths: Sequence[str],
    num_bits: int,
    device: Optional[torch.device] = None,
    predictor_path: Optional[str] = None,
    cache_dir: str = "",
    canonical_bits: int = 128,
    bits_per_value: int = 4,
) -> torch.Tensor:
    encoder = build_landmark_bit_encoder(
        predictor_path=predictor_path,
        cache_dir=cache_dir,
        canonical_bits=canonical_bits,
        bits_per_value=bits_per_value,
    )
    arr: List[np.ndarray] = [encoder.bits_from_path(p, num_bits=num_bits) for p in image_paths]
    if len(arr) == 0:
        return torch.empty((0, num_bits), device=device, dtype=torch.float32)
    stacked = np.stack(arr, axis=0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(stacked)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.float()


def bits_from_image_tensors(
    images: torch.Tensor,
    num_bits: int,
    device: Optional[torch.device] = None,
    predictor_path: Optional[str] = None,
    cache_dir: str = "",
    canonical_bits: int = 128,
    bits_per_value: int = 4,
) -> torch.Tensor:
    encoder = build_landmark_bit_encoder(
        predictor_path=predictor_path,
        cache_dir=cache_dir,
        canonical_bits=canonical_bits,
        bits_per_value=bits_per_value,
    )
    arr: List[np.ndarray] = []
    for image in images:
        rgb = _tensor_image_to_rgb_u8(image)
        arr.append(encoder.bits_from_rgb_u8(rgb, num_bits=num_bits))
    if len(arr) == 0:
        return torch.empty((0, num_bits), device=device, dtype=torch.float32)
    stacked = np.stack(arr, axis=0).astype(np.float32, copy=False)
    tensor = torch.from_numpy(stacked)
    if device is not None:
        tensor = tensor.to(device=device)
    return tensor.float()
