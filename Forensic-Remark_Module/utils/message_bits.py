import hashlib
import os
from typing import Iterable, List

import numpy as np
import torch

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in os.sys.path:
    os.sys.path.insert(0, _REPO_ROOT)


def _normalize_key(key: str) -> str:
    return os.path.abspath(os.path.expanduser(str(key)))


def deterministic_bits_from_key(key: str, length: int, salt: str = "remark_v1") -> np.ndarray:
    """
    Generate stable {0,1} bit vector from key string.

    Uses SHA-256 in counter mode so any length is supported.
    """
    if length <= 0:
        return np.zeros((0,), dtype=np.float32)
    key_n = _normalize_key(key)
    out = bytearray()
    counter = 0
    need_bytes = (length + 7) // 8
    while len(out) < need_bytes:
        payload = f"{salt}|{counter}|{key_n}".encode("utf-8")
        out.extend(hashlib.sha256(payload).digest())
        counter += 1
    bits = np.unpackbits(np.frombuffer(bytes(out[:need_bytes]), dtype=np.uint8))[:length]
    return bits.astype(np.float32, copy=False)


def deterministic_messages_from_paths(
    paths: Iterable[str],
    message_len: int,
    device: torch.device,
    salt: str = "remark_v1",
) -> torch.Tensor:
    arr: List[np.ndarray] = [
        deterministic_bits_from_key(p, message_len, salt=salt) for p in paths
    ]
    if len(arr) == 0:
        return torch.empty((0, message_len), device=device, dtype=torch.float32)
    stacked = np.stack(arr, axis=0)
    return torch.from_numpy(stacked).to(device=device, dtype=torch.float32)


def landmark_messages_from_paths(
    paths: Iterable[str],
    message_len: int,
    device: torch.device,
    predictor_path: str = "",
    cache_dir: str = "",
    canonical_bits: int = 128,
    bits_per_value: int = 4,
) -> torch.Tensor:
    # Lazy import: only require dlib pipeline when landmark messages are actually used.
    from common.landmark_bits import bits_from_image_paths

    return bits_from_image_paths(
        image_paths=list(paths),
        num_bits=message_len,
        device=device,
        predictor_path=predictor_path or None,
        cache_dir=cache_dir,
        canonical_bits=canonical_bits,
        bits_per_value=bits_per_value,
    )
