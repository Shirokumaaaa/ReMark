import os
import sys
import importlib
import importlib.util
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import BaseAttack
from .registry import register_attack


def _unwrap_data_parallel(module: nn.Module):
    """递归解包 nn.DataParallel，避免 DDP 场景下跨卡冲突。"""
    for name, child in module.named_children():
        if isinstance(child, nn.DataParallel):
            setattr(module, name, child.module)
        else:
            _unwrap_data_parallel(child)


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_LAMPMARK_ROOT = os.path.join(_PROJECT_ROOT, 'Forensic-LampMark')
_MASKWM_ROOT = os.path.join(_PROJECT_ROOT, 'Forensic-MaskWM')


def _purge_conflicting_models_namespace():
    """
    SimSwap 反序列化 arcface checkpoint 时会 import `models.arcface_models`。
    若当前进程先加载过 MaskWM（其包名也叫 `models`），会发生命名冲突。
    这里仅清理来自 MaskWM 的 `models*` 模块。
    """
    to_del = []
    for name, mod in list(sys.modules.items()):
        if not (name == 'models' or name.startswith('models.')):
            continue
        path = getattr(mod, '__file__', '') or ''
        if _MASKWM_ROOT in path:
            to_del.append(name)
    for name in to_del:
        del sys.modules[name]


def _ensure_simswap_models_namespace():
    """
    Force-bind `models.*` to SimSwap's package so torch.load can resolve
    pickled refs like `models.arcface_models`.
    """
    simswap_root = os.path.join(_LAMPMARK_ROOT, 'model', 'SimSwap')
    models_dir = os.path.join(simswap_root, 'models')
    init_py = os.path.join(models_dir, '__init__.py')

    if simswap_root not in sys.path:
        sys.path.insert(0, simswap_root)

    # If already pointing to SimSwap models, keep it.
    mod = sys.modules.get('models')
    mod_file = getattr(mod, '__file__', '') or ''
    if mod is not None and models_dir in mod_file:
        try:
            importlib.import_module('models.arcface_models')
            return
        except Exception:
            pass

    spec = importlib.util.spec_from_file_location(
        'models', init_py, submodule_search_locations=[models_dir]
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Cannot create spec for SimSwap models package: {init_py}')
    pkg = importlib.util.module_from_spec(spec)
    sys.modules['models'] = pkg
    spec.loader.exec_module(pkg)
    importlib.import_module('models.arcface_models')


@register_attack('simswap')
class SimSwapAttack(BaseAttack):
    """
    SimSwap 攻击适配器。

    复用 Forensic-LampMark/model/deepfake_manipulations.py::SimSwapModel，
    统一到 ReMark 的 canonical 接口：
      input/output: [-1, 1], BCHW
    """

    _model_instance = None

    def __init__(self, cfg):
        super().__init__(cfg)
        opts = getattr(cfg, 'attack_options', None)
        self._swap_check_enabled = bool(getattr(opts, 'enforce_nontrivial_swap', True))
        self._swap_check_eps = float(getattr(opts, 'nontrivial_swap_eps', 1e-4))
        self._load_model()

    def _load_model(self):
        if SimSwapAttack._model_instance is not None:
            self._model = SimSwapAttack._model_instance
            return

        _purge_conflicting_models_namespace()
        _ensure_simswap_models_namespace()

        if _LAMPMARK_ROOT not in sys.path:
            sys.path.insert(0, _LAMPMARK_ROOT)

        try:
            from model.deepfake_manipulations import SimSwapModel
        except ImportError as e:
            raise ImportError(
                f"SimSwap 依赖加载失败，请检查路径 {_LAMPMARK_ROOT}\n{e}"
            )

        img_size = getattr(getattr(self.cfg, 'data', None), 'image_size', 128)

        # SimSwapModel 内部会 parse argparse + 依赖相对路径，需隔离 argv/cwd
        _saved_argv = sys.argv
        _saved_cwd = os.getcwd()
        sys.argv = sys.argv[:1]
        os.chdir(_LAMPMARK_ROOT)
        try:
            self._model = SimSwapModel(img_size=img_size, mode='test')
        finally:
            sys.argv = _saved_argv
            os.chdir(_saved_cwd)

        self._model.eval()
        for p in self._model.parameters():
            p.requires_grad_(False)
        _unwrap_data_parallel(self._model)

        SimSwapAttack._model_instance = self._model

    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        # canonical [-1,1] -> SimSwap 期望的 [0,1]
        return ((images.clamp(-1, 1) + 1.0) / 2.0).clamp(0, 1)

    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        无 cover 时退化为 self-source（同图），用于兼容基类 __call__。
        训练路径优先走 attack_with_cover()。
        """
        device = preprocessed.device
        self._model.to(device)
        with torch.no_grad():
            fake = self._model([preprocessed, preprocessed, device])
        return fake

    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        # SimSwap 输出约为 [0,1]，转回 canonical [-1,1]
        restored = F.interpolate(output, size=original_size, mode='bilinear', align_corners=False).clamp(0, 1)
        return (restored * 2.0 - 1.0).clamp(-1, 1)

    def attack_with_cover(self, wm_images: torch.Tensor, cover_images: torch.Tensor) -> torch.Tensor:
        """
        与 LampMark 调用方式对齐：
          source <- roll(clean cover)
          target <- wm image
        由 SimSwapModel 内部 forward 实现。
        """
        original_size = (wm_images.shape[-2], wm_images.shape[-1])
        wm_pre = self.preprocess(wm_images)
        cover_pre = self.preprocess(cover_images)

        device = wm_pre.device
        self._model.to(device)
        with torch.no_grad():
            fake = self._model([wm_pre, cover_pre, device])

        out = self.postprocess(fake, original_size)
        self._validate_output(out, original_size)
        self._assert_nontrivial_swap(wm_images, out)
        return out

    def _assert_nontrivial_swap(self, original: torch.Tensor, swapped: torch.Tensor):
        """防止 silent fallback：输出若与输入几乎完全相同，则判为非真实换脸。"""
        if not self._swap_check_enabled:
            return
        with torch.no_grad():
            per_sample_diff = (original - swapped).abs().mean(dim=(1, 2, 3))
            bad = (per_sample_diff < self._swap_check_eps)
        if bool(bad.any()):
            bad_n = int(bad.sum().item())
            raise RuntimeError(
                f"[SimSwapAttack] nontrivial swap check failed: "
                f"{bad_n}/{original.shape[0]} samples have mean_abs_diff < "
                f"{self._swap_check_eps:.1e}. Possible non-deepfake fallback."
            )
