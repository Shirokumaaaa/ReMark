import torch
from abc import ABC, abstractmethod


class BaseAttack(ABC):
    """
    攻击模型统一基类。

    所有攻击 adapter 继承此类，实现三个方法：
      - preprocess:  canonical [-1,1] BCHW → 攻击模型期望格式
      - generate:    模型推理（子类实现）
      - postprocess: 攻击模型输出 → canonical [-1,1] BCHW

    外部调用只需：fake = attack(images)
    __call__ 由基类统一调度，保证输出格式一致，子类无法绕过 postprocess。
    """

    def __init__(self, cfg):
        self.cfg = cfg

    @abstractmethod
    def preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: canonical 格式，float32，[-1, 1]，BCHW
        Returns:
            攻击模型期望的输入格式
        """

    @abstractmethod
    def generate(self, preprocessed: torch.Tensor) -> torch.Tensor:
        """
        攻击模型推理，子类实现。
        输入/输出格式均为模型内部格式（preprocess/postprocess 负责转换）。
        """

    @abstractmethod
    def postprocess(self, output: torch.Tensor, original_size: tuple) -> torch.Tensor:
        """
        Args:
            output:        攻击模型的原始输出
            original_size: (H, W)，用于恢复到输入尺寸
        Returns:
            canonical 格式，float32，[-1, 1]，BCHW，尺寸与输入一致
        """

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        统一调度入口，外部调用此方法。
        保证：输入和输出都是 canonical [-1,1] BCHW。
        """
        original_size = (images.shape[-2], images.shape[-1])
        x = self.preprocess(images)
        x = self.generate(x)
        out = self.postprocess(x, original_size)
        self._validate_output(out, original_size)
        return out

    def _validate_output(self, out: torch.Tensor, original_size: tuple):
        """输出格式断言，写错立刻报错，不会等到训练几百步后才发现"""
        assert out.shape[-2:] == torch.Size(list(original_size)), \
            f"[{self.__class__.__name__}] postprocess 未恢复原始尺寸: " \
            f"期望 {original_size}，得到 {tuple(out.shape[-2:])}"
        assert out.min() >= -1.1 and out.max() <= 1.1, \
            f"[{self.__class__.__name__}] postprocess 输出值域超出 [-1,1]: " \
            f"[{out.min():.3f}, {out.max():.3f}]"

    @staticmethod
    def to_display(tensor: torch.Tensor) -> torch.Tensor:
        """canonical [-1,1] → [0,1]，用于 sample 可视化，统一调用"""
        return (tensor.clamp(-1, 1) + 1) / 2
