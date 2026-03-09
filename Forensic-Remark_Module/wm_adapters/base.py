import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseWMAdapter(ABC):
    """
    水印编解码器统一适配基类。

    ReMark 训练每次绑定一个固定的水印模型（由 config.wm_model 指定）。
    adapter 负责：
      1. 加载并冻结原始水印模型的权重
      2. 将外部 canonical 格式（{0,1} message, [-1,1] image）
         转换为该模型期望的内部格式，调用后再转换回来
      3. 自动适配尺寸：若 WM 模型在固定分辨率上预训练，
         encode/decode 会自动 resize 输入，并将输出 resize 回原始尺寸

    子类需实现：
      _load_models()           加载模型权重
      _encode(images, messages) WM 编码（输入已保证为 self.image_size）
      _decode(images)           WM 解码（输入已保证为 self.image_size）
      message_length            属性

    外部调用（勿直接重写 encode/decode，尺寸适配在此处统一处理）：
      wm_images = adapter.encode(images, messages)  → no_grad
      logits    = adapter.decode(images)             → 允许梯度流
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self._encoder = None
        self._decoder = None
        self._load_models()
        self._freeze()

    @abstractmethod
    def _load_models(self):
        """加载原始水印模型的 encoder 和 decoder，赋值给 self._encoder / self._decoder"""

    def _freeze(self):
        """冻结所有参数，ReMark 训练过程中不更新水印模型"""
        for model in [self._encoder, self._decoder]:
            if model is not None and isinstance(model, nn.Module):
                model.eval()
                for p in model.parameters():
                    p.requires_grad_(False)

    # ── 尺寸适配 ──────────────────────────────────────────────────────────────

    @property
    def image_size(self) -> int:
        """
        WM 模型预训练时的图像尺寸（正方形边长，像素）。
        None 表示模型支持任意尺寸，无需 resize。

        子类覆盖示例：
            @property
            def image_size(self):
                return 128
        """
        return None

    def _resize(self, images: torch.Tensor, size) -> torch.Tensor:
        """
        将图像双线性 resize 到目标尺寸。
        size=None 或尺寸已匹配时原样返回（零拷贝）。
        """
        if size is None:
            return images
        if isinstance(size, int):
            size = (size, size)
        if images.shape[-2:] == torch.Size(list(size)):
            return images
        return F.interpolate(images, size=size, mode='bilinear', align_corners=False)

    # ── 公共接口（由基类统一调度，子类勿覆盖）───────────────────────────────────

    @torch.no_grad()
    def encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        嵌入水印。

        Args:
            images:   canonical [-1,1] BCHW，任意尺寸
            messages: {0,1} float tensor，shape [B, message_length]
        Returns:
            wm_images: canonical [-1,1] BCHW，与输入同尺寸

        流程：resize → _encode → resize back
        注意：整体包裹在 no_grad 中，encoder 不参与反传。
        """
        orig_size = (images.shape[-2], images.shape[-1])
        images_in = self._resize(images, self.image_size)
        wm = self._encode(images_in, messages)
        return self._resize(wm, orig_size)

    def decode(self, images: torch.Tensor) -> torch.Tensor:
        """
        提取水印。

        Args:
            images: canonical [-1,1] BCHW（通常是 ReMark 修复后的图像），任意尺寸
        Returns:
            logits: shape [B, message_length]，未经 sigmoid，BCE-ready

        注意：不加 no_grad，梯度需经由 decoder 反传到 ReMark。
              只对 decoder 做 resize，不影响梯度流。
        """
        images_in = self._resize(images, self.image_size)
        return self._decode(images_in)

    # ── 子类实现 ──────────────────────────────────────────────────────────────

    @abstractmethod
    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        WM 编码实现。
        Args:
            images:   canonical [-1,1] BCHW，尺寸已保证为 self.image_size（若非 None）
            messages: {0,1} float tensor，shape [B, message_length]
        Returns:
            wm_images: canonical [-1,1] BCHW，与 images 同尺寸
        """

    @abstractmethod
    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        """
        WM 解码实现。
        Args:
            images: canonical [-1,1] BCHW，尺寸已保证为 self.image_size（若非 None）
        Returns:
            logits: shape [B, message_length]，BCE-ready
        """

    @property
    def message_length(self) -> int:
        """水印消息的 bit 长度，子类实现"""
        raise NotImplementedError
