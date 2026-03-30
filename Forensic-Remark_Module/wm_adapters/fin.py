import os
import sys
import torch
import numpy as np

from .base import BaseWMAdapter
from .registry import register_wm

# FIN 根目录（通过 config 传入，默认指向项目内的 Forensic-FIN）
_FIN_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'Forensic-FIN'
)


@register_wm('fin')
class FINAdapter(BaseWMAdapter):
    """
    FIN 水印模型适配器。

    FIN 内部格式：
      - 图像：[-1, 1]（与 canonical 一致，无需转换）
      - 消息：{-0.5, 0.5}（canonical {0,1} 需转换）
      - forward([image, message]) → (stego, left_noise)
      - forward([stego, zeros], rev=True) → (re_image, re_message)

    预训练尺寸：128×128（CelebA-HQ）
    若输入图像尺寸不同，基类会自动 resize 到 128×128 再调用本 adapter，
    并将输出 resize 回原始尺寸。
    """

    _model_instance = None  # 类级缓存，避免重复加载权重

    def __init__(self, cfg):
        # 先设置路径再调用父类（父类 __init__ 会调用 _load_models）
        self._fin_root = getattr(
            getattr(cfg, 'wm_adapter_paths', None), 'fin', _FIN_ROOT
        )
        self._ckpt_path = getattr(
            getattr(cfg, 'wm_adapter_ckpts', None), 'fin',
            os.path.join(self._fin_root, 'experiments', 'celeba_hq_128', 'FED.pt')
        )
        self._message_length = 128
        super().__init__(cfg)

    @property
    def image_size(self) -> int:
        """FIN 在 128×128 上预训练，声明后基类自动适配输入/输出尺寸"""
        return 128

    def _load_models(self):
        if FINAdapter._model_instance is not None:
            self._encoder = FINAdapter._model_instance
            self._decoder = self._encoder
            return

        if self._fin_root not in sys.path:
            sys.path.insert(0, self._fin_root)

        try:
            from models.encoder_decoder import FED
        except ImportError as e:
            raise ImportError(
                f"FIN 依赖加载失败，请检查路径 {self._fin_root}\n{e}"
            )

        model = FED(diff=False, length=self._message_length)

        if not os.path.exists(self._ckpt_path):
            raise FileNotFoundError(
                f"FIN checkpoint 不存在：{self._ckpt_path}\n"
                f"请先训练 FIN 或指定正确的 ckpt 路径（config.wm_adapter_ckpts.fin）"
            )

        state = torch.load(self._ckpt_path, map_location='cpu')
        net_state = {k: v for k, v in state['net'].items() if 'tmp_var' not in k}
        model.load_state_dict(net_state)

        self._encoder = model
        self._decoder = model   # FIN 的 encoder/decoder 是同一个可逆网络
        FINAdapter._model_instance = model

    @property
    def message_length(self) -> int:
        return self._message_length

    def _encode(self, images: torch.Tensor, messages: torch.Tensor) -> torch.Tensor:
        """
        FIN 编码实现（基类已保证 images 尺寸为 128×128）。
        canonical {0,1} message → FIN 内部 {-0.5, 0.5}。
        """
        device = images.device
        self._encoder.to(device)
        fin_messages = messages.float() - 0.5  # {0,1} → {-0.5, 0.5}
        stego, _ = self._encoder([images, fin_messages])
        return stego.clamp(-1, 1)

    def _decode(self, images: torch.Tensor) -> torch.Tensor:
        """
        FIN 逆向传播提取消息（基类已保证 images 尺寸为 128×128）。
        不加 no_grad，梯度需要经由 decoder 反传到 ReMark。
        """
        device = images.device
        self._decoder.to(device)
        zeros = torch.zeros(images.shape[0], self._message_length).to(device)
        _, re_message = self._decoder([images, zeros], rev=True)
        # re_message 范围约为 {-0.5, 0.5}，转为 logits（乘以较大系数使 BCE 有效）
        return re_message * 10.0
