import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 损失函数注册表 ──────────────────────────────────────────────────────────────

LOSS_REGISTRY = {}


def register_loss(name: str):
    def decorator(fn):
        LOSS_REGISTRY[name] = fn
        return fn
    return decorator


# ── 各损失函数实现 ──────────────────────────────────────────────────────────────

@register_loss('l1')
def l1_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


@register_loss('mse')
def mse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


@register_loss('bce')
def bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """水印消息损失，logits 未经 sigmoid，target 为 {0,1}"""
    return F.binary_cross_entropy_with_logits(logits, target)


@register_loss('kl')
def kl_loss(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """VAE KL 散度：-0.5 * sum(1 + logvar - mu^2 - exp(logvar))"""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


@register_loss('lpips')
def lpips_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    感知损失，依赖 lpips 库（懒加载，避免无 lpips 时整个模块报错）。
    首次调用时初始化，之后复用。
    """
    if lpips_loss._fn is None:
        try:
            import lpips
            lpips_loss._fn = lpips.LPIPS(net='vgg').to(pred.device)
            lpips_loss._fn.eval()
            for p in lpips_loss._fn.parameters():
                p.requires_grad_(False)
        except ImportError:
            raise ImportError("lpips 未安装，请运行 pip install lpips，或在 config 中禁用 lpips loss")
    return lpips_loss._fn(pred, target).mean()

lpips_loss._fn = None  # 懒加载状态


# ── 组合损失计算器 ──────────────────────────────────────────────────────────────

class LossComputer:
    """
    根据 config 中的 losses 配置，动态组合多个损失项。

    支持单独开关每个损失项（enabled: false）和 β warm-up（warmup: true）。

    用法：
        lc = LossComputer(cfg.losses)
        total, breakdown = lc.compute(
            l1=(pred, target),
            bce=(logits, messages),
            kl=(mu, logvar),
        )
    """

    def __init__(self, losses_cfg, current_epoch: int = 0, warmup_epochs: int = 0):
        self.cfg = losses_cfg
        self.current_epoch = current_epoch
        self.warmup_epochs = warmup_epochs

    def _kl_weight(self, base_weight: float) -> float:
        """KL warm-up：前 warmup_epochs 个 epoch 线性从 0 增到目标权重"""
        if self.warmup_epochs <= 0:
            return base_weight
        ratio = min(1.0, self.current_epoch / self.warmup_epochs)
        return base_weight * ratio

    @staticmethod
    def _linear_interp(epoch: int, start_epoch: int, end_epoch: int, start_v: float, end_v: float) -> float:
        if end_epoch <= start_epoch:
            return end_v if epoch >= start_epoch else start_v
        if epoch <= start_epoch:
            return start_v
        if epoch >= end_epoch:
            return end_v
        ratio = float(epoch - start_epoch) / float(end_epoch - start_epoch)
        return start_v + (end_v - start_v) * ratio

    def _scheduled_weight(self, name: str, loss_cfg) -> float:
        """
        统一损失权重调度（向后兼容）：
        - 默认使用 loss_cfg.weight
        - kl + warmup: 沿用历史 KL warm-up 逻辑
        - 可选 schedule:
            losses.<name>.schedule:
              enabled: true
              start_epoch: 0
              end_epoch: 40   # 或者 ramp_epochs
              start_weight: <float>  # 默认取当前 weight
              end_weight:   <float>  # 默认取 target_weight 或当前 weight
        """
        base_weight = float(getattr(loss_cfg, 'weight', 1.0))
        if name == 'kl' and getattr(loss_cfg, 'warmup', False):
            base_weight = self._kl_weight(base_weight)

        schedule = getattr(loss_cfg, 'schedule', None)
        if schedule is None or not bool(getattr(schedule, 'enabled', False)):
            return base_weight

        start_epoch = int(getattr(schedule, 'start_epoch', 0))
        if hasattr(schedule, 'end_epoch'):
            end_epoch = int(getattr(schedule, 'end_epoch'))
        else:
            ramp_epochs = int(getattr(schedule, 'ramp_epochs', 0))
            end_epoch = start_epoch + ramp_epochs

        start_weight = float(getattr(schedule, 'start_weight', base_weight))
        end_weight = float(
            getattr(
                schedule,
                'end_weight',
                getattr(schedule, 'target_weight', base_weight),
            )
        )
        return self._linear_interp(
            epoch=self.current_epoch,
            start_epoch=start_epoch,
            end_epoch=end_epoch,
            start_v=start_weight,
            end_v=end_weight,
        )

    def get_effective_weight(self, name: str) -> float:
        loss_cfg = getattr(self.cfg, name, None)
        if loss_cfg is None:
            return 0.0
        return self._scheduled_weight(name, loss_cfg)

    def compute(self, **kwargs) -> tuple:
        """
        Args:
            **kwargs: {loss_name: (pred, target, ...) 或 (mu, logvar) for kl}
        Returns:
            (total_loss, breakdown_dict)
        """
        total = torch.tensor(0.0, device=next(iter(kwargs.values()))[0].device)
        breakdown = {}

        for name, inputs in kwargs.items():
            loss_cfg = getattr(self.cfg, name, None)
            if loss_cfg is None:
                continue
            enabled = getattr(loss_cfg, 'enabled', True)
            if not enabled:
                continue

            weight = self._scheduled_weight(name, loss_cfg)

            if name not in LOSS_REGISTRY:
                raise KeyError(f"未注册的损失函数：'{name}'，可用：{list(LOSS_REGISTRY.keys())}")

            val = LOSS_REGISTRY[name](*inputs)
            breakdown[name] = val.item()
            total = total + weight * val

        return total, breakdown
