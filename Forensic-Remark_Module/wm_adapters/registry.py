WM_REGISTRY = {}


def register_wm(name: str):
    """
    装饰器，将水印 adapter 注册到全局 registry。

    用法：
        @register_wm('fin')
        class FINAdapter(BaseWMAdapter):
            ...
    """
    def decorator(cls):
        if name in WM_REGISTRY:
            raise ValueError(f"水印模型 '{name}' 已注册，请检查是否重复定义")
        WM_REGISTRY[name] = cls
        return cls
    return decorator


def build_wm_adapter(name: str, cfg):
    """
    根据 config.wm_model 实例化水印 adapter。
    """
    if name not in WM_REGISTRY:
        available = list(WM_REGISTRY.keys())
        raise KeyError(
            f"未找到水印模型 '{name}'。"
            f"已注册的模型：{available}。"
            f"请确认对应的 adapter 文件已 import（通常在 wm_adapters/__init__.py 中引入）。"
        )
    return WM_REGISTRY[name](cfg)
