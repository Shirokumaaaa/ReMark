ATTACK_REGISTRY = {}


def register_attack(name: str):
    """
    装饰器，将攻击类注册到全局 registry。

    用法：
        @register_attack('stargan')
        class StarGANAttack(BaseAttack):
            ...

    注册后可通过 build_attack('stargan', cfg) 实例化。
    """
    def decorator(cls):
        if name in ATTACK_REGISTRY:
            raise ValueError(f"攻击模型 '{name}' 已注册，请检查是否重复定义")
        ATTACK_REGISTRY[name] = cls
        return cls
    return decorator


def build_attack(name: str, cfg):
    """
    根据名字实例化攻击模型。

    Args:
        name: 注册名（对应 config 中 attacks.online/offline 列表里的字符串）
        cfg:  完整配置对象（SimpleNamespace）
    """
    if name not in ATTACK_REGISTRY:
        available = list(ATTACK_REGISTRY.keys())
        raise KeyError(
            f"未找到攻击模型 '{name}'。"
            f"已注册的模型：{available}。"
            f"请确认对应的 adapter 文件已 import（通常在 attacks/__init__.py 中引入）。"
        )
    return ATTACK_REGISTRY[name](cfg)
