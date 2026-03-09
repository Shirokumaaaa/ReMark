import os
import yaml
from types import SimpleNamespace


def _dict_to_ns(d):
    """递归将 dict 转为 SimpleNamespace，支持 cfg.model.latent_channels 访问"""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_ns(v) for k, v in d.items()})
    if isinstance(d, list):
        return [_dict_to_ns(i) for i in d]
    return d


def load_config(path: str, override: str = None) -> SimpleNamespace:
    """
    加载 YAML 配置文件。

    Args:
        path:     基础配置文件路径（如 configs/stage1_vae.yaml）
        override: 可选的实验 override 文件（configs/experiments/xxx.yaml）
                  其中的字段会深度合并覆盖基础配置

    Returns:
        SimpleNamespace，支持属性访问（cfg.training.lr 等）
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(root, path)) as f:
        cfg = yaml.safe_load(f)

    if override is not None:
        with open(os.path.join(root, override)) as f:
            ov = yaml.safe_load(f)
        cfg = _deep_merge(cfg, ov)

    # 将所有相对路径解析为基于模块根目录的绝对路径
    cfg = _resolve_paths(cfg, root)
    return _dict_to_ns(cfg)


def _deep_merge(base: dict, override: dict) -> dict:
    """深度合并，override 中的字段覆盖 base，不存在的字段保留"""
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _resolve_paths(cfg, root):
    """将 paths 字段下的相对路径转为绝对路径"""
    if isinstance(cfg, dict):
        if 'paths' in cfg:
            cfg['paths'] = {
                k: os.path.join(root, v) if not os.path.isabs(v) else v
                for k, v in cfg['paths'].items()
            }
        if 'data' in cfg:
            for key in ['train_csv', 'val_csv', 'test_csv']:
                if key in cfg['data'] and not os.path.isabs(cfg['data'][key]):
                    cfg['data'][key] = os.path.join(root, cfg['data'][key])
    return cfg
