# utils/config.py
import yaml
from dataclasses import dataclass, fields

@dataclass
class Config:
    # 环境参数
    env_name: str = 'CartPole-v1'
    max_steps_per_episode: int = 500
    seed: int = 42
    
    # 训练参数
    num_episodes: int = 1000
    gamma: float = 0.99
    actor_lr: float = 5e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 128
    
    # 设备
    device: str = 'cpu'

    @classmethod
    def _coerce_value(cls, field_name, value):
        """按 Config 的字段类型对覆盖值进行转换。"""
        expected_type = cls.__dataclass_fields__[field_name].type

        if expected_type is int and isinstance(value, str):
            return int(value)
        if expected_type is float and isinstance(value, str):
            return float(value)
        if expected_type is str:
            return str(value)
        return value

    @classmethod
    def from_yaml(cls, path: str, overrides=None):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if overrides:
            data.update(overrides)

        # 自动类型修复（仅处理 Config 中已声明字段）
        valid_keys = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in valid_keys}

        for key, value in filtered.items():
            filtered[key] = cls._coerce_value(key, value)

        return cls(**filtered)
