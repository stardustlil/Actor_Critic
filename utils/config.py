# utils/config.py
import yaml
import os
from dataclasses import dataclass
from typing import Optional

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
    def from_yaml(cls, path: str, overrides=None):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)

        if overrides:
            data.update(overrides)

        # 自动类型修复
        for k, v in data.items():
            if isinstance(v, str):
                try:
                    data[k] = float(v)
                except:
                    try:
                        data[k] = int(v)
                    except:
                        pass

        return cls(**data)