import yaml
from dataclasses import dataclass


@dataclass
class Config:
    env_name: str = 'CartPole-v1'
    max_steps_per_episode: int = 500
    seed: int = 42

    algorithm: str = 'a2c'
    num_episodes: int = 1000
    gamma: float = 0.99
    actor_lr: float = 5e-3
    critic_lr: float = 1e-3
    hidden_dim: int = 128
    log_interval: int = 100

    enable_visualization: bool = True
    plot_path: str = 'outputs/training_curves.png'

    device: str = 'cpu'

    @classmethod
    def from_yaml(cls, path: str, overrides=None):
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f) or {}

        if overrides:
            data.update(overrides)

        for k, v in data.items():
            if isinstance(v, str):
                lower_v = v.lower()
                if lower_v in {'true', 'false'}:
                    data[k] = lower_v == 'true'
                    continue
                try:
                    data[k] = float(v)
                    if data[k].is_integer():
                        data[k] = int(data[k])
                except ValueError:
                    pass

        return cls(**data)
