import argparse
import torch
from utils.config import Config
from env.make_env import make_env
from agents.a2c_agent import A2CAgent
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to YAML config file')
    # 命令行参数（用于覆盖 YAML 中的值）
    parser.add_argument('--env', type=str, help='Override env_name')
    parser.add_argument('--num_episodes', type=int, help='Override num_episodes')
    parser.add_argument('--gamma', type=float, help='Override gamma')
    parser.add_argument('--actor_lr', type=float, help='Override actor_lr')
    parser.add_argument('--critic_lr', type=float, help='Override critic_lr')
    parser.add_argument('--hidden_dim', type=int, help='Override hidden_dim')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--device', type=str, help='Override device')
    args = parser.parse_args()

    # 收集要覆盖的参数（仅保留非 None 的值）
    overrides = {k: v for k, v in vars(args).items() 
                 if v is not None and k != 'config'}

    # 从 YAML 加载配置，并应用覆盖
    config = Config.from_yaml(args.config, overrides)

    env = make_env(config.env_name, seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(state_dim, action_dim, config)
    trainer = Trainer(env, agent, config)
    rewards = trainer.train()

if __name__ == '__main__':
    main()