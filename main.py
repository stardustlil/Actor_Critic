import argparse
import torch
from utils.config import Config
from env.make_env import make_env
from agents.qac_agent import QACAgent
from trainer import Trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='CartPole-v1')
    parser.add_argument('--num_episodes', type=int, default=1000)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--actor_lr', type=float, default=1e-3)
    parser.add_argument('--critic_lr', type=float, default=1e-2)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cpu')
    args = parser.parse_args()

    config = Config()
    config.env_name = args.env
    config.num_episodes = args.num_episodes
    config.gamma = args.gamma
    config.actor_lr = args.actor_lr
    config.critic_lr = args.critic_lr
    config.hidden_dim = args.hidden_dim
    config.seed = args.seed
    config.device = args.device
    config.max_steps_per_episode = 500  # 可配置

    env = make_env(config.env_name, seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = QACAgent(state_dim, action_dim, config)
    trainer = Trainer(env, agent, config)
    rewards = trainer.train()

    # 可以保存模型等

if __name__ == '__main__':
    main()