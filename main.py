import argparse
from utils.config import Config
from env.make_env import make_env
from agents import build_agent
from trainer import Trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/default.yaml',
                        help='Path to YAML config file')

    parser.add_argument('--env', type=str, help='Override env_name')
    parser.add_argument('--algorithm', type=str, help='Override algorithm (a2c/qac)')
    parser.add_argument('--num_episodes', type=int, help='Override num_episodes')
    parser.add_argument('--gamma', type=float, help='Override gamma')
    parser.add_argument('--actor_lr', type=float, help='Override actor_lr')
    parser.add_argument('--critic_lr', type=float, help='Override critic_lr')
    parser.add_argument('--hidden_dim', type=int, help='Override hidden_dim')
    parser.add_argument('--seed', type=int, help='Override seed')
    parser.add_argument('--device', type=str, help='Override device')
    parser.add_argument('--enable_visualization', type=str, help='Override visualization switch')
    parser.add_argument('--plot_path', type=str, help='Override plot_path')
    args = parser.parse_args()

    overrides = {k if k != 'env' else 'env_name': v for k, v in vars(args).items()
                 if v is not None and k != 'config'}

    config = Config.from_yaml(args.config, overrides)

    env = make_env(config.env_name, seed=config.seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = build_agent(config.algorithm, state_dim, action_dim, config)
    trainer = Trainer(env, agent, config)
    trainer.train()


if __name__ == '__main__':
    main()
