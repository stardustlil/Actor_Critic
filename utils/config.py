class Config:
    def __init__(self):
        self.env_name = 'CartPole-v1'  # 示例环境
        self.gamma = 0.99
        self.actor_lr = 5e-3
        self.critic_lr = 1e-3
        self.hidden_dim = 128
        self.num_episodes = 1000
        self.max_steps_per_episode = 500
        self.seed = 42
        self.device = 'cpu'  # 或cuda