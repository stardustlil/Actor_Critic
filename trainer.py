import numpy as np
from env.make_env import make_env

class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.episode_rewards = []
        self._seed_set = False

    def train(self):
        for episode in range(self.config.num_episodes):
            if not self._seed_set:
                state, info = self.env.reset(seed=self.config.seed)
                self._seed_set = True
            else:
                state, info = self.env.reset()

            done = False
            episode_reward = 0
            step = 0

            while not done and step < self.config.max_steps_per_episode:
                # 选择动作
                action, _ = self.agent.select_action(state)   # 忽略 log_prob
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 更新智能体
                transition = (state, action, reward, next_state, done)
                critic_loss, actor_loss = self.agent.update(transition)

                # 移动到下一状态
                state = next_state
                step += 1

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}")

        return self.episode_rewards