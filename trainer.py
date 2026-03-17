import numpy as np
from env.make_env import make_env
from agents.qac_agent import QACAgent

class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.episode_rewards = []
        self._seed_set = False   # 标记是否已设置过种子

    def train(self):
        for episode in range(self.config.num_episodes):
            # 第一次 reset 时设置种子，后续 reset 不再传入种子
            if not self._seed_set:
                state, info = self.env.reset(seed=self.config.seed)
                self._seed_set = True
            else:
                state, info = self.env.reset()

            done = False
            episode_reward = 0
            step = 0

            # 初始动作
            action = self.agent.select_action(state)

            while not done and step < self.config.max_steps_per_episode:
                # Gymnasium step 返回 5 个值
                next_state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                # 根据当前策略选择下一个动作（用于 Critic 的 TD 目标）
                if not done:
                    next_action = self.agent.select_action(next_state)
                else:
                    next_action = None

                # 更新智能体
                transition = (state, action, reward, next_state, done, next_action)
                critic_loss, actor_loss = self.agent.update(transition)

                # 移动到下一时刻
                state = next_state
                if not done:
                    action = next_action   # 沿用之前选择的下一个动作
                step += 1

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode+1}, Avg Reward (last 100): {avg_reward:.2f}")

        return self.episode_rewards