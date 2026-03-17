import numpy as np
from utils.visualization import plot_training_curves


class Trainer:
    def __init__(self, env, agent, config):
        self.env = env
        self.agent = agent
        self.config = config
        self.episode_rewards = []
        self.actor_losses = []
        self.critic_losses = []
        self._seed_set = False

    def train(self):
        for episode in range(self.config.num_episodes):
            if not self._seed_set:
                state, _ = self.env.reset(seed=self.config.seed)
                self._seed_set = True
            else:
                state, _ = self.env.reset()

            done = False
            episode_reward = 0
            step = 0

            while not done and step < self.config.max_steps_per_episode:
                action = self.agent.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward

                transition = (state, action, reward, next_state, done)
                critic_loss, actor_loss = self.agent.update(transition)
                self.critic_losses.append(critic_loss)
                self.actor_losses.append(actor_loss)

                state = next_state
                step += 1

            self.episode_rewards.append(episode_reward)

            if (episode + 1) % self.config.log_interval == 0:
                avg_reward = np.mean(self.episode_rewards[-self.config.log_interval:])
                print(f"Episode {episode+1}, Avg Reward (last {self.config.log_interval}): {avg_reward:.2f}")

        plot_path = None
        if self.config.enable_visualization:
            plot_path = plot_training_curves(
                rewards=self.episode_rewards,
                actor_losses=self.actor_losses,
                critic_losses=self.critic_losses,
                output_path=self.config.plot_path,
            )
            print(f"Training curves saved to: {plot_path}")

        return {
            "episode_rewards": self.episode_rewards,
            "actor_losses": self.actor_losses,
            "critic_losses": self.critic_losses,
            "plot_path": plot_path,
        }
