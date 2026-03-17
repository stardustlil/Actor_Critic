import torch
import torch.nn.functional as F
from models.actor import DiscreteActor
from models.critic import QCritic

class QACAgent:
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config.gamma
        self.device = config.device

        self.actor = DiscreteActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = QCritic(state_dim, action_dim, config.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_action(self, state, deterministic=False):
        """给定状态，根据当前策略选择动作（用于环境交互）"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.get_action(state, deterministic)
        return action.item()

    def update(self, transition):
        """
        执行单步更新：先更新 Critic（TD(0)），再更新 Actor（策略梯度）。
        transition 元组包含：
            (state, action, reward, next_state, done, next_action)
        其中 next_action 在 done=True 时为 None。
        """
        state, action, reward, next_state, done, next_action = transition

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        # ---------- Critic 更新 ----------
        with torch.no_grad():
            if done:
                target = reward
            else:
                # 使用 next_action 评估下一状态的 Q 值
                next_action_tensor = torch.LongTensor([next_action]).to(self.device)
                next_q = self.critic(next_state, next_action_tensor)
                target = reward + self.gamma * next_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Actor 更新（使用更新后的 Critic）----------
        with torch.no_grad():
            q_val = self.critic(state, action)   # 更新后的 Q 值

        log_prob = self.actor.evaluate(state, action)
        actor_loss = -(log_prob * q_val).mean()   # 负号：梯度上升

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()