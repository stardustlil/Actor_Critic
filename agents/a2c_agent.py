import torch
import torch.nn.functional as F
from models.actor import DiscreteActor
from models.critic import VCritic  


class A2CAgent:
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config.gamma
        self.device = config.device

        self.actor = DiscreteActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = VCritic(state_dim, config.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.actor.get_action(state, deterministic)
        return action.item(), log_prob   # 返回动作和 log_prob，但 trainer 中不需要 log_prob

    def update(self, transition):
        """
        transition: (state, action, reward, next_state, done)
        """
        state, action, reward, next_state, done = transition

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        # ---------- Critic 更新 ----------
        v = self.critic(state)
        with torch.no_grad():
            v_next = self.critic(next_state)
            target = reward + self.gamma * v_next * (1 - done)   # done 时 target = reward
        critic_loss = F.mse_loss(v, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------- Actor 更新：使用优势 = r + γV(s') - V(s) ----------
        with torch.no_grad():
            advantage = target - v   # target = r + γV(s')，v 是当前 V(s)
        log_prob = self.actor.evaluate(state, action)
        actor_loss = -(log_prob * advantage).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()