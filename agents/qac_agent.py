import torch
import torch.nn.functional as F
from agents.base_agent import BaseAgent
from agents import register_agent
from models.actor import DiscreteActor
from models.critic import QCritic


@register_agent("qac")
class QACAgent(BaseAgent):
    def __init__(self, state_dim, action_dim, config):
        self.gamma = config.gamma
        self.device = config.device

        self.actor = DiscreteActor(state_dim, action_dim, config.hidden_dim).to(self.device)
        self.critic = QCritic(state_dim, action_dim, config.hidden_dim).to(self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=config.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.critic_lr)

    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, _ = self.actor.get_action(state, deterministic)
        return action.item()

    def update(self, transition):
        state, action, reward, next_state, done = transition

        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = torch.LongTensor([action]).to(self.device)
        reward = torch.FloatTensor([reward]).to(self.device)
        next_state = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
        done = torch.FloatTensor([done]).to(self.device)

        with torch.no_grad():
            if done.item() == 1:
                target = reward
            else:
                next_action, _ = self.actor.get_action(next_state, deterministic=False)
                next_q = self.critic(next_state, next_action)
                target = reward + self.gamma * next_q

        current_q = self.critic(state, action)
        critic_loss = F.mse_loss(current_q, target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        with torch.no_grad():
            q_val = self.critic(state, action)

        log_prob = self.actor.evaluate(state, action)
        actor_loss = -(log_prob * q_val).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        return critic_loss.item(), actor_loss.item()
