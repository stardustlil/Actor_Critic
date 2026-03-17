import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

class DiscreteActor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.logits = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.logits(x)
        return logits

    def get_action(self, state, deterministic=False):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        if deterministic:
            action = probs.argmax(dim=-1)
            log_prob = None
        else:
            m = dist.Categorical(probs)
            action = m.sample()
            log_prob = m.log_prob(action)
        return action, log_prob

    def evaluate(self, state, action):
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        m = dist.Categorical(probs)
        log_prob = m.log_prob(action)
        return log_prob