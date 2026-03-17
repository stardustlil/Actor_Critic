import torch
import torch.nn as nn
import torch.nn.functional as F

class QCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        # 对于离散动作，我们输出每个动作的q值，然后根据动作索引
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_values = nn.Linear(hidden_dim, action_dim)  # 输出每个动作的q

    def forward(self, state, action=None):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q = self.q_values(x)  # shape: (batch, action_dim)
        if action is not None:
            # 如果提供了动作，根据动作索引取出对应的q值
            # action shape: (batch,) 或 scalar
            q = q.gather(1, action.long().unsqueeze(1)).squeeze(1)
        return q
    
class VCritic(nn.Module):
    def __init__(self , state_dim , hidden_dim = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim , 1)
        )
    
    def forward(self , state):
        return self.net(state).squeeze(-1)
    
