import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable


class DQNZ(nn.Module):
    def __init__(self, state_size, hidden_size, action_size, reward_size=1, qvalue_size=1, termination_size=1):
        super(DQNZ, self).__init__()

        self.linear_i2h = nn.Linear(state_size, hidden_size)

        self.linear_h2a = nn.Linear(hidden_size, action_size)
        self.linear_h2r = nn.Linear(hidden_size, reward_size)
        self.linear_h2v = nn.Linear(hidden_size, qvalue_size)
        self.linear_h2t = nn.Linear(hidden_size, termination_size)

    def forward(self, s):
        h = torch.tanh(self.linear_i2h(s))

        action = self.linear_h2a(h)
        reward = self.linear_h2r(h)
        qvalue = self.linear_h2v(h)
        term = torch.sigmoid(self.linear_h2t(h))

        return action, reward, qvalue, term

    def predict(self, s):
        h = torch.tanh(self.linear_i2h(s))

        action = self.linear_h2a(h)
        reward = self.linear_h2r(h)
        qvalue = self.linear_h2v(h)
        term = torch.sigmoid(self.linear_h2t(h))

        return torch.argmax(action, 1), reward, qvalue, term

