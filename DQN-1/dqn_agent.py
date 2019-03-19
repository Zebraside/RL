import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):
        """A simple DQN agent"""
        nn.Module.__init__(self)
        self.epsilon = epsilon
        self.n_actions = n_actions
        img_c, img_w, img_h = state_shape
        # Define your network body here. Please make sure agent is fully contained here
        self.c1 = nn.Conv2d(4, 16, kernel_size=3)
        self.c2 = nn.Conv2d(16, 32, kernel_size=3)
        self.c3 = nn.Conv2d(32, 64, kernel_size=3)
        self.fc1 = nn.Linear(215296, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

    def forward(self, state_t):
        """
        takes agent's observation (Variable), returns qvalues (Variable)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        Hint: if you're running on GPU, use state_t.cuda() right here.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        state_t.to(device)
        state_t = F.relu(self.c1(state_t))#F.max_pool2d(F.relu(self.c1(state_t)), 2)
        state_t = F.relu(self.c2(state_t))#F.max_pool2d(F.relu(self.c2(state_t)), 2)
        state_t = F.relu(self.c3(state_t))#F.max_pool2d(F.relu(self.c3(state_t)), 2)
        size = state_t.size()[1:]
        num_f = 1
        for s in size:
            num_f *= s
        state_t = state_t.view(-1, num_f)
        state_t = F.relu(self.fc1(state_t))
        state_t = F.relu(self.fc2(state_t))
        qvalues = state_t

        assert isinstance(qvalues, Variable) and qvalues.requires_grad, "qvalues must be a torch variable with grad"
        assert len(qvalues.shape) == 2
        assert qvalues.shape[0] == state_t.shape[0] 
        assert qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not Variables
        """
        states = Variable(torch.FloatTensor(np.asarray(states)))
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape
        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)
        should_explore = np.random.choice([0, 1], batch_size, p=[1 - epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)