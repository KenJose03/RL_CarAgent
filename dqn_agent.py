import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

class DQNAgent:
    def __init__(self, state_size, action_size, lr=0.001, gamma=0.99, epsilon=1.0,
                 epsilon_min=0.01, epsilon_decay=0.995, batch_size=32, memory_size=10000):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.model = DQNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.model(state)
        return torch.argmax(q_values).item()

    def replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states))
        next_states = torch.FloatTensor(np.array(next_states))
        actions = torch.LongTensor(actions).unsqueeze(1)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        q_vals = self.model(states).gather(1, actions)
        with torch.no_grad():
            next_q_vals = self.model(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (1 - dones) * self.gamma * next_q_vals

        loss = self.loss_fn(q_vals, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, filepath):
        """Save the model weights to a file."""
        torch.save(self.model.state_dict(), filepath)

    def load(self, filepath):
        """Load the model weights from a file."""
        self.model.load_state_dict(torch.load(filepath))
        self.model.eval()
    def set_lidar_bias(self):
        # Assume state: [L1, L2, L3, L4, L5, ..., R1, R2, R3, R4, R5] (left to right)
        # For 5 rays, each with 3 features: [dist_lane, dist_edge, nearest] * 5
        # We'll use only the 'nearest' feature for biasing (indices 2, 5, 8, 11, 14)
        # Action 0: left, 1: right, 2: no steering
        with torch.no_grad():
            w = self.model.fc1.weight
            # Steer right if left sensors (first two rays) are close
            w[0, 2] = -2.0  # leftmost 'nearest' feature, left action
            w[1, 2] = 2.0   # leftmost 'nearest' feature, right action
            w[0, 5] = -1.0  # next left 'nearest' feature, left action
            w[1, 5] = 1.0   # next left 'nearest' feature, right action
            # Steer left if right sensors (last two rays) are close
            w[0, 11] = 2.0  # rightmost 'nearest' feature, left action
            w[1, 11] = -2.0 # rightmost 'nearest' feature, right action
            w[0, 8] = 1.0   # next right 'nearest' feature, left action
            w[1, 8] = -1.0  # next right 'nearest' feature, right action
            # Center ray: encourage no steering if clear
            w[2, 8] = 2.0   # center 'nearest' feature, no steering
# Update the input dimension of the first linear layer from 5 to 15
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(15, 64)  # changed input dim from 5 to 15
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
 
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
