import torch
import torch.nn as nn
from dqn_agent import DQNetwork

# Define the old network architecture (input size 5)
class OldDQNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(OldDQNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# Load old model weights
old_state_size = 5
new_state_size = 15
action_size = 3
old_model = OldDQNetwork(old_state_size, action_size)
old_model.load_state_dict(torch.load('dqn_agent_final.pth'))

# Create new model
new_model = DQNetwork(new_state_size, action_size)
new_state_dict = new_model.state_dict()
old_state_dict = old_model.state_dict()

# Copy fc1 weights: old (64,5) -> new (64,15)
with torch.no_grad():
    # Copy first 5 columns from old weights
    new_state_dict['fc1.weight'][:, :old_state_size] = old_state_dict['fc1.weight']
    # Initialize the rest (columns 5-14) to zeros
    new_state_dict['fc1.weight'][:, old_state_size:] = 0.0
    # Copy bias
    new_state_dict['fc1.bias'] = old_state_dict['fc1.bias']
    # Copy fc2 and out layers (same shape)
    new_state_dict['fc2.weight'] = old_state_dict['fc2.weight']
    new_state_dict['fc2.bias'] = old_state_dict['fc2.bias']
    new_state_dict['out.weight'] = old_state_dict['out.weight']
    new_state_dict['out.bias'] = old_state_dict['out.bias']

# Load the new state dict into the new model
new_model.load_state_dict(new_state_dict)

# Save the new model weights
torch.save(new_model.state_dict(), 'dqn_agent_final_converted.pth')
print('Converted weights saved to dqn_agent_final_converted.pth')
