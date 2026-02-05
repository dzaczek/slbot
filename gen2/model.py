import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixPolicy(nn.Module):
    """
    Dueling DQN architecture with larger network.
    Separates value estimation (how good is this state?) from 
    advantage estimation (how much better is each action?).
    """
    def __init__(self, action_dim=6):
        super(MatrixPolicy, self).__init__()

        # Input: 3 Channels (Food, Enemies, Self), 64x64 Grid
        # Increased filters for better feature extraction
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)   # -> 32 x 15 x 15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  # -> 64 x 6 x 6
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1) # -> 128 x 4 x 4 (was 64)

        flat_size = 128 * 4 * 4  # 2048 (was 1024)

        # Shared feature layer
        self.fc_shared = nn.Linear(flat_size, 512)
        
        # Dueling streams
        # Value stream: estimates V(s) - how good is this state?
        self.fc_value1 = nn.Linear(512, 256)
        self.fc_value2 = nn.Linear(256, 1)
        
        # Advantage stream: estimates A(s,a) - how much better is each action?
        self.fc_advantage1 = nn.Linear(512, 256)
        self.fc_advantage2 = nn.Linear(256, action_dim)

    def forward(self, x):
        # x shape: (B, 3, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1)  # Flatten to (B, 2048)
        x = F.relu(self.fc_shared(x))
        
        # Value stream
        value = F.relu(self.fc_value1(x))
        value = self.fc_value2(value)  # (B, 1)
        
        # Advantage stream
        advantage = F.relu(self.fc_advantage1(x))
        advantage = self.fc_advantage2(advantage)  # (B, action_dim)
        
        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        # Subtracting mean ensures identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
