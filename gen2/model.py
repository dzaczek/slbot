import torch
import torch.nn as nn
import torch.nn.functional as F

class MatrixPolicy(nn.Module):
    def __init__(self, action_dim=6):
        super(MatrixPolicy, self).__init__()

        # Input: 3 Channels (Food, Enemies, Self), 64x64 Grid
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4) # -> 32 x 15 x 15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # -> 64 x 6 x 6
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # -> 64 x 4 x 4

        flat_size = 64 * 4 * 4 # 1024

        self.fc1 = nn.Linear(flat_size, 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        # x shape: (B, 3, 64, 64)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.reshape(x.size(0), -1) # Flatten
        x = F.relu(self.fc1(x))
        return self.fc2(x)
