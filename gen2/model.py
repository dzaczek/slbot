import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    Separates value estimation (how good is this state?) from 
    advantage estimation (how much better is each action?).
    """
    def __init__(self, input_channels, action_dim=6):
        super(DuelingDQN, self).__init__()

        # Input: (input_channels, 64, 64)
        # Using 3 layers of convolutions as per Nature paper, adapted for 64x64
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4) # -> 32 x 15 x 15
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)             # -> 64 x 6 x 6
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)             # -> 64 x 4 x 4

        # Calculate flat size dynamically
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 84, 84)
            x = self.conv1(dummy_input)
            x = self.conv2(x)
            x = self.conv3(x)
            flat_size = x.view(1, -1).size(1)

        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, 1)
        )

        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(flat_size, 512),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(512, action_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x shape: (B, C, 64, 64)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)

        x = x.reshape(x.size(0), -1)  # Flatten

        value = self.value_stream(x)       # (B, 1)
        advantage = self.advantage_stream(x) # (B, action_dim)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values
