import torch
import torch.nn as nn
import torch.nn.functional as F

class DuelingDQN(nn.Module):
    """
    Dueling DQN architecture.
    Separates value estimation (how good is this state?) from 
    advantage estimation (how much better is each action?).
    """
    def __init__(self, input_channels, action_dim=6, input_size=(84, 84)):
        super(DuelingDQN, self).__init__()

        # Input: (input_channels, H, W)
        # Using 3 layers of convolutions as per Nature paper
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate flat size dynamically based on input_size
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, *input_size)
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
        # x shape: (B, C, H, W)
        x = F.leaky_relu(self.conv1(x), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)

        x = x.reshape(x.size(0), -1)  # Flatten

        value = self.value_stream(x)       # (B, 1)
        advantage = self.advantage_stream(x) # (B, action_dim)

        # Combine: Q(s,a) = V(s) + (A(s,a) - mean(A(s,a')))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class HybridDuelingDQN(nn.Module):
    """
    Hybrid CNN + Sector Vector architecture.
    CNN branch processes spatial matrix (food/enemies/self).
    Sector branch processes 99-float vector with precise distances + enemy approach.
    """
    def __init__(self, input_channels, action_dim=10, input_size=(64, 64), sector_dim=99):
        super(HybridDuelingDQN, self).__init__()

        # --- CNN Branch (spatial matrix) ---
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # Extra conv for 128x128: compresses 12x12 -> 5x5 with stride=2
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2)

        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, *input_size)
            x = F.leaky_relu(self.conv1(dummy), 0.01)
            x = F.leaky_relu(self.conv2(x), 0.01)
            x = F.leaky_relu(self.conv3(x), 0.01)
            x = F.leaky_relu(self.conv4(x), 0.01)
            self.cnn_flat_size = x.view(1, -1).size(1)

        # --- Sector Branch (scalar vector) ---
        self.sector_net = nn.Sequential(
            nn.Linear(sector_dim, 128),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.01),
        )
        self.sector_out_size = 128

        # --- Merge ---
        merge_size = self.cnn_flat_size + self.sector_out_size
        self.merge = nn.Sequential(
            nn.Linear(merge_size, 512),
            nn.LeakyReLU(negative_slope=0.01),
        )

        # --- Dueling heads (deeper than DuelingDQN) ---
        self.value_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.01),
            nn.Linear(256, action_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, matrix, sectors):
        # CNN branch
        x = F.leaky_relu(self.conv1(matrix), 0.01)
        x = F.leaky_relu(self.conv2(x), 0.01)
        x = F.leaky_relu(self.conv3(x), 0.01)
        x = F.leaky_relu(self.conv4(x), 0.01)
        x = x.reshape(x.size(0), -1)

        # Sector branch
        s = self.sector_net(sectors)

        # Merge
        combined = torch.cat([x, s], dim=1)
        merged = self.merge(combined)

        # Dueling
        value = self.value_stream(merged)
        advantage = self.advantage_stream(merged)
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        return q_values
