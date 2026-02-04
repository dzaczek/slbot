import torch
import torch.nn as nn
import torch.nn.functional as F
import os

class DDQN(nn.Module):
    def __init__(self, output_dim=3):
        super(DDQN, self).__init__()

        # Grid Processing (CNN)
        # Input: 1 channel (values -1 to 1), 20x20 grid
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) # -> 32x20x20
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1) # -> 64x20x20
        # We can add pooling to reduce parameters, but 20x20 is small.
        # Let's do one pooling layer after conv2 or conv1.
        # If we do MaxPool(2) after conv2: -> 64x10x10
        self.pool = nn.MaxPool2d(2, 2)

        # Flatten size calculation:
        # 64 channels * 10 height * 10 width = 6400
        self.flatten_dim = 64 * 10 * 10

        # Orientation Vector Processing
        # Input: 4 values
        self.orient_dim = 4

        # Fusion (Fully Connected)
        self.fc1 = nn.Linear(self.flatten_dim + self.orient_dim, 512)
        self.fc2 = nn.Linear(512, output_dim)

    def forward(self, grid, orientation):
        # grid shape: (batch, 1, 20, 20)
        # orientation shape: (batch, 4)

        # CNN Path
        x = F.relu(self.conv1(grid))
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, self.flatten_dim) # Flatten

        # Concatenate with Orientation
        x = torch.cat((x, orientation), dim=1)

        # FC Path
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def save(self, file_name='model.pth'):
        model_folder_path = './model'
        if not os.path.exists(model_folder_path):
            os.makedirs(model_folder_path)

        file_name = os.path.join(model_folder_path, file_name)
        torch.save(self.state_dict(), file_name)
