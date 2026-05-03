import torch.nn as nn
import torch

class TerrariaBot(nn.Module):
    def __init__(self, action_dim):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2),
            nn.ReLU(),

            nn.Conv2d(32, 64, 5, stride=2),
            nn.ReLU(),

            nn.Conv2d(64, 128, 5, stride=2),
            nn.ReLU(),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 160, 90)
            conv_out = self.conv(dummy)
            self.flatten_size = conv_out.view(1, -1).size(1)

        self.fc = nn.Sequential(
            nn.Linear(128 * 18 * 10, 256),
            nn.ReLU(), 

            nn.Linear(256, action_dim),
            nn.Sigmoid(), 
        )

    def forward(self, x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        return self.fc(x)