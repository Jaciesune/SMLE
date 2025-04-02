import torch
import torch.nn as nn
import torch.nn.functional as F

class MCNN(nn.Module):
    def __init__(self):
        super(MCNN, self).__init__()

        # 🔹 Trzy ścieżki konwolucyjne o różnych rozmiarach kerneli
        self.branch1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=7, stride=1, padding=3),
            nn.ReLU()
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU()
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        # 🔹 Połączenie ścieżek
        self.fuse = nn.Sequential(
            nn.Conv2d(96, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1)  # Wyjście mapy gęstości
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        # Połączenie wyników
        out = torch.cat((b1, b2, b3), dim=1)
        out = self.fuse(out)
        return out
