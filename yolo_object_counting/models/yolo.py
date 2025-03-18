import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, num_classes=1, num_anchors=3):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),  # Zwiększono Dropout
        )

        self.final_conv = nn.Conv2d(
            512,
            self.num_anchors * (5 + self.num_classes),
            kernel_size=1
        )

    def forward(self, x):
        x = self.features(x)
        x = self.final_conv(x)

        batch_size, _, grid_height, grid_width = x.size()
        x = x.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_height, grid_width)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x