import torch
import torch.nn as nn

class YOLO(nn.Module):
    def __init__(self, num_classes=1, num_anchors=8):
        super(YOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        # Warstwy ekstrakcji cech
        self.features_early = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 208x208
        )

        self.features = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 104x104
        )

        self.features_mid = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 52x52
        )

        self.features_mid2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2, 2),  # 26x26
        )

        self.features_deep = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Dropout(0.4),
            nn.MaxPool2d(2, 2),  # 13x13
        )

        # Warstwy predykcji dla różnych skal
        self.final_conv_104 = nn.Conv2d(32, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.final_conv_52 = nn.Conv2d(64, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.final_conv_26 = nn.Conv2d(128, self.num_anchors * (5 + self.num_classes), kernel_size=1)
        self.final_conv_13 = nn.Conv2d(256, self.num_anchors * (5 + self.num_classes), kernel_size=1)

    def forward(self, x):
        # Ekstrakcja cech na różnych skalach
        x_208 = self.features_early(x)  # 208x208
        x_104 = self.features(x_208)  # 104x104
        x_52 = self.features_mid(x_104)  # 52x52
        x_26 = self.features_mid2(x_52)  # 26x26
        x_13 = self.features_deep(x_26)  # 13x13

        # Predykcje na różnych skalach
        out_104 = self.final_conv_104(x_104)
        out_52 = self.final_conv_52(x_52)
        out_26 = self.final_conv_26(x_26)
        out_13 = self.final_conv_13(x_13)

        # Reshape dla każdej skali
        batch_size, _, grid_height, grid_width = out_104.size()
        out_104 = out_104.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_height, grid_width)
        out_104 = out_104.permute(0, 3, 4, 1, 2).contiguous()
        out_104[..., 0:2] = torch.sigmoid(out_104[..., 0:2])  # Normalizacja x, y do [0, 1]

        batch_size, _, grid_height, grid_width = out_52.size()
        out_52 = out_52.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_height, grid_width)
        out_52 = out_52.permute(0, 3, 4, 1, 2).contiguous()
        out_52[..., 0:2] = torch.sigmoid(out_52[..., 0:2])

        batch_size, _, grid_height, grid_width = out_26.size()
        out_26 = out_26.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_height, grid_width)
        out_26 = out_26.permute(0, 3, 4, 1, 2).contiguous()
        out_26[..., 0:2] = torch.sigmoid(out_26[..., 0:2])

        batch_size, _, grid_height, grid_width = out_13.size()
        out_13 = out_13.view(batch_size, self.num_anchors, 5 + self.num_classes, grid_height, grid_width)
        out_13 = out_13.permute(0, 3, 4, 1, 2).contiguous()
        out_13[..., 0:2] = torch.sigmoid(out_13[..., 0:2])

        return [out_104, out_52, out_26, out_13]