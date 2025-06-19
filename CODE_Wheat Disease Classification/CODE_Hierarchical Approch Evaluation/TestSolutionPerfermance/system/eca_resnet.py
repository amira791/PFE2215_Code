import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ECALayer(nn.Module):
    def __init__(self, channel, gamma=2, b=1):
        super().__init__()
        t = int(abs((torch.log(torch.tensor(channel, dtype=torch.float32)) / torch.log(torch.tensor(2.0))) + b) / gamma)
        k = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k, padding=(k - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = self.sigmoid(y).transpose(-1, -2).unsqueeze(-1)
        return x * y.expand_as(x)

class CropDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone.fc = nn.Identity()
        self.eca = ECALayer(channel=2048)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        x = x.view(x.size(0), 2048, 1, 1)
        x = self.eca(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        return self.fc(x)