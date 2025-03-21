import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=12, kernel_size=3, stride=1, padding=1, bias=True
        )  # Passage de 28x28x1 à 28x28x12
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)  # 28x28 -> 14x14
        self.conv2 = nn.Conv2d(
            in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1, bias=True
        )  # Passage de 14x14x12 à 14x14x24
        self.fc1 = nn.Linear(7 * 7 * 24, 10)  # Couche fully connected pour 10 classes

    def forward(self, x):
        x = self.conv1(x)       # [B, 1, 28, 28] -> [B, 12, 28, 28]
        x = F.relu(x)
        x = self.pool(x)        # -> [B, 12, 14, 14]
        x = self.conv2(x)       # -> [B, 24, 14, 14]
        x = F.relu(x)
        x = self.pool(x)        # -> [B, 24, 7, 7]
        x = x.view(-1, 7 * 7 * 24)
        x = self.fc1(x)         # -> [B, 10]
        return x

# Fonction pour obtenir un modèle ResNet18 pré-entraîné
def get_pretrained_resnet(num_classes, feature_extract=True):
    model = models.resnet18(pretrained=True)
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False
    # Remplacer la dernière couche fully connected pour correspondre à num_classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model

# Modèle en ensemble qui combine ResNet et SimpleCNN
class EnsembleModel(nn.Module):
    def __init__(self, num_classes, feature_extract=True):
        super(EnsembleModel, self).__init__()
        # Branche ResNet 
        self.resnet = get_pretrained_resnet(num_classes, feature_extract)
        # Branche SimpleCNN
        self.cnn = SimpleCNN()
    
    def forward(self, x):
       
        # Passage dans la branche ResNet
        resnet_out = self.resnet(x)
        # Pour le CNN, on convertit l'image en niveaux de gris en gardant le premier canal
        x_gray = x[:, :1, :, :]
        x_gray = F.interpolate(x_gray, size=(28,28), mode='bilinear', align_corners=False)
        cnn_out = self.cnn(x_gray)
        out = (resnet_out + cnn_out) / 2
        return out

if __name__ == "__main__":
    num_classes = 6
    model = EnsembleModel(num_classes)
    print(model)

      

