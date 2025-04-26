import torch
import torchvision.models as models
import torch.nn as nn

def ResNet50(num_classes, pretrained=True):
    weights = models.ResNet50_Weights.DEFAULT if pretrained else None
    model = models.resnet50(weights=weights)

    # Freeze the weights in the pre-trained model
    for param in model.parameters():
        param.requires_grad = False

    # Replace final fully connected layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    return model


if __name__ == '__main__':
    toy_data = torch.rand(16, 3, 256, 256)
    model = ResNet50(num_classes=12)
    output = model(toy_data)
    print(output)
    prediction = torch.argmax(output, dim=1)
    print(prediction)
