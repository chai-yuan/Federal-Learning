import torch
import torchvision


class SkinModel(torch.nn.Module):
    def __init__(self) -> None:
        super(SkinModel, self).__init__()
        self.cnn = torchvision.models.vgg11()
        self.cnn.classifier = torch.nn.Sequential(
            torch.nn.Linear(25088, 4096, bias=True),
            torch.nn.ReLU(True),
            torch.nn.Dropout(),
            torch.nn.Linear(4096, 2, bias=True))

    def forward(self, x):
        return self.cnn(x)


if __name__ == "__main__":
    test = SkinModel()
