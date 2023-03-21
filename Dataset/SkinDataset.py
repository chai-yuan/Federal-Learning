from torchvision import datasets, transforms
from torch.utils.data import Dataset


# 数据集定义
class SkinDataset(Dataset):
    def __init__(self,  transform, dataPath="./",) -> None:
        super().__init__()
        # 加载训练集和验证集
        self.data = datasets.ImageFolder(dataPath, transform=transform)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index=index)
