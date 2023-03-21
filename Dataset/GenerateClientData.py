import os
from torchvision import transforms
from SkinDataset import SkinDataset
from torch.utils.data import Subset

# 定义转换函数
transform = transforms.Compose([
    transforms.Resize(256),  # 调整图像大小为256x256
    transforms.ToTensor(),  # 转换为张量
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))  # 标准化
])


def generateClientData(dataPath: str, nums: int) -> list:
    dataset = SkinDataset(transform=transform, dataPath=dataPath)
    dataLen = len(dataset)
    dataStep = int(dataLen / nums)

    ret = []
    for i in range(nums):   # 划分为nums份
        ret.append(Subset(dataset=dataset, indices=range(
            dataStep*i, dataStep*(i+1))))

    return ret
