import torch
import copy
from torch.utils.data import DataLoader
import torch.optim as optim

from Client.Cilent import Cilent
from Server.Server import Server
from Dataset.SkinDataset import SkinDataset
from Model.SkinModel import SkinModel

from Dataset.GenerateClientData import *
from Model.TrainLoop import *
from Model.TestLoop import *

cilentNum = 5
trainEpoch = 1
batchSize = 4
learningRate = 0.01

if __name__ == "__main__":
    trainDataList = generateClientData("./Dataset/train/", cilentNum)

    # 生成一系列客户端和单个服务端
    cilentList = [Cilent(trainDataList[i]) for i in range(cilentNum)]
    server = Server(SkinDataset(transform=transform,
                    dataPath="./Dataset/test/"), SkinModel())

    # 分发模型
    for cilent in cilentList:
        cilent.updataCilentModel(copy.copy(server.getModel()))

    # 客户端训练
    for i in range(trainEpoch):
        for cilent in cilentList:
            dataLoader = DataLoader(
                cilent.dataset, batch_size=batchSize, shuffle=True, num_workers=6, pin_memory=True)

            optimizer = optim.SGD(cilent.model.parameters(),
                                  lr=learningRate, momentum=0.85)

            loss_fn = torch.nn.CrossEntropyLoss()

            trainLoop(cilent.model, dataLoader, optimizer, loss_fn)

    # 模型合并
    server.updataServerModel([cilent.model for cilent in cilentList])
