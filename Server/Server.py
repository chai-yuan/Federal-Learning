import torch
from torch.utils.data import Dataset


class Server:
    def __init__(self, data: Dataset, model: torch.nn.Module) -> None:
        self.dataset = data
        self.model = model

    def updataServerModel(self, modelList: list):
        self.model = modelList[0]

    def getModel(self) -> torch.nn.Module:
        return self.model
