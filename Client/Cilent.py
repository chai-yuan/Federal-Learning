import torch
from torch.utils.data import Dataset


class Cilent:
    def __init__(self, data: Dataset, model: torch.nn.Module) -> None:
        self.dataset = data
        self.model = model

    def updataModel(self, model: torch.nn.Module):
        self.model = model

    def getModel(self) -> torch.nn.Module:
        return self.model
