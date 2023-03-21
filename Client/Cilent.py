import torch
from torch.utils.data import Dataset


class Cilent:
    def __init__(self, data: Dataset) -> None:
        self.dataset = data
        self.model = None

    def updataCilentModel(self, model: torch.nn.Module):
        self.model = model

    def getModel(self) -> torch.nn.Module:
        return self.model
