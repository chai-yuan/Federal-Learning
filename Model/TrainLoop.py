import torch
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def trainLoop(model: torch.nn.Module, dataLoader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()

    for batch_idx, data in enumerate(dataLoader):
        (inputs, labels) = data
        print(batch_idx, labels)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = loss_fn(outputs, labels)

        loss.backward()
        optimizer.step()
