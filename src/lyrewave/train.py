import numpy as np
import datetime
import torch
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer


def train_network(
    model: nn.Module,
    optimizer: Optimizer,
    criterion: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_epoch: int = 1000,
) -> tuple[nn.Module, Optimizer]:
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    model.train()

    for epoch in range(1, num_epoch+1):
        time_now = datetime.datetime.now()
        time_now = time_now.strftime("%H:%M")
        desc = f'{time_now} Starting Epoch {epoch:>3}'
        pbar = tqdm(train_loader, desc=f'{desc:<25}', ncols=80)

        for x_t, noise, timestamp, label in pbar:
            output = model(x_t, timestamp, label)
            loss = criterion(output, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        result = test_network(model, criterion, test_loader)
    return model, optimizer


@torch.no_grad()
def test_network(
    model: nn.Module,
    criterion: nn.Module,
    dataloader: DataLoader,
) -> float:
    model.eval()
    result = []
    desc = f'{"      Testing Network":<25}'
    pbar = tqdm(dataloader, desc=desc, ncols=80)
    for i, (x_t, noise, timestamp, label) in enumerate(pbar):
        output = model(x_t, timestamp, label)
        loss = criterion(output, noise)
        result.append(loss)
        if i == len(dataloader) - 1:
            cur_desc = f'      error: {np.mean(result):.6f}'
            pbar.desc = f'{cur_desc:<25}'
    model.train()
    return np.mean(result)
