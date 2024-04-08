import torch
import toml

from dataloader import get_dataloaders, AudioDataset
from diffusion import Diffusion
from model import UNet
from utils import open_config
from train import train_network


def run(config_path: str):
    config = open_config(config_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    diffusion = Diffusion(device=device, **config)
    trainloader, testloader = get_dataloaders(
        diffusion=diffusion, device=device, **config
    )

    model = UNet(label_count=config['classes'], **config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = torch.nn.MSELoss()

    model, optimizer = train_network(
        model, optimizer, criterion, trainloader, testloader, config['num_epochs']
    )


def main():
    run('config.toml')


if __name__ == '__main__':
    main()
