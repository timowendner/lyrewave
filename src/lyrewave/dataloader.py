import os
import glob
import torch
import torchaudio
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split


from .diffusion import Diffusion
from .utils import twist


def get_dataloaders(
    data_path: str,
    label_train: list[str],
    data_shape: list[int],
    data_twist: int,
    train_split: float,
    batch_size: int,
    diffusion: Diffusion,
    device: torch.device,
    data_on_device: bool = False,
    **kwargs,
):
    data_paths = []
    for label in label_train:
        data_paths.append(os.path.join(data_path, label))

    dataset = AudioDataset(
        data_paths, data_shape, data_twist, diffusion, device, data_on_device
    )

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_split)
    test_size = dataset_size - train_size

    traindata, testdata = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(traindata, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testdata, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


class AudioDataset(Dataset):
    def __init__(
        self,
        data_paths: list[str],
        data_shape: list[int],
        data_twist: int,
        diffusion: Diffusion,
        device: torch.device,
        data_on_device: bool = False
    ) -> None:
        dataset = []
        for label, dir_path in enumerate(data_paths):
            files = glob.glob(os.path.join(dir_path, '*.wav'))
            for path in files:
                waveform, sr = torchaudio.load(path)
                waveform = waveform * 0.98 / torch.max(waveform)
                waveform = twist(waveform, data_twist)
                if data_on_device:
                    waveform = waveform.to(device)
                dataset.append((waveform, label))

        if len(dataset) == 0:
            raise AttributeError('Data-path seems to be empty.')

        self.dataset = dataset
        self.device = device
        self.diffusion = diffusion
        self.data_shape = data_shape

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        waveform, label = self.dataset[idx]
        waveform = waveform.to(self.device)

        # apply gain and rollover
        waveform = waveform * (1 - np.random.normal(0, 0.15) ** 2)
        # waveform = torch.roll(waveform, np.random.randint(waveform.shape[0]))

        # diffusion
        max_timestamp = self.diffusion.steps
        timestamp = np.random.randint(1, max_timestamp)
        x_t, noise = self.diffusion(waveform, timestamp)

        timestamp = torch.Tensor([timestamp]).long().to(self.device)
        label = torch.Tensor([label]).long().to(self.device)
        x_t = x_t.to(self.device)
        noise = noise.to(self.device)

        return x_t, noise, timestamp, label
