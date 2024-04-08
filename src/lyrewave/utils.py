import numpy as np
import toml
import torch


def twist(arr: torch.Tensor, size: int):
    arr = arr.T.flatten()
    add = torch.zeros(size - arr.shape[0] % size)
    arr = torch.cat([arr, add])
    arr = arr.reshape(-1, size).T
    return arr


def untwist(arr: torch.Tensor, shape: list[int]):
    arr = arr.T.flatten()
    arr = arr[:np.prod(shape)]
    arr = arr.reshape(list(reversed(shape))).T
    return arr


def open_config(config_path: str) -> dict:
    try:
        with open(config_path, "r") as file:
            config = toml.load(file)
    except FileNotFoundError:
        print("The config file does not exist or the path is incorrect.")
    except Exception as e:
        print("An error occurred while loading the config file:", e)

    if 'load_model_file' in config:
        pass

    return config
