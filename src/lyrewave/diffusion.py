import numpy as np
import torch
from torch import nn, sqrt


class Diffusion(nn.Module):
    def __init__(
        self,
        step_count: int,
        data_shape: list[int],
        beta_start: float,
        beta_end: float,
        device: torch.device,
        **kwargs,
    ) -> None:
        super().__init__()
        self.steps = step_count
        self.length = data_shape[1]

        start = beta_start
        end = beta_end

        t = torch.linspace(0, 1, self.steps, device=device)
        self.beta = start + (end - start) * t**2

        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

    def forward(self, x, t):
        noise = torch.randn_like(x)
        alpha_hat_t = self.alpha_hat[t]
        x_t = sqrt(alpha_hat_t) * x + sqrt(1 - alpha_hat_t) * noise

        return x_t, noise

    @torch.no_grad()
    def sample(self, model: nn.Module, create_label: int):
        n = 1
        labels = [create_label] * n
        model.eval()

        x = torch.randn(n, 1, self.length, device=self.device)
        l = torch.tensor(labels, device=self.device).view(-1, 1)

        for i in range(1, self.steps):
            t = torch.ones(n, device=self.device).long() * (self.steps - i)
            alpha = self.alpha[t].view(-1, 1, 1)
            alpha_hat = self.alpha_hat[t].view(-1, 1, 1)
            beta = self.beta[t].view(-1, 1, 1)

            # predict the added noise
            timestamp = torch.ones(n, device=self.device) * (self.steps - i)
            timestamp = timestamp.view(-1, 1)
            predicted_noise = model(x, timestamp, l)

            # remove noise
            if i == self.steps - 1:
                noise = torch.zeros_like(x)
            else:
                noise = torch.randn_like(x)
            x = 1 / sqrt(alpha) * (x - (1 - alpha) / (sqrt(1 - alpha_hat))
                                   * predicted_noise) + sqrt(beta) * noise

        model.train()
        return x
