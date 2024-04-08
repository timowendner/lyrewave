import numpy as np
import torch
from torch import nn, Tensor, sin, cos, pow


def dual(in_channel: int, out_channel: int, kernel=9):
    return nn.Sequential(
        nn.Conv1d(in_channel, out_channel, kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(out_channel, out_channel, kernel, padding=kernel//2),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


def up(in_channel, out_channel, scale=2, kernel=9, pad=0):
    return nn.Sequential(
        nn.Conv1d(in_channel, in_channel, kernel, padding=kernel//2),
        nn.Dropout1d(p=0.2),
        nn.ReLU(),
        nn.Conv1d(in_channel, out_channel, kernel, padding=kernel//2),
        nn.ReLU(),
        nn.ConvTranspose1d(
            out_channel, out_channel, scale, stride=scale, output_padding=pad
        ),
        nn.BatchNorm1d(out_channel),
        nn.ReLU(),
    )


class UNetEmbedding(nn.Module):
    def __init__(self, step_count: int, label_count: int, embedding_dim: int):
        super().__init__()
        self.step_emb = nn.Embedding(step_count, embedding_dim)
        self.label_emb = nn.Embedding(label_count, embedding_dim)

    def forward(self, x: Tensor, step: Tensor, label: Tensor):
        step_emb = self.step_emb(step).squeeze(1).unsqueeze(2)
        step_emb = step_emb.repeat(1, 1, x.size(-1))

        label_emb = self.label_emb(label).squeeze(1).unsqueeze(2)
        label_emb = label_emb.repeat(1, 1, x.size(-1))
        return x + step_emb + label_emb


class UNetResBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int,
        embedding: UNetEmbedding,
        block_size: int,
        dropout: float = 0.0
    ):
        super().__init__()
        self.out_channel = out_channel
        self.embedding = embedding
        self.rescale = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, kernel, padding=kernel//2),
            nn.ReLU(),
        )
        residual = []
        for i in range(block_size):
            residual.extend([
                nn.Conv1d(out_channel, out_channel, kernel, padding=kernel//2),
                nn.Dropout1d(p=dropout),
                nn.ReLU()
            ])
        self.residual = nn.Sequential(*residual)

    def forward(self, x: Tensor, step: Tensor, label: Tensor, x_prev: Tensor = None):
        x = self.embedding(x, step, label)
        if x_prev is not None:
            x = torch.cat([x, x_prev], dim=1)

        x = self.rescale(x)
        x = self.residual(x) + x
        return x


class UNetUpBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int,
        block_size: int,
        embedding: UNetEmbedding,
        scale: int,
        dropout: float = 0.0,
        pad: int = 0
    ) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.ConvTranspose1d(
                in_channel, out_channel, scale, stride=scale, output_padding=pad
            ),
            nn.ReLU()
        )
        self.res_block = UNetResBlock(
            out_channel*2, out_channel, kernel, embedding, block_size, dropout
        )

    def forward(self, x: Tensor, step: Tensor, label: Tensor, x_prev: Tensor):
        x = self.up(x)
        x = self.res_block(x, step, label, x_prev)
        return x


class UNetDownBlock(nn.Module):
    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel: int,
        block_size: int,
        embedding: UNetEmbedding,
        scale: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.pool = nn.MaxPool1d(scale, stride=scale)
        self.res_block = UNetResBlock(
            in_channel, out_channel, kernel, embedding, block_size, dropout
        )

    def forward(self, x: Tensor, step: Tensor, label: Tensor):
        x = self.pool(x)
        x = self.res_block(x, step, label)
        return x


class UNet(nn.Module):
    def __init__(
        self,
        data_shape: list[int],
        data_twist: int,
        label_count: int,
        step_count: int,
        model_scale: int,
        model_kernel: int,
        model_layers: list[int],
        model_block_size: int,
        model_dropout: float,
        **kwargs,
    ) -> None:
        super().__init__()

        length = int(np.ceil(data_shape[1] / data_twist))
        scale = model_scale
        kernel = model_kernel

        first_emb = UNetEmbedding(step_count, label_count, data_twist)
        down_blocks = [UNetResBlock(
            data_twist, model_layers[0], kernel, first_emb, model_block_size, model_dropout
        )]
        up_blocks = []

        channels = [*model_layers, model_layers[-1]]
        for prev, channel in zip(channels, channels[1:]):
            embedding = UNetEmbedding(step_count, label_count, prev)
            length, pad = divmod(length, scale)
            down_blocks.append(UNetDownBlock(
                prev, channel, kernel, model_block_size, embedding, scale, model_dropout
            ))
            up_blocks.append(UNetUpBlock(
                channel, prev, kernel, model_block_size, embedding, scale, model_dropout, pad
            ))
        self.output = nn.Conv1d(
            model_layers[0], data_twist, kernel, padding=kernel//2
        )
        self.down_blocks = nn.ModuleList(down_blocks)
        self.up_blocks = nn.ModuleList(reversed(up_blocks))

    def forward(self, x: Tensor, step: Tensor, label: Tensor) -> Tensor:
        encoder = []
        for down_block in self.down_blocks:
            x = down_block(x, step, label)
            encoder.append(x)

        encoder = reversed(encoder[:-1])
        for x_prev, up_block in zip(encoder, self.up_blocks):
            x = up_block(x, step, label, x_prev)
        x = self.output(x)
        return x


class UNet_old(nn.Module):
    def __init__(
        self,
        data_shape: list[int],
        model_scale: int,
        model_kernel: int,
        model_layers: list[int],
        model_out: list[int],
        device: torch.device,
        **kwargs
    ) -> None:
        super().__init__()

        # define the pooling layer
        length = data_shape[1]
        scale = model_scale
        kernel = model_kernel
        self.pool = nn.MaxPool1d(scale, stride=scale)
        self.device = device

        # define the encoder
        last = data_shape[0]
        pad = []
        self.length = [length]
        self.down = nn.ModuleList([])
        for channel in model_layers:
            cur_pad, length = length % scale, length // scale
            self.length.append(length)
            pad.append(cur_pad)
            layer = dual(last+2, channel, kernel=kernel)
            self.down.append(layer)
            last = channel

        # define the decoder
        self.up = nn.ModuleList([])
        for channel in reversed(model_layers):
            layer = up(last+2, channel, scale=scale,
                       kernel=kernel, pad=pad.pop())
            self.up.append(layer)
            last = channel * 2

        # define the output layer
        output = []
        last += 2
        for channel in model_out:
            output.append(
                nn.Conv1d(last, channel, kernel, padding=kernel//2)
            )
            output.append(nn.ReLU())
            last = channel
        output.append(
            nn.Conv1d(last, data_shape[0], kernel, padding=kernel//2)
        )
        self.output = nn.Sequential(*output)

    def forward(self, x: Tensor, timestamp: Tensor, label: Tensor) -> Tensor:
        timestamp = timestamp.to(self.device)
        label = (label * 100).to(self.device)

        # apply the encoder
        encoder = []
        length = [i for i in reversed(self.length)]
        for layer in self.down:
            t = self.sinusoidal(timestamp, length[-1])
            l = self.sinusoidal(label, length.pop())
            x = torch.cat([l, t, x], 1)
            x = layer(x)
            encoder.append(torch.cat([l, t, x], 1))
            x = self.pool(x)

        # apply the decoder
        t = self.sinusoidal(timestamp, length[-1])
        l = self.sinusoidal(label, length.pop())
        x = torch.cat([l, t, x], 1)
        for layer in self.up:
            x = layer(x)
            x = torch.cat([encoder.pop(), x], 1)

        # apply the output
        x = self.output(x)
        return x
