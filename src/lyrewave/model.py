import numpy as np
import torch
from torch import nn, Tensor


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
        step_count: int,
        model_scale: int,
        model_kernel: int,
        model_layers: list[int],
        model_block_size: int,
        model_dropout: float,
        classes: int,
        **kwargs,
    ) -> None:
        super().__init__()

        length = int(np.ceil(data_shape[1] / data_twist))
        scale = model_scale
        kernel = model_kernel

        embeddings = [UNetEmbedding(step_count, classes, data_twist)]
        down_blocks = [UNetResBlock(
            data_twist, model_layers[0], kernel, embeddings[0], model_block_size, model_dropout
        )]
        up_blocks = []

        channels = [*model_layers, model_layers[-1]]
        for prev, channel in zip(channels, channels[1:]):
            embedding = UNetEmbedding(step_count, classes, prev)
            length, pad = divmod(length, scale)
            down_blocks.append(UNetDownBlock(
                prev, channel, kernel, model_block_size, embedding, scale, model_dropout
            ))
            up_blocks.append(UNetUpBlock(
                channel, prev, kernel, model_block_size, embedding, scale, model_dropout, pad
            ))
            embeddings.append(embedding)
        self.embedding = nn.ModuleList(embeddings)
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
