
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):

    def __init__(
        self,
        channels,
        eps=1e-6
    ):

        super().__init__()

        self.weight = nn.Parameter(
            torch.ones(channels)
        )

        self.bias = nn.Parameter(
            torch.zeros(channels)
        )

        self.eps = eps

    def forward(self, x):

        mean = x.mean(
            dim=1,
            keepdim=True
        )

        var = (
            x - mean
        ).pow(2).mean(
            dim=1,
            keepdim=True
        )

        x = (
            x - mean
        ) / torch.sqrt(
            var + self.eps
        )

        return (
            self.weight.view(
                1, -1, 1, 1
            ) * x
            +
            self.bias.view(
                1, -1, 1, 1
            )
        )


class SimpleGate(nn.Module):

    def forward(self, x):

        x1, x2 = x.chunk(
            2,
            dim=1
        )

        return x1 * x2


class NAFBlock(nn.Module):

    def __init__(
        self,
        channels,
        dw_expand=2,
        ffn_expand=2,
        dropout=0.0
    ):

        super().__init__()

        dw_channels = channels * dw_expand
        ffn_channels = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)

        self.conv1 = nn.Conv2d(
            channels,
            dw_channels,
            kernel_size=1,
            bias=True
        )

        self.dwconv = nn.Conv2d(
            dw_channels,
            dw_channels,
            kernel_size=3,
            padding=1,
            groups=dw_channels,
            bias=True
        )

        self.simple_gate = SimpleGate()

        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),

            nn.Conv2d(
                channels,
                channels,
                kernel_size=1,
                bias=True
            )
        )

        self.conv2 = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self.dropout1 = nn.Dropout(dropout)

        self.beta = nn.Parameter(
            torch.zeros(
                (1, channels, 1, 1)
            )
        )

        self.norm2 = LayerNorm2d(channels)

        self.conv3 = nn.Conv2d(
            channels,
            ffn_channels * 2,
            kernel_size=1,
            bias=True
        )

        self.simple_gate2 = SimpleGate()

        self.conv4 = nn.Conv2d(
            ffn_channels,
            channels,
            kernel_size=1,
            bias=True
        )

        self.dropout2 = nn.Dropout(dropout)

        self.gamma = nn.Parameter(
            torch.zeros(
                (1, channels, 1, 1)
            )
        )

    def forward(self, x):

        y = self.norm1(x)

        y = self.conv1(y)
        y = self.dwconv(y)
        y = self.simple_gate(y)

        y = y * self.sca(y)

        y = self.conv2(y)

        y = self.dropout1(y)

        x = x + self.beta * y

        y = self.norm2(x)

        y = self.conv3(y)
        y = self.simple_gate2(y)
        y = self.conv4(y)

        y = self.dropout2(y)

        x = x + self.gamma * y

        return x


class NAFNetSR(nn.Module):

    def __init__(
        self,
        img_channel=1,
        width=32,
        enc_blocks=(2, 2, 4),
        middle_blocks=4,
        dec_blocks=(2, 2, 2)
    ):

        super().__init__()

        self.intro = nn.Conv2d(
            img_channel,
            width,
            kernel_size=3,
            padding=1
        )

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        channels = width

        for num_blocks in enc_blocks:

            self.encoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(channels)
                        for _ in range(num_blocks)
                    ]
                )
            )

            self.downs.append(
                nn.Conv2d(
                    channels,
                    channels * 2,
                    kernel_size=2,
                    stride=2
                )
            )

            channels *= 2

        self.middle = nn.Sequential(
            *[
                NAFBlock(channels)
                for _ in range(middle_blocks)
            ]
        )

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()

        for num_blocks in dec_blocks:

            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(
                        channels,
                        channels * 2,
                        kernel_size=1
                    ),
                    nn.PixelShuffle(2)
                )
            )

            channels //= 2

            self.decoders.append(
                nn.Sequential(
                    *[
                        NAFBlock(channels)
                        for _ in range(num_blocks)
                    ]
                )
            )

        self.up2 = nn.Sequential(
            nn.Conv2d(
                width,
                width * 4,
                kernel_size=3,
                padding=1
            ),
            nn.PixelShuffle(2)
        )

        self.outro = nn.Conv2d(
            width,
            img_channel,
            kernel_size=3,
            padding=1
        )

    def forward(self, x):

        base = F.interpolate(
            x,
            scale_factor=2,
            mode="bicubic",
            align_corners=False
        )

        x = self.intro(x)

        skips = []

        for encoder, down in zip(
            self.encoders,
            self.downs
        ):

            x = encoder(x)

            skips.append(x)

            x = down(x)

        x = self.middle(x)

        for up, decoder, skip in zip(
            self.ups,
            self.decoders,
            reversed(skips)
        ):

            x = up(x)

            x = x + skip

            x = decoder(x)

        x = self.up2(x)

        residual = self.outro(x)

        return base + residual
