# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import paddle
import paddle.nn as nn

from ppmat.models.common.runtime import RuntimeMixin
from ppmat.models.common.runtime import runtime_boundary

# BatchNorm semantic alignment:
# PyTorch: running = (1 - m_torch) * running + m_torch * batch.
# Paddle: running = m_paddle * running + (1 - m_paddle) * batch.
# so m_paddle = 1 - m_torch = 0.9
TORCH_BN_MOMENTUM = 0.1
PADDLE_BN_MOMENTUM = 1.0 - TORCH_BN_MOMENTUM
BN_EPSILON = 1e-5


def _kaiming_uniform_attr():
    return paddle.ParamAttr(
        initializer=nn.initializer.KaimingUniform(
            negative_slope=5**0.5,
            mode="fan_in",
            nonlinearity="leaky_relu",
        )
    )


def _uniform_attr(bound: float):
    return paddle.ParamAttr(initializer=nn.initializer.Uniform(-bound, bound))


def _bn_aligned(num_features: int) -> nn.BatchNorm2D:
    """Create BatchNorm2D with PyTorch-aligned momentum semantics."""
    return nn.BatchNorm2D(num_features, momentum=PADDLE_BN_MOMENTUM, epsilon=BN_EPSILON)


class FourierUnit(nn.Layer):
    """Fourier Unit for processing frequency domain features."""

    def __init__(self, in_channels: int, out_channels: int):
        super(FourierUnit, self).__init__()

        self.conv_layer = nn.Conv2D(
            in_channels=in_channels * 2 + 2,
            out_channels=out_channels * 2,
            kernel_size=1,
            stride=1,
            padding=0,
            bias_attr=False,
            weight_attr=_kaiming_uniform_attr(),
        )
        self.bn = _bn_aligned(out_channels * 2)
        self.relu = nn.ReLU()

    def forward(self, x):
        batch = x.shape[0]
        fft_dim = (-2, -1)

        # Real FFT with ortho normalization
        ffted = paddle.fft.rfftn(x, axes=fft_dim, norm="ortho")

        # Split into real/imaginary parts
        ffted_real = paddle.real(ffted)  # (B, C, H, W/2+1)
        ffted_imag = paddle.imag(ffted)  # (B, C, H, W/2+1)
        ffted = paddle.stack([ffted_real, ffted_imag], axis=-1)  # (B, C, H, W/2+1, 2)

        # Permute to (B, C, 2, H, W/2+1)
        ffted = ffted.transpose([0, 1, 4, 2, 3])  # (B, C, 2, H, W/2+1)
        ffted = ffted.reshape([batch, -1] + list(ffted.shape[3:]))  # (B, C*2, H, W/2+1)

        height, width = ffted.shape[-2:]
        coords_vert = paddle.linspace(0, 1, height).reshape([1, 1, height, 1])
        coords_vert = coords_vert.expand([x.shape[0], 1, height, width])

        coords_hor = paddle.linspace(0, 1, width).reshape([1, 1, 1, width])
        coords_hor = coords_hor.expand([x.shape[0], 1, height, width])

        # Concatenate coordinates and FFT features
        ffted = paddle.concat(
            [coords_vert, coords_hor, ffted], axis=1
        )  # (B, C*2+2, H, W/2+1)

        # Process through convolution
        ffted = self.conv_layer(ffted)
        ffted = self.relu(self.bn(ffted))  # (B, C*2, H, W/2+1)

        # Reshape back to complex format.
        ffted = ffted.reshape([batch, -1, 2] + list(ffted.shape[2:]))
        ffted = ffted.transpose([0, 1, 3, 4, 2])  # (B, C, H, W/2+1, 2)

        # Convert back to complex tensor
        ffted = paddle.complex(ffted[..., 0], ffted[..., 1])  # (B, C, H, W/2+1) complex

        # Inverse FFT with exact shape matching
        output = paddle.fft.irfftn(ffted, s=x.shape[-2:], axes=fft_dim, norm="ortho")
        return output


class SpectralTransform(nn.Layer):
    """Spectral Transform block combining spatial and frequency domain processing."""

    def __init__(self, in_channels: int):
        super(SpectralTransform, self).__init__()
        st1_fan_in = (in_channels // 2) * 3 * 3
        st1_bias_bound = 1.0 / st1_fan_in**0.5

        st2_fan_in = in_channels * 3 * 3
        st2_bias_bound = 1.0 / st2_fan_in**0.5

        self.conv1 = nn.Conv2D(
            in_channels // 2,
            in_channels // 2,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(st1_bias_bound),
        )
        self.fu = FourierUnit(in_channels // 2, in_channels // 2)
        self.conv2 = nn.Conv2D(
            in_channels,
            in_channels // 2,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(st2_bias_bound),
        )

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.fu(x1)
        x = self.conv2(paddle.concat([x, x2], axis=1))
        return x


class FFC(nn.Layer):
    """Fast Fourier Convolution block for spatial-frequency interaction."""

    def __init__(self, in_channels: int):
        super(FFC, self).__init__()
        ffc_fan_in = (in_channels // 2) * 3 * 3
        ffc_bias_bound = 1.0 / ffc_fan_in**0.5

        self.convl2l = nn.Conv2D(
            in_channels // 2,
            in_channels // 2,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(ffc_bias_bound),
        )
        self.convl2g = nn.Conv2D(
            in_channels // 2,
            in_channels // 2,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(ffc_bias_bound),
        )
        self.convg2l = nn.Conv2D(
            in_channels // 2,
            in_channels // 2,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(ffc_bias_bound),
        )
        self.convg2g = SpectralTransform(in_channels)

    def forward(self, x):
        if isinstance(x, tuple):
            x_l, x_g = x
        else:
            C = x.shape[1]
            x_l, x_g = paddle.split(x, [C // 2, C // 2], axis=1)

        out_xl = self.convl2l(x_l) + self.convg2l(x_g)
        out_xg = self.convl2g(x_l) + self.convg2g(x_g)
        return out_xl, out_xg


class SFIB(nn.Layer):
    """Spatial-Frequency Interactive Block."""

    def __init__(self, in_channels: int):
        super(SFIB, self).__init__()
        self.ffc = FFC(in_channels)
        self.bn_l = _bn_aligned(in_channels // 2)
        self.bn_g = _bn_aligned(in_channels // 2)
        self.act_l = nn.ReLU()
        self.act_g = nn.ReLU()

    def forward(self, x):
        x_l, x_g = self.ffc(x)
        x_l = self.act_l(self.bn_l(x_l))
        x_g = self.act_g(self.bn_g(x_g))
        return x_l, x_g


class ResnetBlock(nn.Layer):
    """Residual block with SFIB."""

    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.conv1 = SFIB(in_channels)
        self.conv2 = SFIB(in_channels)

    def forward(self, x):
        x_l, x_g = paddle.split(
            x, [self.in_channels // 2, self.in_channels // 2], axis=1
        )
        id_l, id_g = x_l, x_g
        x_l, x_g = self.conv1((x_l, x_g))
        x_l, x_g = self.conv2((x_l, x_g))

        x_l = id_l + x_l
        x_g = id_g + x_g

        out = paddle.concat([x_l, x_g], axis=1)
        return out


class SFIN(RuntimeMixin, nn.Layer):
    """
    SFIN: Noise Calibration and Spatial-Frequency Interactive Network for STEM
    Image Enhancement.

    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale images)
        base_channels (int): Base number of channels (default: 64)
        num_blocks (int): Number of ResNet blocks (default: 8)
        input_name (str): Dataset input key for noisy STEM images.
        target_name (str): Dataset target key and prediction key.
        loss_type (str): Loss function type, either ``l1`` or ``mse``.
        loss_weight (float): Loss scaling weight.

    Reference:
        Li et al., "Noise Calibration and Spatial-Frequency Interactive Network for
        STEM Image Enhancement", CVPR 2025.
        https://arxiv.org/pdf/2504.02555
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 8,
        input_name: str = "noisy",
        target_name: str = "gt_enhance",
        loss_type: str = "l1",
        loss_weight: float = 1.0,
        execution_backend: str = "eager",
        runtime_options: dict | None = None,
    ):
        super(SFIN, self).__init__()
        self._init_runtime(execution_backend, runtime_options)
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_blocks = num_blocks
        self.input_name = input_name
        self.target_name = target_name
        self.loss_type = loss_type.lower()
        self.loss_weight = loss_weight

        if self.loss_type == "l1":
            self.criterion = nn.L1Loss()
        elif self.loss_type == "mse":
            self.criterion = nn.MSELoss()
        else:
            raise ValueError(
                f"Unsupported loss_type '{loss_type}', expected 'l1' or 'mse'."
            )

        blocks = [ResnetBlock(base_channels) for _ in range(num_blocks)]
        self.body = nn.Sequential(*blocks)

        # Head convolution initialization
        head_fan_in = in_channels * 3 * 3
        head_bias_bound = 1.0 / head_fan_in**0.5
        self.head_conv = nn.Conv2D(
            in_channels,
            base_channels,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(head_bias_bound),
        )

        # Tail convolution initialization
        tail_fan_in = base_channels * 3 * 3
        tail_bias_bound = 1.0 / tail_fan_in**0.5
        self.tail_conv = nn.Conv2D(
            base_channels,
            in_channels,
            3,
            padding=1,
            weight_attr=_kaiming_uniform_attr(),
            bias_attr=_uniform_attr(tail_bias_bound),
        )

    @runtime_boundary("forward")
    def _forward(self, x: paddle.Tensor) -> paddle.Tensor:
        """
        Tensor-only forward pass of SFIN.

        Args:
            x: Input tensor of shape (B, C, H, W)

        Returns:
            Enhanced image tensor of shape (B, C, H, W)
        """
        x = self.head_conv(x)
        shortcut = x
        x = self.body(x)
        return self.tail_conv(x + shortcut)

    @staticmethod
    def _to_tensor(data) -> paddle.Tensor:
        """Convert image data to a batched Paddle tensor."""
        if not isinstance(data, paddle.Tensor):
            data = paddle.to_tensor(data)
        if data.ndim == 3:
            data = data.unsqueeze(0)
        if data.ndim != 4:
            raise ValueError(
                "SFIN expects image data with shape [C, H, W] or "
                f"[B, C, H, W], but got {list(data.shape)}."
            )
        return data.astype(paddle.get_default_dtype())

    def forward(self, batch):
        """
        Unified forward for both:
        1) tensor -> enhanced tensor (for direct use / legacy scripts)
        2) dict -> trainer-ready output with loss_dict and pred_dict
        """
        if isinstance(batch, dict):
            if self.input_name not in batch:
                raise KeyError(
                    f"SFIN expects '{self.input_name}' in batch, but got keys: "
                    f"{list(batch.keys())}"
                )
            if self.target_name not in batch:
                raise KeyError(
                    f"SFIN expects '{self.target_name}' in batch, but got keys: "
                    f"{list(batch.keys())}"
                )

            x = self._to_tensor(batch[self.input_name])
            enhanced = self._forward(x)

            pred_dict = {
                self.target_name: enhanced,
            }
            label = self._to_tensor(batch[self.target_name])
            loss = self.criterion(enhanced, label) * self.loss_weight
            loss_dict = {"loss": loss}

            return {"loss_dict": loss_dict, "pred_dict": pred_dict}

        return self._forward(self._to_tensor(batch))

    @paddle.no_grad()
    def predict(self, samples):
        is_list = isinstance(samples, list)
        samples = samples if is_list else [samples]

        results = []
        for sample in samples:
            sample = self._to_tensor(sample)
            enhanced = self._forward(sample)
            results.append({self.target_name: enhanced})

        return results if is_list else results[0]
