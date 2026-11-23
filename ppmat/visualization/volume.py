# Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import math
from pathlib import Path

import cvve
import numpy as np
import plotly.graph_objects as go


class VolumeVisualizer:
    """Render a CVVE scalar field and its associated atomic structure."""

    def __init__(self, max_points: int = 250_000):
        if max_points <= 0:
            raise ValueError("max_points must be positive.")
        self.max_points = int(max_points)

    def render_arrays(self, field: cvve.GridField):
        """Return display arrays and deterministic downsampling metadata."""

        if not isinstance(field, cvve.GridField):
            raise TypeError(
                f"field must be cvve.GridField, got {type(field).__name__}."
            )
        shape = field.grid.shape
        total = math.prod(shape)
        stride = max(1, math.ceil((total / self.max_points) ** (1 / 3)))
        axes = [np.arange(0, size, stride, dtype=float) for size in shape]
        indices = np.stack(
            np.meshgrid(*axes, indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)
        coordinates = field.grid.origin + indices @ field.grid.vectors
        values = field.data[::stride, ::stride, ::stride].reshape(-1)
        return coordinates, values, {
            "downsampled": stride > 1,
            "stride": stride,
            "original_points": total,
            "rendered_points": int(values.size),
        }

    def render(
        self,
        field: cvve.GridField,
        *,
        isomin: float | None = None,
        isomax: float | None = None,
        surface_count: int = 5,
        opacity: float = 0.1,
        title: str | None = None,
        show_atoms: bool = True,
    ) -> go.Figure:
        coordinates, values, _ = self.render_arrays(field)
        figure = go.Figure()
        figure.add_trace(
            go.Volume(
                x=coordinates[:, 0],
                y=coordinates[:, 1],
                z=coordinates[:, 2],
                value=values,
                isomin=isomin,
                isomax=isomax,
                opacity=opacity,
                surface_count=surface_count,
                caps=dict(x_show=False, y_show=False, z_show=False),
            )
        )
        if show_atoms and field.structure is not None:
            positions = field.structure.cartesian_positions()
            symbols = field.structure.symbols
            figure.add_trace(
                go.Scatter3d(
                    x=positions[:, 0],
                    y=positions[:, 1],
                    z=positions[:, 2],
                    text=symbols,
                    mode="markers",
                    marker=dict(size=10, opacity=0.7),
                )
            )

        axis = dict(
            showgrid=False,
            showbackground=False,
            zeroline=False,
            visible=False,
        )
        figure.update_layout(
            autosize=False,
            width=800,
            height=800,
            showlegend=False,
            scene=dict(xaxis=axis, yaxis=axis, zaxis=axis),
            title=title,
            title_font_family="Times New Roman",
        )
        return figure

    @staticmethod
    def save_png(
        figure: go.Figure,
        path: str | Path,
        *,
        scale: float = 1.0,
    ) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_image(path, scale=scale)
        return path

    @staticmethod
    def save_html(figure: go.Figure, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.write_html(path)
        return path
