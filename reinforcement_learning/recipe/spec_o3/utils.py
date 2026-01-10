# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import warnings

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import AutoMinorLocator, MultipleLocator

warnings.filterwarnings("ignore")


def plot_spectrum(
    wavelength,
    flux,
    wavelength_range=None,
    title="Spectrum",
    line_color="blue",
    line_width=1.0,
    alpha=0.8,
    show_grid=True,
):
    """
    Plot spectrum visualization with support for full and partial wavelength ranges.
    Automatically applies redshift correction and optimal figure sizing for AI model efficiency.

    Parameters:
    -----------
    wavelength : array-like
        Wavelength data in Angstroms (observed frame)
    flux : array-like
        Flux data
    wavelength_range : tuple, optional
        Wavelength range to display after redshift correction (min_wavelength, max_wavelength)
        If None, shows full range
    title : str, optional
        Plot title
    line_color : str, optional
        Spectrum line color
    line_width : float, optional
        Spectrum line width
    alpha : float, optional
        Transparency
    show_grid : bool, optional
        Whether to show grid

    Returns:
    --------
    fig, ax : matplotlib figure and axes objects
    """

    # Data validation and conversion
    wavelength = np.array(wavelength)
    flux = np.array(flux)

    if len(wavelength) != len(flux):
        raise ValueError("wavelength and flux must have same length")

    if len(wavelength) == 0:
        raise ValueError("Input data cannot be empty")

    # Filter invalid data (NaN, inf, etc.)
    valid_mask = np.isfinite(wavelength) & np.isfinite(flux)
    wavelength = wavelength[valid_mask]
    flux = flux[valid_mask]

    if len(wavelength) == 0:
        raise ValueError("No valid data after filtering")

    # Sort by wavelength
    sort_idx = np.argsort(wavelength)
    wavelength = wavelength[sort_idx]
    flux = flux[sort_idx]

    # Apply wavelength range filtering
    if wavelength_range is not None:
        min_wave, max_wave = wavelength_range
        if min_wave >= max_wave:
            raise ValueError("wavelength_range minimum must be less than maximum")

        mask = (wavelength >= min_wave) & (wavelength <= max_wave)
        if not np.any(mask):
            raise ValueError(f"No data points in specified wavelength range {wavelength_range}")

        wavelength = wavelength[mask]
        flux = flux[mask]

    # Determine optimal figure size automatically (optimized for AI model token efficiency)
    wave_range = wavelength.max() - wavelength.min()
    data_points = len(wavelength)

    # Balanced size calculation to minimize token consumption while ensuring readability
    if wavelength_range is not None:
        # For zoomed-in views, ensure minimum readable size
        # Calculate base width from range but with reasonable minimums
        if wave_range < 50:  # Very narrow ranges like H-alpha (10Å)
            base_width = 6
            base_height = 4
        elif wave_range < 200:  # Narrow regions
            base_width = 7
            base_height = 4.5
        elif wave_range < 1000:  # Medium regions
            base_width = 8
            base_height = 5
        else:  # Large regions
            base_width = 9
            base_height = 5.5
    else:
        # For full spectrum views, use moderately sized figures
        if wave_range > 3000:  # Full spectrum
            base_width = 10
            base_height = 6
        elif wave_range > 1000:  # Large region
            base_width = 8
            base_height = 5
        elif wave_range > 200:  # Medium region
            base_width = 7
            base_height = 4.5
        else:  # Small region
            base_width = 6
            base_height = 4

    # Minimal adjustment based on data density
    if data_points > 10000:
        base_width *= 1.05
    elif data_points < 50:
        base_width *= 0.95

    figsize = (base_width, base_height)

    # Create figure with optimized DPI for clarity while minimizing size
    fig, ax = plt.subplots(figsize=figsize, dpi=80)

    # 绘制光谱
    ax.plot(wavelength, flux, color=line_color, linewidth=line_width, alpha=alpha, label="Spectrum")

    # 设置标签和标题
    ax.set_xlabel("Wavelength (Å)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Flux", fontsize=12, fontweight="bold")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)

    # 网格设置
    if show_grid:
        ax.grid(True, alpha=0.3, linestyle="-", linewidth=0.5)
        ax.grid(True, which="minor", alpha=0.1, linestyle="-", linewidth=0.3)

    # 设置刻度
    ax.tick_params(axis="both", which="major", labelsize=10, direction="in")
    ax.tick_params(axis="both", which="minor", direction="in")

    # 自动设置刻度间隔 - 优化了窄波长范围的显示
    wave_range = wavelength.max() - wavelength.min()
    if wave_range >= 5000:
        ax.xaxis.set_major_locator(MultipleLocator(1000))
        ax.xaxis.set_minor_locator(MultipleLocator(200))
    elif wave_range >= 1000:
        ax.xaxis.set_major_locator(MultipleLocator(200))
        ax.xaxis.set_minor_locator(MultipleLocator(50))
    elif wave_range >= 500:
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(20))
    elif wave_range >= 300:  # 新增：300-500Å范围，使用更稀疏的刻度
        ax.xaxis.set_major_locator(MultipleLocator(100))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    elif wave_range >= 150:  # 优化：150-300Å范围，如6500-6700这种情况
        ax.xaxis.set_major_locator(MultipleLocator(50))
        ax.xaxis.set_minor_locator(MultipleLocator(25))
    elif wave_range >= 100:  # 100-150Å范围
        ax.xaxis.set_major_locator(MultipleLocator(25))
        ax.xaxis.set_minor_locator(MultipleLocator(5))
    elif wave_range >= 50:  # 50-100Å范围
        ax.xaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_minor_locator(MultipleLocator(2))
    elif wave_range >= 20:  # 20-50Å范围w
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
    elif wave_range >= 5:  # 5-20Å范围，如H-alpha
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.xaxis.set_minor_locator(MultipleLocator(0.5))
    else:  # 极窄范围 <5Å
        ax.xaxis.set_major_locator(MultipleLocator(1))
        ax.xaxis.set_minor_locator(MultipleLocator(0.2))

    # 设置y轴格式
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # 紧凑布局
    plt.tight_layout()

    # 智能放置统计信息文本框 - 选择相对空旷的角落
    stats_text = f"Data points: {len(wavelength)}\n"
    stats_text += f"λ range: {wavelength.min():.1f} - {wavelength.max():.1f} Å\n"
    stats_text += f"Flux range: {flux.min():.2e} - {flux.max():.2e}"

    # 将图像区域分成四个角落，检查哪个角落的数据点最少
    x_mid = (wavelength.min() + wavelength.max()) / 2
    y_mid = (flux.min() + flux.max()) / 2

    # 统计四个角落的数据点密度
    corners = {
        "top_left": np.sum((wavelength <= x_mid) & (flux >= y_mid)),
        "top_right": np.sum((wavelength >= x_mid) & (flux >= y_mid)),
        "bottom_left": np.sum((wavelength <= x_mid) & (flux <= y_mid)),
        "bottom_right": np.sum((wavelength >= x_mid) & (flux <= y_mid)),
    }

    # 选择数据点最少的角落
    best_corner = min(corners, key=corners.get)

    # 根据选择的角落设置文本框位置
    if best_corner == "top_left":
        x_pos, y_pos = 0.02, 0.98
        va, ha = "top", "left"
    elif best_corner == "top_right":
        x_pos, y_pos = 0.98, 0.98
        va, ha = "top", "right"
    elif best_corner == "bottom_left":
        x_pos, y_pos = 0.02, 0.02
        va, ha = "bottom", "left"
    else:  # bottom_right
        x_pos, y_pos = 0.98, 0.02
        va, ha = "bottom", "right"

    ax.text(
        x_pos,
        y_pos,
        stats_text,
        transform=ax.transAxes,
        verticalalignment=va,
        horizontalalignment=ha,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

    return fig, ax
