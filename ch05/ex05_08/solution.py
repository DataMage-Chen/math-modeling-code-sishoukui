"""
例题 5.8：散乱点海底曲面插值（scatteredInterpolant 的 Python 对应实现）。

运行：
  python ch05/ex05_08/solution.py
  python ch05/ex05_08/solution.py --method linear --grid-size 120
  python ch05/ex05_08/solution.py --method cubic --no-fill-nearest
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import griddata
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy scipy matplotlib"
    ) from exc


# 让 matplotlib 尽量正确显示中文和负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


# 表 5.7 数据
X_DATA = np.array(
    [129, 140, 103.5, 88, 185.5, 195, 105, 157.5, 107.5, 77, 81, 162, 162, 117.5],
    dtype=float,
)
Y_DATA = np.array(
    [7.5, 141.5, 23, 147, 22.5, 137.5, 85.5, -6.5, -81, 3, 56.5, -66.5, 84, -33.5],
    dtype=float,
)
Z_DATA = np.array([4, 8, 6, 8, 6, 8, 8, 9, 9, 8, 8, 9, 4, 9], dtype=float)


def build_grid(x_data, y_data, grid_size):
    """构造矩形规则网格。"""
    x_min, x_max = float(np.min(x_data)), float(np.max(x_data))
    y_min, y_max = float(np.min(y_data)), float(np.max(y_data))

    x_grid = np.linspace(x_min, x_max, grid_size)
    y_grid = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(x_grid, y_grid)
    return xx, yy, x_min, x_max, y_min, y_max


def interpolate_surface(xx, yy, method="linear", fill_nearest=True):
    """散乱点插值并可选最近邻补全凸包外缺失点。"""
    points = np.column_stack([X_DATA, Y_DATA])
    z_grid = griddata(points, Z_DATA, (xx, yy), method=method)

    if fill_nearest and method != "nearest":
        z_near = griddata(points, Z_DATA, (xx, yy), method="nearest")
        mask = np.isnan(z_grid)
        z_grid[mask] = z_near[mask]

    return z_grid


def summarize_surface(xx, yy, zz):
    """输出曲面统计信息。"""
    valid_mask = ~np.isnan(zz)
    if not np.any(valid_mask):
        raise ValueError("插值结果全为 NaN，请调整方法或补全策略。")

    z_valid = zz[valid_mask]
    z_min = float(np.min(z_valid))
    z_max = float(np.max(z_valid))

    flat_idx_min = int(np.nanargmin(zz))
    flat_idx_max = int(np.nanargmax(zz))
    row_min, col_min = np.unravel_index(flat_idx_min, zz.shape)
    row_max, col_max = np.unravel_index(flat_idx_max, zz.shape)

    x_min_pos = float(xx[row_min, col_min])
    y_min_pos = float(yy[row_min, col_min])
    x_max_pos = float(xx[row_max, col_max])
    y_max_pos = float(yy[row_max, col_max])

    return z_min, z_max, (x_min_pos, y_min_pos), (x_max_pos, y_max_pos)


def plot_surface(xx, yy, zz, method, fill_nearest):
    """绘制三维曲面图和等高线图。"""
    fig1 = plt.figure(figsize=(10, 6.5))
    ax1 = fig1.add_subplot(111, projection="3d")
    surf = ax1.plot_surface(xx, yy, zz, cmap="viridis", linewidth=0, antialiased=True, alpha=0.92)
    ax1.scatter(X_DATA, Y_DATA, Z_DATA, color="red", s=35, depthshade=True, label="观测点")
    ax1.set_title(
        f"例5.8 海底曲面插值（三维）\nmethod={method}, fill_nearest={fill_nearest}"
    )
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper left")
    fig1.colorbar(surf, ax=ax1, shrink=0.65, pad=0.08, label="z")

    fig2, ax2 = plt.subplots(figsize=(9, 6))
    cf = ax2.contourf(xx, yy, zz, levels=16, cmap="viridis")
    cs = ax2.contour(xx, yy, zz, levels=16, colors="k", linewidths=0.45, alpha=0.6)
    ax2.clabel(cs, inline=True, fontsize=7, fmt="%.1f")
    ax2.scatter(X_DATA, Y_DATA, c="red", s=28, label="观测点")
    ax2.set_title(f"例5.8 海底曲面等高线图（method={method}）")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.colorbar(cf, ax=ax2, label="z")

    plt.tight_layout()
    plt.show()


def solve(method="linear", grid_size=120, fill_nearest=True, show_plot=True):
    xx, yy, x_min, x_max, y_min, y_max = build_grid(X_DATA, Y_DATA, grid_size)
    zz = interpolate_surface(xx, yy, method=method, fill_nearest=fill_nearest)

    z_min, z_max, min_pos, max_pos = summarize_surface(xx, yy, zz)
    nan_ratio = float(np.isnan(zz).sum()) / zz.size * 100.0

    print("=== 例题 5.8 求解结果 ===")
    print(f"插值方法: {method}")
    print(f"网格规模: {grid_size} x {grid_size}")
    print(f"矩形区域: x in [{x_min:.2f}, {x_max:.2f}], y in [{y_min:.2f}, {y_max:.2f}]")
    print(f"凸包外缺失补全: {fill_nearest}")
    print(f"NaN 比例: {nan_ratio:.2f}%")
    print(f"插值曲面最小值: {z_min:.6f}, 位置约 ({min_pos[0]:.2f}, {min_pos[1]:.2f})")
    print(f"插值曲面最大值: {z_max:.6f}, 位置约 ({max_pos[0]:.2f}, {max_pos[1]:.2f})")

    if show_plot:
        plot_surface(xx, yy, zz, method=method, fill_nearest=fill_nearest)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.8 散乱点海底曲面插值")
    parser.add_argument(
        "--method",
        choices=["linear", "nearest", "cubic"],
        default="linear",
        help="插值方法，默认 linear（接近 scatteredInterpolant 常用口径）",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=120,
        help="矩形网格分辨率，默认 120",
    )
    parser.add_argument(
        "--no-fill-nearest",
        action="store_true",
        help="不使用最近邻填补凸包外 NaN（可能出现曲面空洞）",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示图形",
    )
    args = parser.parse_args()

    solve(
        method=args.method,
        grid_size=args.grid_size,
        fill_nearest=not args.no_fill_nearest,
        show_plot=not args.no_plot,
    )
