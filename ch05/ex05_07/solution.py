"""
例题 5.7：双三次样条高程插值与地表面积估计。

运行：
  python ch05/ex05_07/solution.py
  python ch05/ex05_07/solution.py --grid-step 10 --area-step 10
  python ch05/ex05_07/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import RectBivariateSpline
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


# 表 5.6 原始网格坐标（单位 m）
X_BASE = np.arange(0.0, 1400.0 + 100.0, 100.0)  # 0..1400，共 15 列
Y_BASE = np.arange(0.0, 1200.0 + 100.0, 100.0)  # 0..1200，共 13 行

# Z_BASE 按 y 从小到大（0->1200）排列，每行对应一个 y、每列对应一个 x
Z_BASE = np.array(
    [
        [370, 470, 550, 600, 670, 690, 670, 620, 580, 450, 400, 300, 100, 150, 250],  # y=0
        [510, 620, 730, 800, 850, 870, 850, 780, 720, 650, 500, 200, 300, 350, 320],  # y=100
        [650, 760, 880, 970, 1020, 1050, 1020, 830, 800, 700, 300, 500, 550, 480, 350],  # y=200
        [740, 880, 1080, 1130, 1250, 1280, 1230, 1040, 900, 500, 700, 780, 750, 650, 550],  # y=300
        [830, 980, 1180, 1320, 1450, 1420, 400, 1300, 700, 900, 850, 810, 380, 780, 750],  # y=400
        [880, 1060, 1230, 1390, 1500, 1500, 1400, 900, 1100, 1060, 950, 870, 900, 936, 950],  # y=500
        [910, 1090, 1270, 1500, 1200, 1100, 1350, 1450, 1200, 1150, 1010, 880, 1000, 1050, 1100],  # y=600
        [950, 1190, 1370, 1500, 1200, 1100, 1550, 1600, 1550, 1380, 1070, 900, 1050, 1150, 1200],  # y=700
        [1430, 1450, 1460, 1500, 1550, 1600, 1550, 1600, 1600, 1600, 1550, 1500, 1500, 1550, 1550],  # y=800
        [1420, 1430, 1450, 1480, 1500, 1550, 1510, 1430, 1300, 1200, 980, 850, 750, 550, 500],  # y=900
        [1380, 1410, 1430, 1450, 1470, 1320, 1280, 1200, 1080, 940, 780, 620, 460, 370, 350],  # y=1000
        [1370, 1390, 1410, 1430, 1440, 1140, 1110, 1050, 950, 820, 690, 540, 380, 300, 210],  # y=1100
        [1350, 1370, 1390, 1400, 1410, 960, 940, 880, 800, 690, 570, 430, 290, 210, 150],  # y=1200
    ],
    dtype=float,
)


def make_axis(max_value, step):
    """构造含端点的等间距坐标轴。"""
    count = int(round(max_value / step))
    return np.linspace(0.0, max_value, count + 1)


def trapezoid_2d(f, x, y):
    """二维梯形积分（先对 x，再对 y）。"""
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(trapz_fn(f, x, axis=1), y, axis=0))


def solve(grid_step=10.0, area_step=10.0, show_plot=True):
    if grid_step <= 0 or area_step <= 0:
        raise ValueError("grid_step 和 area_step 必须为正数。")

    # 双三次样条插值器：第1维是 y，第2维是 x
    spline = RectBivariateSpline(Y_BASE, X_BASE, Z_BASE, kx=3, ky=3)

    # 原始节点插值误差校验
    z_fit_base = spline(Y_BASE, X_BASE)
    fit_err = float(np.max(np.abs(z_fit_base - Z_BASE)))

    # 题目要求：x,y 方向间隔 10 的节点高程
    x_grid = make_axis(1400.0, grid_step)
    y_grid = make_axis(1200.0, grid_step)
    z_grid = spline(y_grid, x_grid)  # shape=(len(y_grid), len(x_grid))

    max_idx = np.unravel_index(np.argmax(z_grid), z_grid.shape)
    min_idx = np.unravel_index(np.argmin(z_grid), z_grid.shape)
    z_max = float(z_grid[max_idx])
    z_min = float(z_grid[min_idx])
    x_max, y_max = float(x_grid[max_idx[1]]), float(y_grid[max_idx[0]])
    x_min, y_min = float(x_grid[min_idx[1]]), float(y_grid[min_idx[0]])

    # 地表面积：A = ∬ sqrt(1 + sx^2 + sy^2) dxdy
    x_area = make_axis(1400.0, area_step)
    y_area = make_axis(1200.0, area_step)
    dz_dy = spline(y_area, x_area, dx=1, dy=0)
    dz_dx = spline(y_area, x_area, dx=0, dy=1)
    integrand = np.sqrt(1.0 + dz_dx * dz_dx + dz_dy * dz_dy)
    area_m2 = trapezoid_2d(integrand, x_area, y_area)
    area_km2 = area_m2 / 1e6

    print("=== 例题 5.7 求解结果 ===")
    print(f"原始节点插值校验 max|s-z| = {fit_err:.3e}")
    print(
        f"间隔 {grid_step:g} 的插值网格规模: "
        f"{len(y_grid)} x {len(x_grid)} = {len(y_grid) * len(x_grid)} 个节点"
    )
    print(f"最大高程: {z_max:.6f} m, 位置 (x,y)=({x_max:.1f},{y_max:.1f})")
    print(f"最小高程: {z_min:.6f} m, 位置 (x,y)=({x_min:.1f},{y_min:.1f})")
    print(f"地表面积估计: {area_m2:.6f} m^2")
    print(f"地表面积估计: {area_km2:.6f} km^2")
    print(
        f"用于面积积分的网格步长: {area_step:g} "
        f"(网格 {len(y_area)} x {len(x_area)})"
    )

    if show_plot:
        xx, yy = np.meshgrid(x_grid, y_grid)

        # 等高线图
        fig1, ax1 = plt.subplots(figsize=(9, 6))
        cf = ax1.contourf(xx, yy, z_grid, levels=18, cmap="terrain")
        cs = ax1.contour(xx, yy, z_grid, levels=18, colors="k", linewidths=0.5, alpha=0.5)
        ax1.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
        ax1.scatter([x_max, x_min], [y_max, y_min], c=["red", "blue"], s=40, zorder=4)
        ax1.text(x_max + 20, y_max + 15, "最大值", color="red")
        ax1.text(x_min + 20, y_min + 15, "最小值", color="blue")
        ax1.set_title("例5.7 双三次样条插值等高线图")
        ax1.set_xlabel("x (m)")
        ax1.set_ylabel("y (m)")
        ax1.set_aspect("equal", adjustable="box")
        fig1.colorbar(cf, ax=ax1, label="高程 (m)")
        ax1.grid(alpha=0.2)

        # 三维网格图
        fig2 = plt.figure(figsize=(9.5, 6.8))
        ax2 = fig2.add_subplot(111, projection="3d")
        surf = ax2.plot_surface(xx, yy, z_grid, cmap="terrain", linewidth=0, antialiased=True, alpha=0.95)
        ax2.set_title("例5.7 双三次样条插值三维地形图")
        ax2.set_xlabel("x (m)")
        ax2.set_ylabel("y (m)")
        ax2.set_zlabel("高程 (m)")
        fig2.colorbar(surf, ax=ax2, shrink=0.65, pad=0.08, label="高程 (m)")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.7 双三次样条插值与地表面积估计")
    parser.add_argument(
        "--grid-step",
        type=float,
        default=10.0,
        help="插值高程网格步长（x,y），默认 10",
    )
    parser.add_argument(
        "--area-step",
        type=float,
        default=10.0,
        help="地表面积积分网格步长，默认 10",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示等高线与三维图",
    )
    args = parser.parse_args()

    solve(grid_step=args.grid_step, area_step=args.area_step, show_plot=not args.no_plot)
