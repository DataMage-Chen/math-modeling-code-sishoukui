"""
例题 5.4：插值加工轨迹（拉格朗日 / 分段线性 / 三次样条）。

运行：
  python ch05/ex05_04/solution.py
  python ch05/ex05_04/solution.py --step 0.1 --no-table
  python ch05/ex05_04/solution.py --spline-bc natural
"""

import argparse
import warnings

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BarycentricInterpolator, CubicSpline, interp1d
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


X_DATA = np.array([0, 3, 5, 7, 9, 11, 12, 13, 14, 15], dtype=float)
Y_DATA = np.array([0, 1.2, 1.7, 2.0, 2.1, 2.0, 1.8, 1.2, 1.0, 1.6], dtype=float)


def build_interpolators(spline_bc):
    """构造三种插值函数。"""
    lagrange_interp = BarycentricInterpolator(X_DATA, Y_DATA)
    linear_interp = interp1d(X_DATA, Y_DATA, kind="linear")
    spline_interp = CubicSpline(X_DATA, Y_DATA, bc_type=spline_bc)
    return lagrange_interp, linear_interp, spline_interp


def lagrange_slope_at_zero():
    """用等价多项式导数计算拉格朗日插值在 x=0 处的斜率。"""
    degree = len(X_DATA) - 1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs = np.polyfit(X_DATA, Y_DATA, deg=degree)
    der_coeffs = np.polyder(coeffs)
    return float(np.polyval(der_coeffs, 0.0))


def interval_minimum(func, left, right, num_grid):
    """在区间上用稠密网格搜索最小值。"""
    x_grid = np.linspace(left, right, num_grid)
    y_grid = np.asarray(func(x_grid), dtype=float)
    idx = int(np.argmin(y_grid))
    return float(x_grid[idx]), float(y_grid[idx])


def print_coordinate_table(x_query, y_lagr, y_linear, y_spline):
    """打印步长坐标表。"""
    print("\n=== 步长坐标表（x 每 0.1 变化） ===")
    print("      x    y_拉格朗日      y_分段线性      y_三次样条")
    for x, yl, yp, ys in zip(x_query, y_lagr, y_linear, y_spline):
        print(f"{x:8.1f}{yl:14.6f}{yp:14.6f}{ys:14.6f}")


def plot_curves(
    x_dense,
    y_lagr_dense,
    y_linear_dense,
    y_spline_dense,
    mins,
    left,
    right,
):
    """绘制三种插值曲线与原始数据点。"""
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(x_dense, y_lagr_dense, linewidth=2.0, color="#d95f02", label="拉格朗日")
    ax.plot(x_dense, y_linear_dense, linewidth=2.0, color="#1b9e77", label="分段线性")
    ax.plot(x_dense, y_spline_dense, linewidth=2.0, color="#7570b3", label="三次样条")
    ax.scatter(X_DATA, Y_DATA, color="black", s=35, zorder=4, label="原始数据点")

    ax.axvspan(left, right, color="#cccccc", alpha=0.2, label=f"最小值搜索区间 [{left},{right}]")

    for name, (xm, ym), color in mins:
        ax.scatter([xm], [ym], marker="*", s=180, color=color, zorder=5)
        ax.text(xm + 0.05, ym + 0.03, f"{name}最小值", color=color, fontsize=9)

    ax.set_title("例5.4 三种插值方法曲线对比")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(step, left, right, num_grid, no_table, spline_bc):
    lagrange_interp, linear_interp, spline_interp = build_interpolators(spline_bc)

    # 0~15，步长 0.1 的离散轨迹点
    total_steps = int(round((X_DATA[-1] - X_DATA[0]) / step))
    x_query = np.linspace(X_DATA[0], X_DATA[-1], total_steps + 1)
    y_lagr = np.asarray(lagrange_interp(x_query), dtype=float)
    y_linear = np.asarray(linear_interp(x_query), dtype=float)
    y_spline = np.asarray(spline_interp(x_query), dtype=float)

    # 节点插值误差校验
    err_lagr = float(np.max(np.abs(lagrange_interp(X_DATA) - Y_DATA)))
    err_linear = float(np.max(np.abs(linear_interp(X_DATA) - Y_DATA)))
    err_spline = float(np.max(np.abs(spline_interp(X_DATA) - Y_DATA)))

    # x=0 处斜率
    slope_lagr = lagrange_slope_at_zero()
    slope_linear = float((Y_DATA[1] - Y_DATA[0]) / (X_DATA[1] - X_DATA[0]))  # 右导数
    slope_spline = float(spline_interp(0.0, 1))

    # [13,15] 区间最小值
    min_lagr = interval_minimum(lagrange_interp, left, right, num_grid)
    min_linear = interval_minimum(linear_interp, left, right, num_grid)
    min_spline = interval_minimum(spline_interp, left, right, num_grid)

    print("=== 例题 5.4 计算结果 ===")
    print(f"三次样条边界条件: {spline_bc}")
    print("节点插值误差校验（应接近 0）：")
    print(f"  拉格朗日: max|P(x_i)-y_i| = {err_lagr:.3e}")
    print(f"  分段线性: max|P(x_i)-y_i| = {err_linear:.3e}")
    print(f"  三次样条: max|P(x_i)-y_i| = {err_spline:.3e}")

    print("\nx=0 处斜率：")
    print(f"  拉格朗日  P'_L(0)   = {slope_lagr:.10f}")
    print(f"  分段线性  P'_(0+)   = {slope_linear:.10f}")
    print(f"  三次样条  P'_S(0)   = {slope_spline:.10f}")

    print(f"\n区间 [{left},{right}] 内最小值（网格搜索）：")
    print(f"  拉格朗日: min y = {min_lagr[1]:.10f} at x = {min_lagr[0]:.10f}")
    print(f"  分段线性: min y = {min_linear[1]:.10f} at x = {min_linear[0]:.10f}")
    print(f"  三次样条: min y = {min_spline[1]:.10f} at x = {min_spline[0]:.10f}")

    if not no_table:
        print_coordinate_table(x_query, y_lagr, y_linear, y_spline)

    x_dense = np.linspace(X_DATA[0], X_DATA[-1], 2000)
    y_lagr_dense = np.asarray(lagrange_interp(x_dense), dtype=float)
    y_linear_dense = np.asarray(linear_interp(x_dense), dtype=float)
    y_spline_dense = np.asarray(spline_interp(x_dense), dtype=float)

    mins = [
        ("拉格朗日", min_lagr, "#d95f02"),
        ("分段线性", min_linear, "#1b9e77"),
        ("三次样条", min_spline, "#7570b3"),
    ]
    plot_curves(
        x_dense=x_dense,
        y_lagr_dense=y_lagr_dense,
        y_linear_dense=y_linear_dense,
        y_spline_dense=y_spline_dense,
        mins=mins,
        left=left,
        right=right,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.4 三种插值方法比较")
    parser.add_argument("--step", type=float, default=0.1, help="离散轨迹步长，默认 0.1")
    parser.add_argument("--interval-left", type=float, default=13.0, help="最小值搜索左端点")
    parser.add_argument("--interval-right", type=float, default=15.0, help="最小值搜索右端点")
    parser.add_argument(
        "--num-grid",
        type=int,
        default=20001,
        help="区间最小值搜索网格点数，默认 20001",
    )
    parser.add_argument(
        "--no-table",
        action="store_true",
        help="不打印 x 每 0.1 变化时的坐标表",
    )
    parser.add_argument(
        "--spline-bc",
        choices=["not-a-knot", "natural"],
        default="not-a-knot",
        help="三次样条边界条件，默认 not-a-knot",
    )
    args = parser.parse_args()

    solve(
        step=args.step,
        left=args.interval_left,
        right=args.interval_right,
        num_grid=args.num_grid,
        no_table=args.no_table,
        spline_bc=args.spline_bc,
    )
