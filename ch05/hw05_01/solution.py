"""
习题 5.1：三次样条插值与积分比较

运行示例：
  python ch05/hw05_01/solution.py
  python ch05/hw05_01/solution.py --bc-type natural
  python ch05/hw05_01/solution.py --n-points 1000 --plot-grid 5000 --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import quad
    from scipy.interpolate import CubicSpline
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


def g_func(x):
    """题目给定函数 g(x)。"""
    x = np.asarray(x, dtype=float)
    numerator = 3.0 * x ** 2 + 4.0 * x + 6.0
    denominator = x ** 2 + 8.0 * x + 6.0
    return (numerator / denominator) * np.sin(x)


def build_spline(x_nodes, y_nodes, bc_type):
    """构造三次样条插值函数。"""
    if bc_type == "not-a-knot":
        return CubicSpline(x_nodes, y_nodes, bc_type="not-a-knot")
    if bc_type == "natural":
        return CubicSpline(x_nodes, y_nodes, bc_type="natural")
    raise ValueError(f"不支持的边界条件: {bc_type}")


def compute_integrals(spline, a=0.0, b=10.0):
    """计算真函数积分与样条函数积分。"""
    i_true = float(quad(lambda t: float(g_func(t)), a, b, epsabs=1e-12, epsrel=1e-12, limit=400)[0])
    i_spline = float(spline.integrate(a, b))
    abs_err = abs(i_true - i_spline)
    rel_err = abs_err / abs(i_true) if abs(i_true) > 1e-15 else np.nan
    return i_true, i_spline, abs_err, rel_err


def print_report(n_points, bc_type, node_err, i_true, i_spline, abs_err, rel_err):
    """打印结果。"""
    print("=== 习题 5.1 求解结果 ===")
    print(f"节点数 n = {n_points}")
    print(f"样条边界条件 = {bc_type}")
    print(f"节点插值误差 max|hat_g(x_i)-g(x_i)| = {node_err:.3e}")
    print("\n积分结果：")
    print(f"  I_true   = ∫[0,10] g(x)dx      = {i_true:.12f}")
    print(f"  I_spline = ∫[0,10] hat_g(x)dx  = {i_spline:.12f}")
    print(f"  绝对误差 |I_true-I_spline|     = {abs_err:.12e}")
    print(f"  相对误差                    = {rel_err:.12e}")


def plot_result(x_dense, y_true_dense, y_spline_dense):
    """绘制 g(x) 与样条插值曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)

    ax1.plot(x_dense, y_true_dense, color="#1f77b4", linewidth=2.0, label="g(x)")
    ax1.plot(x_dense, y_spline_dense, color="#d62728", linewidth=1.8, linestyle="--", label="hat_g(x)")
    ax1.set_title("习题 5.1：函数与三次样条插值对比")
    ax1.set_ylabel("函数值")
    ax1.grid(alpha=0.3)
    ax1.legend()

    err_curve = np.abs(y_true_dense - y_spline_dense)
    ax2.plot(x_dense, err_curve, color="#2ca02c", linewidth=1.8, label="|g(x)-hat_g(x)|")
    ax2.set_xlabel("x")
    ax2.set_ylabel("绝对误差")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(n_points=1000, bc_type="not-a-knot", plot_grid=4000, show_plot=True):
    if n_points < 4:
        raise ValueError("三次样条至少需要 4 个节点。")
    if plot_grid < 200:
        raise ValueError("plot_grid 建议不小于 200。")

    a, b = 0.0, 10.0
    x_nodes = np.linspace(a, b, n_points)
    y_nodes = g_func(x_nodes)

    spline = build_spline(x_nodes, y_nodes, bc_type=bc_type)
    node_err = float(np.max(np.abs(spline(x_nodes) - y_nodes)))

    i_true, i_spline, abs_err, rel_err = compute_integrals(spline, a=a, b=b)
    print_report(n_points, bc_type, node_err, i_true, i_spline, abs_err, rel_err)

    if show_plot:
        x_dense = np.linspace(a, b, plot_grid)
        y_true_dense = g_func(x_dense)
        y_spline_dense = spline(x_dense)
        plot_result(x_dense, y_true_dense, y_spline_dense)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.1 三次样条插值与积分比较")
    parser.add_argument("--n-points", type=int, default=1000, help="等距节点个数，默认 1000")
    parser.add_argument(
        "--bc-type",
        type=str,
        choices=["not-a-knot", "natural"],
        default="not-a-knot",
        help="三次样条边界条件，默认 not-a-knot",
    )
    parser.add_argument("--plot-grid", type=int, default=4000, help="绘图致密网格点数，默认 4000")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_points=args.n_points,
        bc_type=args.bc_type,
        plot_grid=args.plot_grid,
        show_plot=not args.no_plot,
    )
