"""
例题 5.21：求 cos(x) 在 Span{1, x^2, x^4} 上的最佳平方逼近多项式。

运行：
  python ch05/ex05_21/solution.py
  python ch05/ex05_21/solution.py --num-grid 2001 --no-plot
"""

import argparse
import math

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy matplotlib"
    ) from exc


# 让 matplotlib 尽量正确显示中文和负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def even_power_integral(power, a):
    """∫_{-a}^{a} x^power dx（power 为非负整数）。"""
    if power % 2 == 1:
        return 0.0
    return 2.0 * a ** (power + 1) / (power + 1)


def rhs_integrals(a):
    """
    计算 b = [<cos x, 1>, <cos x, x^2>, <cos x, x^4>]。
    使用分部积分得到的解析表达式。
    """
    # I0 = ∫_{-a}^{a} cos x dx = 2 sin(a)
    i0 = 2.0 * math.sin(a)

    # I2 = 2 * [x^2 sin x + 2x cos x - 2 sin x]_{0}^{a}
    i2 = 2.0 * (a * a * math.sin(a) + 2.0 * a * math.cos(a) - 2.0 * math.sin(a))

    # I4 = 2 * [x^4 sin x + 4x^3 cos x - 12x^2 sin x - 24x cos x + 24 sin x]_{0}^{a}
    i4 = 2.0 * (
        a ** 4 * math.sin(a)
        + 4.0 * a ** 3 * math.cos(a)
        - 12.0 * a * a * math.sin(a)
        - 24.0 * a * math.cos(a)
        + 24.0 * math.sin(a)
    )

    return np.array([i0, i2, i4], dtype=float)


def solve_coefficients():
    """构造正规方程并求解系数 c0,c1,c2。"""
    a = math.pi / 2.0
    powers = [0, 2, 4]

    g = np.zeros((3, 3), dtype=float)
    for i, pi in enumerate(powers):
        for j, pj in enumerate(powers):
            g[i, j] = even_power_integral(pi + pj, a)

    b = rhs_integrals(a)
    coef = np.linalg.solve(g, b)
    return coef, g, b, a


def poly_value(x, coef):
    """p(x) = c0 + c1*x^2 + c2*x^4。"""
    c0, c1, c2 = coef
    return c0 + c1 * x * x + c2 * x ** 4


def inner_product_trapz(f_val, g_val, x):
    """离散网格下近似内积 ∫ f*g dx。"""
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(f_val * g_val, x))


def print_report(coef, num_grid, a):
    """打印结果与校验信息。"""
    c0, c1, c2 = coef
    x = np.linspace(-a, a, num_grid)
    f = np.cos(x)
    p = poly_value(x, coef)
    e = f - p

    max_abs_err = float(np.max(np.abs(e)))
    rmse = float(np.sqrt(np.mean(e ** 2)))

    ip0 = inner_product_trapz(e, np.ones_like(x), x)
    ip2 = inner_product_trapz(e, x ** 2, x)
    ip4 = inner_product_trapz(e, x ** 4, x)

    print("=== 例题 5.21 求解结果 ===")
    print("最优逼近多项式：")
    print(f"  p(x) = {c0:.12f} + ({c1:.12f})*x^2 + ({c2:.12f})*x^4")

    print("误差指标（网格近似）：")
    print(f"  max|cos(x)-p(x)| = {max_abs_err:.10e}")
    print(f"  RMSE            = {rmse:.10e}")

    print("正交性校验（应接近 0）：")
    print(f"  <f-p, 1>   ≈ {ip0:.3e}")
    print(f"  <f-p, x^2> ≈ {ip2:.3e}")
    print(f"  <f-p, x^4> ≈ {ip4:.3e}")

    return x, f, p, e


def plot_result(x, f, p, e):
    """绘制函数对比图和误差曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    ax1.plot(x, f, color="#1f77b4", linewidth=2.1, label="f(x)=cos x")
    ax1.plot(x, p, color="#d62728", linewidth=2.1, linestyle="--", label="最佳平方逼近 p(x)")
    ax1.set_title("cos(x) 与最佳平方逼近多项式对比")
    ax1.set_ylabel("函数值")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x, e, color="#2ca02c", linewidth=1.8, label="误差 e(x)=cos(x)-p(x)")
    ax2.axhline(0.0, color="k", linewidth=1.0)
    ax2.set_title("逼近误差曲线")
    ax2.set_xlabel("x")
    ax2.set_ylabel("误差")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(num_grid=1201, show_plot=True):
    coef, _, _, a = solve_coefficients()
    x, f, p, e = print_report(coef, num_grid=num_grid, a=a)
    if show_plot:
        plot_result(x, f, p, e)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.21 最佳平方逼近多项式")
    parser.add_argument("--num-grid", type=int, default=1201, help="误差评估网格点数，默认 1201")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()
    solve(num_grid=args.num_grid, show_plot=not args.no_plot)
