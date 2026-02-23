"""
习题 5.5：三次多项式拟合（无噪声 vs 加标准正态噪声）

运行示例：
  python ch05/hw05_05/solution.py
  python ch05/hw05_05/solution.py --seed 42 --no-plot
"""

import argparse

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


TRUE_COEF = np.array([8.0, 5.0, 2.0, -1.0], dtype=float)  # [a3, a2, a1, a0]


def poly_value(x, coef):
    """计算多项式值（coef 按高次到低次）。"""
    return np.polyval(coef, x)


def fit_cubic(x, y):
    """最小二乘拟合三次多项式。"""
    coef = np.polyfit(x, y, deg=3)
    y_hat = np.polyval(coef, x)
    residual = y - y_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return coef, y_hat, residual, sse, rmse, r2


def format_poly(coef):
    """将三次多项式格式化为可读表达式。"""
    c3, c2, c1, c0 = [float(v) for v in coef]
    return (
        f"{c3:.10f}*x^3 "
        f"{'+' if c2 >= 0 else '-'} {abs(c2):.10f}*x^2 "
        f"{'+' if c1 >= 0 else '-'} {abs(c1):.10f}*x "
        f"{'+' if c0 >= 0 else '-'} {abs(c0):.10f}"
    )


def print_fit_report(title, coef, sse, rmse, r2):
    """打印单个场景的拟合报告。"""
    delta = coef - TRUE_COEF
    print(f"\n--- {title} ---")
    print(f"拟合多项式: y = {format_poly(coef)}")
    print("系数（a3,a2,a1,a0）：")
    print("  [" + ", ".join(f"{v:.12f}" for v in coef) + "]")
    print("系数误差（拟合值-真值）：")
    print("  [" + ", ".join(f"{v:+.3e}" for v in delta) + "]")
    print(f"SSE  = {sse:.12f}")
    print(f"RMSE = {rmse:.12f}")
    print(f"R^2  = {r2:.12f}")


def plot_result(x, y_clean, y_noisy, coef_clean, coef_noisy):
    """绘制无噪声与有噪声两个场景的拟合结果。"""
    x_dense = np.linspace(float(np.min(x)), float(np.max(x)), 1200)
    y_true_dense = poly_value(x_dense, TRUE_COEF)
    y_fit_clean_dense = poly_value(x_dense, coef_clean)
    y_fit_noisy_dense = poly_value(x_dense, coef_noisy)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2), sharey=True)

    ax1.scatter(x, y_clean, s=18, color="#1f77b4", alpha=0.75, label="无噪声观测点")
    ax1.plot(x_dense, y_true_dense, color="#2ca02c", linewidth=2.0, label="真函数")
    ax1.plot(x_dense, y_fit_clean_dense, color="#d62728", linewidth=1.8, linestyle="--", label="拟合曲线")
    ax1.set_title("无噪声数据拟合")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.scatter(x, y_noisy, s=18, color="#ff7f0e", alpha=0.75, label="加噪声观测点")
    ax2.plot(x_dense, y_true_dense, color="#2ca02c", linewidth=2.0, label="真函数")
    ax2.plot(x_dense, y_fit_noisy_dense, color="#9467bd", linewidth=1.8, linestyle="--", label="拟合曲线")
    ax2.set_title("加标准正态噪声后拟合")
    ax2.set_xlabel("x")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(n_points=100, x_min=-6.0, x_max=6.0, seed=2026, show_plot=True):
    if n_points < 4:
        raise ValueError("三次多项式拟合至少需要 4 个点。")
    if x_max <= x_min:
        raise ValueError("必须满足 x_max > x_min。")

    x = np.linspace(x_min, x_max, n_points)
    y_clean = poly_value(x, TRUE_COEF)

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n_points)  # 服从 N(0,1)
    y_noisy = y_clean + noise

    coef_clean, _, _, sse_clean, rmse_clean, r2_clean = fit_cubic(x, y_clean)
    coef_noisy, _, _, sse_noisy, rmse_noisy, r2_noisy = fit_cubic(x, y_noisy)

    print("=== 习题 5.5 求解结果 ===")
    print(f"真函数: f(x) = {format_poly(TRUE_COEF)}")
    print(f"区间: [{x_min}, {x_max}]，等步长取点数: {n_points}")
    print(f"噪声分布: N(0,1)，随机种子: {seed}")

    print_fit_report("任务(1) 无噪声数据拟合", coef_clean, sse_clean, rmse_clean, r2_clean)
    print_fit_report("任务(2) 加噪声数据拟合", coef_noisy, sse_noisy, rmse_noisy, r2_noisy)

    if show_plot:
        plot_result(x, y_clean, y_noisy, coef_clean, coef_noisy)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.5 三次多项式最小二乘拟合")
    parser.add_argument("--n-points", type=int, default=100, help="等步长采样点数，默认 100")
    parser.add_argument("--x-min", type=float, default=-6.0, help="采样区间左端点，默认 -6")
    parser.add_argument("--x-max", type=float, default=6.0, help="采样区间右端点，默认 6")
    parser.add_argument("--seed", type=int, default=2026, help="标准正态噪声随机种子，默认 2026")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_points=args.n_points,
        x_min=args.x_min,
        x_max=args.x_max,
        seed=args.seed,
        show_plot=not args.no_plot,
    )

