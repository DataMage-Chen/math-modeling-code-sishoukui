"""
例题 5.14：用模拟数据拟合 z = a + b*ln(x) + c*y。

运行：
  python ch05/ex05_14/solution.py
  python ch05/ex05_14/solution.py --seed 7 --n-samples 50 --noise-std 0.15
  python ch05/ex05_14/solution.py --no-plot
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


def simulate_data(n_samples, seed, noise_std, a_true, b_true, c_true):
    """生成模拟数据（x>0）。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(1.0, 10.0, size=n_samples)  # 保证 ln(x) 有定义
    y = rng.uniform(-2.0, 6.0, size=n_samples)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    z = a_true + b_true * np.log(x) + c_true * y + noise
    return x, y, z


def fit_least_squares(x, y, z):
    """线性最小二乘拟合参数 a,b,c。"""
    design = np.column_stack([np.ones_like(x), np.log(x), y])
    theta, *_ = np.linalg.lstsq(design, z, rcond=None)
    a_hat, b_hat, c_hat = [float(v) for v in theta]
    z_hat = design @ theta
    return a_hat, b_hat, c_hat, z_hat


def calc_metrics(z, z_hat):
    """计算误差指标。"""
    residual = z - z_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(z - np.mean(z), z - np.mean(z)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def print_report(a_true, b_true, c_true, a_hat, b_hat, c_hat, sse, rmse, r2):
    """打印拟合结果与参数偏差。"""
    print("=== 例题 5.14 拟合结果 ===")
    print(f"真实参数: a={a_true:.6f}, b={b_true:.6f}, c={c_true:.6f}")
    print(f"拟合参数: a={a_hat:.6f}, b={b_hat:.6f}, c={c_hat:.6f}")
    print(
        "拟合公式: "
        f"z = {a_hat:.6f} + {b_hat:.6f}*ln(x) + {c_hat:.6f}*y"
    )
    print("参数误差：")
    print(f"  a_hat-a_true = {a_hat - a_true:+.6f}")
    print(f"  b_hat-b_true = {b_hat - b_true:+.6f}")
    print(f"  c_hat-c_true = {c_hat - c_true:+.6f}")
    print("拟合指标：")
    print(f"  SSE  = {sse:.10f}")
    print(f"  RMSE = {rmse:.10f}")
    print(f"  R^2  = {r2:.10f}")


def plot_results(x, y, z, a_hat, b_hat, c_hat, z_hat):
    """绘制三维拟合图和观测-预测对比图。"""
    fig = plt.figure(figsize=(12, 5.2))

    # 左图：三维散点 + 拟合曲面
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(x, y, z, color="#1f77b4", s=28, alpha=0.85, label="观测点")

    x_line = np.linspace(float(np.min(x)), float(np.max(x)), 40)
    y_line = np.linspace(float(np.min(y)), float(np.max(y)), 40)
    xg, yg = np.meshgrid(x_line, y_line)
    zg = a_hat + b_hat * np.log(xg) + c_hat * yg
    surf = ax1.plot_surface(xg, yg, zg, cmap="viridis", alpha=0.68, linewidth=0)

    ax1.set_title("三维散点与拟合曲面")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper left")
    fig.colorbar(surf, ax=ax1, shrink=0.6, pad=0.08)

    # 右图：观测-预测对比
    ax2 = fig.add_subplot(122)
    ax2.scatter(z, z_hat, color="#d62728", s=32, alpha=0.85)
    z_min = float(min(np.min(z), np.min(z_hat)))
    z_max = float(max(np.max(z), np.max(z_hat)))
    ax2.plot([z_min, z_max], [z_min, z_max], "k--", linewidth=1.5, label="y=x")
    ax2.set_title("观测值 vs 预测值")
    ax2.set_xlabel("z_obs")
    ax2.set_ylabel("z_hat")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(n_samples, seed, noise_std, a_true, b_true, c_true, show_plot):
    x, y, z = simulate_data(
        n_samples=n_samples,
        seed=seed,
        noise_std=noise_std,
        a_true=a_true,
        b_true=b_true,
        c_true=c_true,
    )
    a_hat, b_hat, c_hat, z_hat = fit_least_squares(x, y, z)
    sse, rmse, r2, _ = calc_metrics(z, z_hat)

    print_report(a_true, b_true, c_true, a_hat, b_hat, c_hat, sse, rmse, r2)
    if show_plot:
        plot_results(x, y, z, a_hat, b_hat, c_hat, z_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.14 模拟数据拟合 z=a+b*ln(x)+c*y")
    parser.add_argument("--n-samples", type=int, default=40, help="模拟样本数，默认 40")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument("--noise-std", type=float, default=0.2, help="噪声标准差，默认 0.2")
    parser.add_argument("--a-true", type=float, default=2.0, help="模拟真实参数 a，默认 2.0")
    parser.add_argument("--b-true", type=float, default=1.5, help="模拟真实参数 b，默认 1.5")
    parser.add_argument("--c-true", type=float, default=-0.8, help="模拟真实参数 c，默认 -0.8")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_samples=args.n_samples,
        seed=args.seed,
        noise_std=args.noise_std,
        a_true=args.a_true,
        b_true=args.b_true,
        c_true=args.c_true,
        show_plot=not args.no_plot,
    )
