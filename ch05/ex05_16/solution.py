"""
例题 5.16：用模拟数据拟合二阶傅里叶级数（固定 w=1）。

运行：
  python ch05/ex05_16/solution.py
  python ch05/ex05_16/solution.py --n-samples 200 --noise-std 0.15
  python ch05/ex05_16/solution.py --no-plot
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


def true_signal(x):
    """真实信号 y = 2*cos(2x) + 6*sin(2x)。"""
    return 2.0 * np.cos(2.0 * x) + 6.0 * np.sin(2.0 * x)


def simulate_data(n_samples, noise_std, seed):
    """生成模拟数据。"""
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 2.0 * np.pi, n_samples)
    y_clean = true_signal(x)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    y = y_clean + noise
    return x, y, y_clean


def fit_fourier_order2_w_fixed(x, y, w=1.0):
    """
    固定 w=1（等价于参数上下界 Lower=Upper=1）后，
    拟合 f(x)=a0+a1 cos(wx)+a2 cos(2wx)+b1 sin(wx)+b2 sin(2wx)。
    """
    design = np.column_stack(
        [
            np.ones_like(x),
            np.cos(1.0 * w * x),
            np.cos(2.0 * w * x),
            np.sin(1.0 * w * x),
            np.sin(2.0 * w * x),
        ]
    )
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    a0, a1, a2, b1, b2 = [float(v) for v in coef]
    y_hat = design @ coef
    return a0, a1, a2, b1, b2, y_hat


def metrics(y, y_hat):
    """计算误差指标。"""
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def print_report(a0, a1, a2, b1, b2, w, sse, rmse, r2):
    """打印拟合结果与理论对照。"""
    def signed(v):
        return f"+{v:.10f}" if v >= 0 else f"{v:.10f}"

    print("=== 例题 5.16 拟合结果 ===")
    print(
        "拟合函数: "
        f"f(x)={a0:.10f}"
        f"{signed(a1)}*cos({w:.1f}*x)"
        f"{signed(a2)}*cos(2*{w:.1f}*x)"
        f"{signed(b1)}*sin({w:.1f}*x)"
        f"{signed(b2)}*sin(2*{w:.1f}*x)"
    )
    print("参数：")
    print(f"  a0={a0:.10f}, a1={a1:.10f}, a2={a2:.10f}, b1={b1:.10f}, b2={b2:.10f}, w={w:.1f}")
    print("拟合指标：")
    print(f"  SSE  = {sse:.10f}")
    print(f"  RMSE = {rmse:.10f}")
    print(f"  R^2  = {r2:.10f}")

    print("与理论系数对照（理论：a0=0,a1=0,a2=2,b1=0,b2=6）：")
    print(f"  Δa0={a0-0:+.6f}, Δa1={a1-0:+.6f}, Δa2={a2-2:+.6f}")
    print(f"  Δb1={b1-0:+.6f}, Δb2={b2-6:+.6f}")


def plot_result(x, y, y_clean, y_hat):
    """绘图：观测点、真实曲线、拟合曲线。"""
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.scatter(x, y, s=15, alpha=0.6, color="#1f77b4", label="模拟观测点")
    ax.plot(x, y_clean, linewidth=2.0, color="#2ca02c", label="真实函数")
    ax.plot(x, y_hat, linewidth=2.0, color="#d62728", linestyle="--", label="傅里叶拟合")
    ax.set_title("例5.16 二阶傅里叶级数拟合（w=1）")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(n_samples, noise_std, seed, show_plot):
    x, y, y_clean = simulate_data(n_samples=n_samples, noise_std=noise_std, seed=seed)
    w = 1.0  # 通过“上下界同值”思想固定为常数
    a0, a1, a2, b1, b2, y_hat = fit_fourier_order2_w_fixed(x, y, w=w)
    sse, rmse, r2 = metrics(y, y_hat)
    print_report(a0, a1, a2, b1, b2, w, sse, rmse, r2)
    if show_plot:
        plot_result(x, y, y_clean, y_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.16 二阶傅里叶级数拟合")
    parser.add_argument("--n-samples", type=int, default=120, help="模拟样本点数，默认 120")
    parser.add_argument("--noise-std", type=float, default=0.10, help="噪声标准差，默认 0.10")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_samples=args.n_samples,
        noise_std=args.noise_std,
        seed=args.seed,
        show_plot=not args.no_plot,
    )
