"""
例题 5.19：利用模拟数据拟合高斯曲面
z = exp(-((x-mu1)^2 + (y-mu2)^2)/(2*sigma^2))

运行：
  python ch05/ex05_19/solution.py
  python ch05/ex05_19/solution.py --n-samples 350 --noise-std 0.03
  python ch05/ex05_19/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares
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


def gaussian_surface(x, y, mu1, mu2, sigma):
    """二维高斯曲面（幅值固定为 1）。"""
    sigma = max(float(sigma), 1e-12)
    r2 = (x - mu1) ** 2 + (y - mu2) ** 2
    return np.exp(-r2 / (2.0 * sigma * sigma))


def simulate_data(n_samples, seed, noise_std, mu1_true, mu2_true, sigma_true):
    """生成模拟样本。"""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-4.0, 4.0, size=n_samples)
    y = rng.uniform(-4.0, 4.0, size=n_samples)
    z_clean = gaussian_surface(x, y, mu1_true, mu2_true, sigma_true)
    noise = rng.normal(0.0, noise_std, size=n_samples)
    z_obs = z_clean + noise
    return x, y, z_obs, z_clean


def residuals(theta, x, y, z_obs):
    """非线性最小二乘残差。"""
    mu1, mu2, sigma = theta
    z_hat = gaussian_surface(x, y, mu1, mu2, sigma)
    return z_hat - z_obs


def fit_parameters(x, y, z_obs):
    """多初值重启拟合 mu1, mu2, sigma。"""
    starts = [
        np.array([0.0, 0.0, 1.0], dtype=float),
        np.array([1.0, -1.0, 1.2], dtype=float),
        np.array([-1.0, 1.0, 0.8], dtype=float),
        np.array([0.5, 0.5, 2.0], dtype=float),
        np.array([-0.5, -0.5, 1.6], dtype=float),
    ]

    best = None
    for x0 in starts:
        res = least_squares(
            residuals,
            x0=x0,
            args=(x, y, z_obs),
            bounds=([-10.0, -10.0, 1e-6], [10.0, 10.0, 20.0]),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=20000,
        )
        sse = float(np.dot(res.fun, res.fun))
        if best is None or sse < best["sse"]:
            best = {"res": res, "sse": sse, "start": x0}

    if best is None:
        raise RuntimeError("拟合失败：未得到可行解。")

    mu1_hat, mu2_hat, sigma_hat = [float(v) for v in best["res"].x]
    z_hat = gaussian_surface(x, y, mu1_hat, mu2_hat, sigma_hat)
    return mu1_hat, mu2_hat, sigma_hat, z_hat, best


def metrics(z_obs, z_hat):
    """计算 SSE / RMSE / R^2。"""
    residual = z_obs - z_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(z_obs - np.mean(z_obs), z_obs - np.mean(z_obs)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def print_report(mu1_true, mu2_true, sigma_true, mu1_hat, mu2_hat, sigma_hat, best, sse, rmse, r2):
    """打印结果。"""
    print("=== 例题 5.19 拟合结果 ===")
    print("真实参数：")
    print(f"  mu1={mu1_true:.6f}, mu2={mu2_true:.6f}, sigma={sigma_true:.6f}")
    print("拟合参数：")
    print(f"  mu1={mu1_hat:.6f}, mu2={mu2_hat:.6f}, sigma={sigma_hat:.6f}")
    print(
        "拟合曲面：\n"
        f"  z = exp(-((x-{mu1_hat:.6f})^2 + (y-{mu2_hat:.6f})^2)/(2*{sigma_hat:.6f}^2))"
    )
    print(f"最佳初值: {best['start']}")
    print("参数误差：")
    print(f"  Δmu1 = {mu1_hat - mu1_true:+.6f}")
    print(f"  Δmu2 = {mu2_hat - mu2_true:+.6f}")
    print(f"  Δsigma = {sigma_hat - sigma_true:+.6f}")
    print("拟合指标：")
    print(f"  SSE  = {sse:.10f}")
    print(f"  RMSE = {rmse:.10f}")
    print(f"  R^2  = {r2:.10f}")


def plot_result(x, y, z_obs, mu1_hat, mu2_hat, sigma_hat):
    """绘制三维拟合图和等高线图。"""
    xg = np.linspace(float(np.min(x)), float(np.max(x)), 100)
    yg = np.linspace(float(np.min(y)), float(np.max(y)), 100)
    xx, yy = np.meshgrid(xg, yg)
    zz = gaussian_surface(xx, yy, mu1_hat, mu2_hat, sigma_hat)

    fig1 = plt.figure(figsize=(10, 6.2))
    ax1 = fig1.add_subplot(111, projection="3d")
    ax1.scatter(x, y, z_obs, s=12, alpha=0.6, color="#1f77b4", label="模拟观测点")
    surf = ax1.plot_surface(xx, yy, zz, cmap="viridis", alpha=0.7, linewidth=0)
    ax1.set_title("例5.19 观测散点与拟合高斯曲面")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper left")
    fig1.colorbar(surf, ax=ax1, shrink=0.65, pad=0.08, label="z")

    fig2, ax2 = plt.subplots(figsize=(7.5, 6))
    cf = ax2.contourf(xx, yy, zz, levels=16, cmap="viridis")
    ax2.contour(xx, yy, zz, levels=16, colors="k", linewidths=0.35, alpha=0.5)
    ax2.scatter(x, y, c="white", edgecolors="black", s=12, alpha=0.8, label="观测点投影")
    ax2.set_title("拟合高斯曲面等高线图")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(alpha=0.25)
    ax2.legend()
    fig2.colorbar(cf, ax=ax2, label="z")

    plt.tight_layout()
    plt.show()


def solve(n_samples, seed, noise_std, mu1_true, mu2_true, sigma_true, show_plot):
    x, y, z_obs, _ = simulate_data(
        n_samples=n_samples,
        seed=seed,
        noise_std=noise_std,
        mu1_true=mu1_true,
        mu2_true=mu2_true,
        sigma_true=sigma_true,
    )
    mu1_hat, mu2_hat, sigma_hat, z_hat, best = fit_parameters(x, y, z_obs)
    sse, rmse, r2 = metrics(z_obs, z_hat)
    print_report(mu1_true, mu2_true, sigma_true, mu1_hat, mu2_hat, sigma_hat, best, sse, rmse, r2)
    if show_plot:
        plot_result(x, y, z_obs, mu1_hat, mu2_hat, sigma_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.19 高斯曲面参数拟合")
    parser.add_argument("--n-samples", type=int, default=260, help="模拟样本数，默认 260")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument("--noise-std", type=float, default=0.03, help="噪声标准差，默认 0.03")
    parser.add_argument("--mu1-true", type=float, default=0.8, help="模拟真实参数 mu1，默认 0.8")
    parser.add_argument("--mu2-true", type=float, default=-0.6, help="模拟真实参数 mu2，默认 -0.6")
    parser.add_argument("--sigma-true", type=float, default=1.4, help="模拟真实参数 sigma，默认 1.4")
    parser.add_argument("--no-plot", action="store_true", help="不显示图形")
    args = parser.parse_args()

    solve(
        n_samples=args.n_samples,
        seed=args.seed,
        noise_std=args.noise_std,
        mu1_true=args.mu1_true,
        mu2_true=args.mu2_true,
        sigma_true=args.sigma_true,
        show_plot=not args.no_plot,
    )
