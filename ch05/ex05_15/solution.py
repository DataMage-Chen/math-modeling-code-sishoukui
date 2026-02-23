"""
例题 5.15：用表 5.12 数据拟合 z = a*exp(b*x1) + c*x2^2。

运行：
  python ch05/ex05_15/solution.py
  python ch05/ex05_15/solution.py --no-plot
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


X1_DATA = np.array([6, 2, 6, 7, 4, 2, 5, 9], dtype=float)
X2_DATA = np.array([4, 9, 5, 3, 8, 5, 8, 2], dtype=float)
Z_DATA = np.array([14.2077, 39.3622, 17.8077, 11.8310, 32.8618, 16.9622, 33.0941, 11.1737], dtype=float)


def model_value(x1, x2, a, b, c):
    """经验函数 z = a*exp(b*x1) + c*x2^2。"""
    return a * np.exp(b * x1) + c * (x2 ** 2)


def residuals(theta, x1, x2, z):
    """残差向量。"""
    a, b, c = theta
    return model_value(x1, x2, a, b, c) - z


def fit_model(x1, x2, z):
    """多初值非线性最小二乘拟合。"""
    starts = [
        np.array([1.0, 0.10, 0.30]),
        np.array([1.0, 0.30, 0.50]),
        np.array([2.0, 0.20, 0.30]),
        np.array([0.5, 0.40, 0.40]),
        np.array([3.0, 0.05, 0.20]),
    ]

    best = None
    for x0 in starts:
        res = least_squares(
            residuals,
            x0=x0,
            args=(x1, x2, z),
            method="trf",
            ftol=1e-12,
            xtol=1e-12,
            gtol=1e-12,
            max_nfev=20000,
        )
        sse = float(np.dot(res.fun, res.fun))
        if best is None or sse < best["sse"]:
            best = {"res": res, "sse": sse}

    if best is None or (not best["res"].success and best["sse"] > 1e12):
        raise RuntimeError("拟合失败：未找到有效参数。")

    a, b, c = [float(v) for v in best["res"].x]
    z_hat = model_value(x1, x2, a, b, c)
    return a, b, c, z_hat


def calc_metrics(z_obs, z_hat):
    """误差指标。"""
    residual = z_obs - z_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.dot(z_obs - np.mean(z_obs), z_obs - np.mean(z_obs)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def print_report(a, b, c, z_hat, sse, rmse, r2, residual):
    """打印拟合结果。"""
    print("=== 例题 5.15 拟合结果 ===")
    print(f"拟合公式: z = {a:.10f}*exp({b:.10f}*x1) + {c:.10f}*x2^2")
    print(f"参数估计: a={a:.10f}, b={b:.10f}, c={c:.10f}")
    print(f"SSE  = {sse:.10f}")
    print(f"RMSE = {rmse:.10f}")
    print(f"R^2  = {r2:.10f}")

    print("\n观测值/预测值/残差：")
    print("    x1    x2      z_obs      z_hat    residual")
    for x1, x2, z, zh, e in zip(X1_DATA, X2_DATA, Z_DATA, z_hat, residual):
        print(f"{x1:6.1f}{x2:6.1f}{z:11.4f}{zh:11.4f}{e:12.6f}")


def plot_results(z_hat):
    """绘制三维点图与观测-预测对比图。"""
    fig = plt.figure(figsize=(12, 5.2))

    # 左图：三维点（观测 vs 预测）
    ax1 = fig.add_subplot(121, projection="3d")
    ax1.scatter(X1_DATA, X2_DATA, Z_DATA, color="#1f77b4", s=45, label="观测值")
    ax1.scatter(X1_DATA, X2_DATA, z_hat, color="#d62728", s=45, marker="^", label="预测值")
    ax1.set_title("三维点图：观测值与预测值")
    ax1.set_xlabel("x1")
    ax1.set_ylabel("x2")
    ax1.set_zlabel("z")
    ax1.legend()

    # 右图：观测-预测对比
    ax2 = fig.add_subplot(122)
    ax2.scatter(Z_DATA, z_hat, color="#2ca02c", s=48)
    z_min = float(min(np.min(Z_DATA), np.min(z_hat)))
    z_max = float(max(np.max(Z_DATA), np.max(z_hat)))
    ax2.plot([z_min, z_max], [z_min, z_max], "k--", linewidth=1.5, label="y=x")
    ax2.set_title("观测值 vs 预测值")
    ax2.set_xlabel("z_obs")
    ax2.set_ylabel("z_hat")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(show_plot=True):
    a, b, c, z_hat = fit_model(X1_DATA, X2_DATA, Z_DATA)
    sse, rmse, r2, residual = calc_metrics(Z_DATA, z_hat)
    print_report(a, b, c, z_hat, sse, rmse, r2, residual)
    if show_plot:
        plot_results(z_hat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.15 非线性经验函数拟合")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()
    solve(show_plot=not args.no_plot)
