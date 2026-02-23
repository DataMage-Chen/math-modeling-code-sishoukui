"""
例题 5.18：拟合 y = exp(-k1*x1)*sin(k2*x2) + x3^2 的参数 k1,k2。

运行：
  python ch05/ex05_18/solution.py
  python ch05/ex05_18/solution.py --no-plot
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


# 表 5.13 数据（25 组）
Y_DATA = np.array(
    [
        15.02, 12.62, 14.86, 13.98, 15.91, 12.47, 15.80, 14.32, 13.76, 15.18,
        14.20, 17.07, 15.40, 15.94, 14.33, 15.11, 13.81, 15.58, 15.85, 15.28,
        16.40, 15.02, 15.73, 14.75, 14.35,
    ],
    dtype=float,
)
X1_DATA = np.array(
    [
        23.73, 22.34, 28.84, 27.67, 20.83, 22.27, 27.57, 28.01, 24.79, 28.96,
        25.77, 23.17, 28.57, 23.52, 21.86, 28.95, 24.53, 27.65, 27.29, 29.07,
        32.47, 29.65, 22.11, 22.43, 20.04,
    ],
    dtype=float,
)
X2_DATA = np.array(
    [
        5.49, 4.32, 5.04, 4.72, 5.35, 4.27, 5.25, 4.62, 4.42, 5.30,
        4.87, 5.80, 5.22, 5.18, 4.86, 5.18, 4.88, 5.02, 5.55, 5.26,
        5.18, 5.08, 4.90, 4.65, 5.08,
    ],
    dtype=float,
)
X3_DATA = np.array(
    [
        1.21, 1.35, 1.92, 1.49, 1.56, 1.50, 1.85, 1.51, 1.46, 1.66,
        1.64, 1.90, 1.66, 1.98, 1.59, 1.37, 1.39, 1.66, 1.70, 1.82,
        1.75, 1.70, 1.81, 1.82, 1.53,
    ],
    dtype=float,
)


def model_value(x1, x2, x3, k1, k2):
    """模型函数：y = exp(-k1*x1)*sin(k2*x2) + x3^2。"""
    exp_arg = np.clip(-k1 * x1, -60.0, 60.0)
    return np.exp(exp_arg) * np.sin(k2 * x2) + x3 * x3


def residuals(theta, x1, x2, x3, y):
    """残差向量。"""
    k1, k2 = theta
    return model_value(x1, x2, x3, k1, k2) - y


def fit_parameters(x1, x2, x3, y):
    """多初值非线性最小二乘拟合。"""
    # 使用多初值降低陷入局部最优的风险
    starts = []
    for k1 in np.linspace(-0.4, 0.4, 9):
        for k2 in np.linspace(-3.0, 3.0, 13):
            starts.append(np.array([k1, k2], dtype=float))

    best = None
    for x0 in starts:
        res = least_squares(
            residuals,
            x0=x0,
            args=(x1, x2, x3, y),
            bounds=([-2.0, -20.0], [2.0, 20.0]),
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
        raise RuntimeError("拟合失败：未获得可行解。")

    k1, k2 = [float(v) for v in best["res"].x]
    y_hat = model_value(x1, x2, x3, k1, k2)
    return k1, k2, y_hat, best


def calc_metrics(y, y_hat):
    """计算 SSE、RMSE、R^2、残差。"""
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def print_report(k1, k2, y_hat, best, sse, rmse, r2, residual):
    """打印拟合结果。"""
    print("=== 例题 5.18 拟合结果 ===")
    print(
        "拟合函数: "
        f"y = exp(-({k1:.10f})*x1) * sin(({k2:.10f})*x2) + x3^2"
    )
    print(f"参数估计: k1 = {k1:.10f}, k2 = {k2:.10f}")
    print(f"最佳初值: k1_0 = {best['start'][0]:.6f}, k2_0 = {best['start'][1]:.6f}")
    print(f"SSE  = {sse:.10f}")
    print(f"RMSE = {rmse:.10f}")
    print(f"R^2  = {r2:.10f}")

    print("\n观测值/预测值/残差：")
    print("  序号      y_obs      y_hat    residual")
    for i, (y, yh, e) in enumerate(zip(Y_DATA, y_hat, residual), start=1):
        print(f"{i:4d}{y:11.4f}{yh:11.4f}{e:12.6f}")


def plot_result(y_hat, residual):
    """绘制观测-预测对比与残差图。"""
    idx = np.arange(1, len(Y_DATA) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    ax1.scatter(Y_DATA, y_hat, color="#1f77b4", s=42)
    y_min = float(min(np.min(Y_DATA), np.min(y_hat)))
    y_max = float(max(np.max(Y_DATA), np.max(y_hat)))
    ax1.plot([y_min, y_max], [y_min, y_max], "k--", linewidth=1.4, label="y=x")
    ax1.set_title("观测值 vs 预测值")
    ax1.set_xlabel("y_obs")
    ax1.set_ylabel("y_hat")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.axhline(0.0, color="k", linewidth=1.0)
    ax2.plot(idx, residual, "o-", color="#d62728", linewidth=1.4, markersize=5)
    ax2.set_title("残差随样本序号变化")
    ax2.set_xlabel("样本序号")
    ax2.set_ylabel("残差 (y_obs - y_hat)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def solve(show_plot=True):
    k1, k2, y_hat, best = fit_parameters(X1_DATA, X2_DATA, X3_DATA, Y_DATA)
    sse, rmse, r2, residual = calc_metrics(Y_DATA, y_hat)
    print_report(k1, k2, y_hat, best, sse, rmse, r2, residual)
    if show_plot:
        plot_result(y_hat, residual)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.18 非线性函数参数拟合")
    parser.add_argument("--no-plot", action="store_true", help="不显示图形")
    args = parser.parse_args()
    solve(show_plot=not args.no_plot)
