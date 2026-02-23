"""
例题 5.12：拟合 y = k*exp(m*t)。

运行：
  python ch05/ex05_12/solution.py
  python ch05/ex05_12/solution.py --no-plot
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


T_DATA = np.array([3, 6, 9, 12, 15, 18, 21, 24], dtype=float)
Y_DATA = np.array([57.6, 41.9, 31.0, 22.7, 16.6, 12.2, 8.9, 6.5], dtype=float)


def fit_log_linear(t, y):
    """
    对 ln(y)=A+m*t 做最小二乘，返回 k=exp(A), m。
    """
    if np.any(y <= 0):
        raise ValueError("对数线性化要求 y > 0。")

    y_log = np.log(y)

    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y_log))
    s_tt = float(np.sum((t - t_mean) ** 2))
    s_ty = float(np.sum((t - t_mean) * (y_log - y_mean)))

    m = s_ty / s_tt
    a = y_mean - m * t_mean  # a = ln(k)
    k = float(np.exp(a))

    y_log_hat = a + m * t
    y_hat = k * np.exp(m * t)

    return k, m, a, y_log_hat, y_hat


def metrics(y, y_hat):
    """返回 SSE, RMSE, R^2, residual。"""
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def print_report(k, m, y_log_hat, y_hat):
    """打印拟合参数与误差指标。"""
    y_log = np.log(Y_DATA)
    sse_log, rmse_log, r2_log, _ = metrics(y_log, y_log_hat)
    sse_raw, rmse_raw, r2_raw, residual_raw = metrics(Y_DATA, y_hat)

    print("=== 例题 5.12 拟合结果 ===")
    print(f"经验公式: y = {k:.10f} * exp({m:.10f} * t)")
    print(f"参数估计: k = {k:.10f}, m = {m:.10f}")
    print(f"趋势判断: m {'<' if m < 0 else '>='} 0")

    print("\n对数尺度拟合指标（ln(y) 线性回归）：")
    print(f"  SSE_log  = {sse_log:.10f}")
    print(f"  RMSE_log = {rmse_log:.10f}")
    print(f"  R^2_log  = {r2_log:.10f}")

    print("\n原始尺度拟合指标（y 预测误差）：")
    print(f"  SSE_raw  = {sse_raw:.10f}")
    print(f"  RMSE_raw = {rmse_raw:.10f}")
    print(f"  R^2_raw  = {r2_raw:.10f}")

    print("\n观测值/预测值/残差：")
    print("      t      y_obs      y_hat    residual")
    for t, y, yh, e in zip(T_DATA, Y_DATA, y_hat, residual_raw):
        print(f"{t:7.1f}{y:11.4f}{yh:11.4f}{e:12.6f}")


def plot_result(k, m):
    """绘制原始尺度与半对数尺度图。"""
    t_dense = np.linspace(float(np.min(T_DATA)), float(np.max(T_DATA)), 400)
    y_dense = k * np.exp(m * t_dense)
    y_hat = k * np.exp(m * T_DATA)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.8))

    ax1.scatter(T_DATA, Y_DATA, color="#1f77b4", s=45, label="观测点")
    ax1.plot(t_dense, y_dense, color="#d62728", linewidth=2.0, label="指数拟合曲线")
    ax1.set_title("原始尺度拟合")
    ax1.set_xlabel("t")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.scatter(T_DATA, np.log(Y_DATA), color="#2ca02c", s=45, label="ln(y) 观测")
    ax2.plot(T_DATA, np.log(y_hat), color="#ff7f0e", linewidth=2.0, label="ln(y) 拟合直线")
    ax2.set_title("对数尺度线性化")
    ax2.set_xlabel("t")
    ax2.set_ylabel("ln(y)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(show_plot=True):
    k, m, _, y_log_hat, y_hat = fit_log_linear(T_DATA, Y_DATA)
    print_report(k, m, y_log_hat, y_hat)
    if show_plot:
        plot_result(k, m)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.12 指数经验公式拟合")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示拟合图",
    )
    args = parser.parse_args()
    solve(show_plot=not args.no_plot)
