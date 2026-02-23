"""
例题 5.9：根据实验数据拟合经验公式 y = a t + b。

运行：
  python ch05/ex05_09/solution.py
  python ch05/ex05_09/solution.py --show-plot
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


T_DATA = np.array([0, 1, 2, 3, 4, 5, 6, 7], dtype=float)
Y_DATA = np.array([27.0, 26.8, 26.5, 26.3, 26.1, 25.7, 25.3, 24.8], dtype=float)


def fit_linear(t, y):
    """最小二乘拟合 y = a*t + b。"""
    t_mean = float(np.mean(t))
    y_mean = float(np.mean(y))
    s_tt = float(np.sum((t - t_mean) ** 2))
    s_ty = float(np.sum((t - t_mean) * (y - y_mean)))

    a = s_ty / s_tt
    b = y_mean - a * t_mean

    y_hat = a * t + b
    residual = y - y_hat

    sse = float(np.sum(residual**2))
    sst = float(np.sum((y - y_mean) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    rmse = float(np.sqrt(np.mean(residual**2)))

    return a, b, y_hat, residual, r2, rmse


def print_report(a, b, y_hat, residual, r2, rmse):
    """打印拟合结果。"""
    print("=== 例题 5.9 拟合结果 ===")
    print(f"经验公式: y = {a:.10f} * t + {b:.10f}")
    print(f"斜率 a（厚度变化率）: {a:.10f}")
    print(f"截距 b（初始厚度估计）: {b:.10f}")
    print(f"R^2: {r2:.10f}")
    print(f"RMSE: {rmse:.10f}")

    print("\n观测值/预测值/残差：")
    print("      t      y_obs      y_hat    residual")
    for t, y, yh, e in zip(T_DATA, Y_DATA, y_hat, residual):
        print(f"{t:7.1f}{y:11.4f}{yh:11.4f}{e:12.6f}")


def plot_result(a, b):
    """绘制散点与拟合直线。"""
    t_dense = np.linspace(float(np.min(T_DATA)), float(np.max(T_DATA)), 300)
    y_dense = a * t_dense + b

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.scatter(T_DATA, Y_DATA, color="#1f78b4", s=55, label="实验观测点")
    ax.plot(t_dense, y_dense, color="#e31a1c", linewidth=2.0, label="最小二乘拟合直线")
    ax.set_title("例5.9 线性经验公式拟合")
    ax.set_xlabel("t")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(show_plot=False):
    a, b, y_hat, residual, r2, rmse = fit_linear(T_DATA, Y_DATA)
    print_report(a, b, y_hat, residual, r2, rmse)
    if show_plot:
        plot_result(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.9 线性经验公式拟合")
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="显示散点与拟合直线图",
    )
    args = parser.parse_args()
    solve(show_plot=args.show_plot)
