"""
习题 5.4：最小二乘法建立 theta 与 p 的线性经验公式

运行示例：
  python ch05/hw05_04/solution.py
  python ch05/hw05_04/solution.py --predict-p 50 70 90
  python ch05/hw05_04/solution.py --no-plot
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


# 表 5.16 数据
P_DATA = np.array([36.9, 46.7, 63.7, 77.8, 84.0, 87.5], dtype=float)
THETA_DATA = np.array([181.0, 197.0, 235.0, 270.0, 283.0, 292.0], dtype=float)


def fit_linear_least_squares(x_data, y_data):
    """最小二乘拟合 y = a*x + b。"""
    x_data = np.asarray(x_data, dtype=float)
    y_data = np.asarray(y_data, dtype=float)

    design = np.column_stack([x_data, np.ones_like(x_data)])
    coef, *_ = np.linalg.lstsq(design, y_data, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    y_hat = a * x_data + b

    residual = y_data - y_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y_data - np.mean(y_data)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    return {
        "a": a,
        "b": b,
        "y_hat": y_hat,
        "residual": residual,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
    }


def print_report(result):
    """打印拟合结果。"""
    print("=== 习题 5.4 求解结果 ===")
    print(f"最小二乘经验公式: theta = {result['a']:.10f} * p + {result['b']:.10f}")
    print("拟合指标：")
    print(f"  SSE  = {result['sse']:.10f}")
    print(f"  RMSE = {result['rmse']:.10f}")
    print(f"  R^2  = {result['r2']:.10f}")

    print("\n样本点拟合明细：")
    print("      p(%)    theta_obs    theta_hat     residual")
    for p, y, yh, e in zip(P_DATA, THETA_DATA, result["y_hat"], result["residual"]):
        print(f"  {p:8.2f}   {y:10.4f}   {yh:10.4f}   {e:10.4f}")


def print_predictions(p_values, result):
    """打印外部给定 p 的预测温度。"""
    if p_values is None:
        return

    p_values = np.asarray(p_values, dtype=float)
    y_pred = result["a"] * p_values + result["b"]

    print("\n指定含铬量下的温度预测：")
    for p, y in zip(p_values, y_pred):
        print(f"  p={float(p):.4f}% -> theta={float(y):.6f} °C")


def plot_result(result):
    """绘制散点与回归直线。"""
    x_min = float(np.min(P_DATA))
    x_max = float(np.max(P_DATA))
    x_line = np.linspace(x_min - 3.0, x_max + 3.0, 300)
    y_line = result["a"] * x_line + result["b"]

    fig, ax = plt.subplots(figsize=(9.0, 5.2))
    ax.scatter(P_DATA, THETA_DATA, color="#1f77b4", s=55, label="观测数据")
    ax.plot(x_line, y_line, color="#d62728", linewidth=2.0, label="最小二乘回归直线")

    ax.set_title("习题 5.4：theta 与 p 的线性最小二乘拟合")
    ax.set_xlabel("p (%)")
    ax.set_ylabel("theta (°C)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(predict_p=None, show_plot=True):
    result = fit_linear_least_squares(P_DATA, THETA_DATA)
    print_report(result)
    print_predictions(predict_p, result)
    if show_plot:
        plot_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.4 线性最小二乘拟合")
    parser.add_argument(
        "--predict-p",
        nargs="*",
        type=float,
        default=None,
        help="可选：输入若干 p(%) 进行预测，如 --predict-p 50 70 90",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(predict_p=args.predict_p, show_plot=not args.no_plot)

