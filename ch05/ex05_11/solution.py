"""
例题 5.11：拟合 y = a*e^x + b*ln(x)，并满足 a>=0, b>=0, a+b<=1。

运行：
  python ch05/ex05_11/solution.py
  python ch05/ex05_11/solution.py --no-plot
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


# 表 5.10 数据（整数）
X_DATA = np.array([3, 5, 6, 7, 4, 8, 5, 9], dtype=float)
Y_DATA = np.array([4, 9, 5, 3, 8, 5, 8, 5], dtype=float)


def model_value(x, a, b):
    """模型 y = a*e^x + b*ln(x)。"""
    return a * np.exp(x) + b * np.log(x)


def objective(a, b, x, y):
    """最小二乘目标函数。"""
    residual = y - model_value(x, a, b)
    return float(np.dot(residual, residual))


def _clamp(v, lo, hi):
    return float(min(max(v, lo), hi))


def fit_constrained(x, y):
    """
    带约束最小二乘（解析+边界枚举）：
      min ||y - a*exp(x) - b*ln(x)||^2
      s.t. a>=0, b>=0, a+b<=1
    该问题是二维凸二次规划，直接比较可行域内候选点可稳定得到全局最优解。
    """
    f1 = np.exp(x)
    f2 = np.log(x)

    candidates = []

    # 1) 无约束最小二乘解（若落在可行域内部）
    m = np.column_stack([f1, f2])
    theta, *_ = np.linalg.lstsq(m, y, rcond=None)
    a_u, b_u = float(theta[0]), float(theta[1])
    if a_u >= 0 and b_u >= 0 and (a_u + b_u) <= 1:
        candidates.append(("interior", a_u, b_u))

    # 2) 边界 a = 0, b in [0,1]
    b_a0 = _clamp(float(np.dot(f2, y) / np.dot(f2, f2)), 0.0, 1.0)
    candidates.append(("edge_a0", 0.0, b_a0))

    # 3) 边界 b = 0, a in [0,1]
    a_b0 = _clamp(float(np.dot(f1, y) / np.dot(f1, f1)), 0.0, 1.0)
    candidates.append(("edge_b0", a_b0, 0.0))

    # 4) 边界 a + b = 1, a in [0,1]
    p = f1 - f2
    q = y - f2
    a_ab1 = _clamp(float(np.dot(p, q) / np.dot(p, p)), 0.0, 1.0)
    b_ab1 = 1.0 - a_ab1
    candidates.append(("edge_ab1", a_ab1, b_ab1))

    # 5) 顶点（防止数值边界遗漏）
    candidates.extend(
        [
            ("vertex_00", 0.0, 0.0),
            ("vertex_10", 1.0, 0.0),
            ("vertex_01", 0.0, 1.0),
        ]
    )

    best = None
    for tag, a, b in candidates:
        sse = objective(a, b, x, y)
        if best is None or sse < best["sse"]:
            best = {"tag": tag, "a": a, "b": b, "sse": sse}

    return np.array([best["a"], best["b"]], dtype=float), best["sse"], best["tag"]


def goodness_of_fit(y_obs, y_hat):
    """计算 RMSE 和 R^2。"""
    residual = y_obs - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual**2)))
    sst = float(np.dot(y_obs - np.mean(y_obs), y_obs - np.mean(y_obs)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def print_report(a, b, sse, rmse, r2, residual, solve_tag):
    """打印拟合结果。"""
    print("=== 例题 5.11 拟合结果 ===")
    print(f"求解口径: 解析+边界枚举（最优候选={solve_tag}）")
    print(f"最优参数: a = {a:.10f}, b = {b:.10f}")
    print(f"经验公式: y = {a:.10f}*e^x + {b:.10f}*ln(x)")
    print("约束校验：")
    print(f"  a >= 0: {a >= -1e-10}")
    print(f"  b >= 0: {b >= -1e-10}")
    print(f"  a + b = {a + b:.10f} <= 1: {a + b <= 1 + 1e-10}")
    print(f"SSE = {sse:.10f}")
    print(f"RMSE = {rmse:.10f}")
    print(f"R^2 = {r2:.10f}")

    print("\n观测值/预测值/残差：")
    y_hat = model_value(X_DATA, a, b)
    print("      x      y_obs      y_hat    residual")
    for x, y, yh, e in zip(X_DATA, Y_DATA, y_hat, residual):
        print(f"{x:7.3f}{y:11.4f}{yh:11.4f}{e:12.6f}")


def plot_result(a, b):
    """绘制观测点与拟合曲线。"""
    x_curve = np.linspace(float(np.min(X_DATA)), float(np.max(X_DATA)), 400)
    y_curve = model_value(x_curve, a, b)

    fig, ax = plt.subplots(figsize=(8.5, 5.3))
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=55, label="观测点")
    ax.plot(x_curve, y_curve, color="#d62728", linewidth=2.0, label="约束最小二乘拟合曲线")
    ax.set_title("例5.11 经验函数拟合")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(show_plot=True):
    (a, b), _, solve_tag = fit_constrained(X_DATA, Y_DATA)
    y_hat = model_value(X_DATA, a, b)
    sse, rmse, r2, residual = goodness_of_fit(Y_DATA, y_hat)
    print_report(a, b, sse, rmse, r2, residual, solve_tag)
    if show_plot:
        plot_result(a, b)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.11 带约束经验函数拟合")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示拟合曲线图",
    )
    args = parser.parse_args()
    solve(show_plot=not args.no_plot)
