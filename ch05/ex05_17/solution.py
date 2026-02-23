"""
例题 5.17：利用给定数据拟合分段线性函数。

模型：
  y = a + b*x,  x < k
  y = c + d*x,  x >= k

运行：
  python ch05/ex05_17/solution.py
  python ch05/ex05_17/solution.py --min-segment-points 2
  python ch05/ex05_17/solution.py --no-plot
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


X_DATA = np.array([0.81, 0.91, 0.13, 0.91, 0.63, 0.098, 0.28, 0.55, 0.96, 0.96, 0.16, 0.97, 0.96], dtype=float)
Y_DATA = np.array([0.17, 0.12, 0.16, 0.0035, 0.37, 0.082, 0.34, 0.56, 0.15, -0.046, 0.17, -0.091, -0.071], dtype=float)


def fit_line(x, y):
    """线性最小二乘拟合 y = p0 + p1*x。"""
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def evaluate_k(x, y, k, min_segment_points):
    """在给定 k 下拟合并返回误差。"""
    left_mask = x < k
    right_mask = ~left_mask
    n_left = int(np.sum(left_mask))
    n_right = int(np.sum(right_mask))

    if n_left < min_segment_points or n_right < min_segment_points:
        return None

    a, b = fit_line(x[left_mask], y[left_mask])
    c, d = fit_line(x[right_mask], y[right_mask])

    y_hat = np.where(left_mask, a + b * x, c + d * x)
    residual = y - y_hat
    sse = float(np.dot(residual, residual))

    return {
        "k": float(k),
        "a": a,
        "b": b,
        "c": c,
        "d": d,
        "y_hat": y_hat,
        "residual": residual,
        "sse": sse,
        "n_left": n_left,
        "n_right": n_right,
    }


def search_best_split(x, y, min_segment_points):
    """在候选 k 上枚举，寻找 SSE 最小方案。"""
    candidates = np.unique(x)
    best = None
    for k in candidates:
        res = evaluate_k(x, y, k, min_segment_points)
        if res is None:
            continue
        if best is None or res["sse"] < best["sse"]:
            best = res

    if best is None:
        raise RuntimeError(
            "未找到可行分段点，请降低 --min-segment-points 或检查数据。"
        )
    return best


def metrics(y, y_hat):
    """计算 SSE/RMSE/R^2。"""
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def print_report(best):
    """打印拟合结果。"""
    sse, rmse, r2 = metrics(Y_DATA, best["y_hat"])

    print("=== 例题 5.17 拟合结果 ===")
    print(
        "拟合函数：\n"
        f"  y = {best['a']:.10f} + {best['b']:.10f}*x, x < {best['k']:.10f}\n"
        f"  y = {best['c']:.10f} + {best['d']:.10f}*x, x >= {best['k']:.10f}"
    )
    print(f"最优分段点: k = {best['k']:.10f}")
    print(f"左右段样本数: left={best['n_left']}, right={best['n_right']}")
    print(f"SSE  = {sse:.10f}")
    print(f"RMSE = {rmse:.10f}")
    print(f"R^2  = {r2:.10f}")

    idx_sorted = np.argsort(X_DATA)
    print("\n按 x 排序的观测/预测/残差：")
    print("        x       y_obs       y_hat    residual")
    for i in idx_sorted:
        print(
            f"{X_DATA[i]:9.4f}{Y_DATA[i]:12.5f}{best['y_hat'][i]:12.5f}{best['residual'][i]:12.6f}"
        )


def plot_result(best):
    """绘制散点和分段拟合曲线。"""
    x_min = float(np.min(X_DATA))
    x_max = float(np.max(X_DATA))
    k = best["k"]

    x_left = np.linspace(x_min, k, 200, endpoint=False) if k > x_min else np.array([])
    x_right = np.linspace(k, x_max, 200) if k < x_max else np.array([k])
    y_left = best["a"] + best["b"] * x_left if x_left.size > 0 else np.array([])
    y_right = best["c"] + best["d"] * x_right

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=45, label="观测点", zorder=3)
    if x_left.size > 0:
        ax.plot(x_left, y_left, color="#d62728", linewidth=2.2, label="左段拟合")
    ax.plot(x_right, y_right, color="#2ca02c", linewidth=2.2, label="右段拟合")
    ax.axvline(k, color="black", linestyle="--", linewidth=1.4, label=f"k={k:.4f}")

    ax.set_title("例5.17 分段线性拟合结果")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(min_segment_points, show_plot):
    best = search_best_split(X_DATA, Y_DATA, min_segment_points=min_segment_points)
    print_report(best)
    if show_plot:
        plot_result(best)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.17 分段线性函数拟合")
    parser.add_argument(
        "--min-segment-points",
        type=int,
        default=2,
        help="每个分段至少样本数，默认 2",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()
    solve(min_segment_points=args.min_segment_points, show_plot=not args.no_plot)
