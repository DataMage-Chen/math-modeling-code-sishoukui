"""
例题 6.20：捕食者-被捕食者系统周期与极值分析。

运行示例：
  python ch06/ex06_20/solution.py
  python ch06/ex06_20/solution.py --t-end 300 --n-grid 60000
  python ch06/ex06_20/solution.py --method DOP853 --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
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


# 模型参数
A = 0.2
B = 0.005
C = 0.5
D = 0.01
X0 = 70.0
Y0 = 40.0


def rhs(_, state):
    """Lotka-Volterra 方程右端。"""
    x, y = state
    dx = A * x - B * x * y
    dy = -C * y + D * x * y
    return [dx, dy]


def solve_system(t_end=220.0, n_grid=50000, method="RK45", rtol=1e-9, atol=1e-12):
    """数值积分系统。"""
    t_eval = np.linspace(0.0, t_end, n_grid)
    sol = solve_ivp(
        fun=rhs,
        t_span=(0.0, t_end),
        y0=[X0, Y0],
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"积分失败：{sol.message}")
    return sol.t, sol.y[0], sol.y[1], sol


def find_extrema_indices(series):
    """寻找局部极大/极小值索引（不包含端点）。"""
    max_idx = []
    min_idx = []
    for i in range(1, len(series) - 1):
        left = series[i - 1]
        mid = series[i]
        right = series[i + 1]
        if mid > left and mid >= right:
            max_idx.append(i)
        if mid < left and mid <= right:
            min_idx.append(i)
    return np.array(max_idx, dtype=int), np.array(min_idx, dtype=int)


def estimate_period_and_extrema(t, series):
    """由峰谷估计周期与极值。"""
    max_idx, min_idx = find_extrema_indices(series)

    result = {
        "max_idx": max_idx,
        "min_idx": min_idx,
        "period": np.nan,
        "max_value": np.nan,
        "min_value": np.nan,
    }

    if max_idx.size >= 2:
        peak_times = t[max_idx]
        result["period"] = float(np.mean(np.diff(peak_times)))
        result["max_value"] = float(np.mean(series[max_idx]))
    elif max_idx.size == 1:
        result["max_value"] = float(series[max_idx[0]])
    else:
        result["max_value"] = float(np.max(series))

    if min_idx.size >= 1:
        result["min_value"] = float(np.mean(series[min_idx]))
    else:
        result["min_value"] = float(np.min(series))

    return result


def print_report(t, x, y, sol, x_stats, y_stats, method):
    """打印结果。"""
    print("=== 例题 6.20 求解结果（捕食者-被捕食者） ===")
    print("方程:")
    print("  dx/dt = 0.2x - 0.005xy, x(0)=70")
    print("  dy/dt = -0.5y + 0.01xy, y(0)=40")
    print(f"方法: {method}")
    print(f"积分区间: [0, {t[-1]}], 网格点数: {t.size}")
    print(f"求解器统计: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

    print("\n(1) 周期估计：")
    print(f"  T_x ≈ {x_stats['period']:.10f} 月")
    print(f"  T_y ≈ {y_stats['period']:.10f} 月")
    if np.isfinite(x_stats["period"]) and np.isfinite(y_stats["period"]):
        print(f"  平均周期 ≈ {(x_stats['period'] + y_stats['period']) * 0.5:.10f} 月")

    print("\n(2) x(t) 极值估计：")
    print(f"  x_max ≈ {x_stats['max_value']:.10f}")
    print(f"  x_min ≈ {x_stats['min_value']:.10f}")

    print("\n(3) y(t) 极值估计：")
    print(f"  y_max ≈ {y_stats['max_value']:.10f}")
    print(f"  y_min ≈ {y_stats['min_value']:.10f}")

    # 输出最后时刻值作参考
    print("\n末时刻状态：")
    print(f"  x(T) = {x[-1]:.10f}, y(T) = {y[-1]:.10f}")


def plot_result(t, x, y, x_stats, y_stats):
    """绘制时间序列和相图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    ax1.plot(t, x, color="#1f77b4", linewidth=1.7, label="兔子 x(t)")
    ax1.plot(t, y, color="#d62728", linewidth=1.7, label="狐狸 y(t)")

    if x_stats["max_idx"].size > 0:
        ax1.scatter(t[x_stats["max_idx"]], x[x_stats["max_idx"]], color="#1f77b4", s=16, alpha=0.6, label="x 峰值点")
    if y_stats["max_idx"].size > 0:
        ax1.scatter(t[y_stats["max_idx"]], y[y_stats["max_idx"]], color="#d62728", s=16, alpha=0.6, label="y 峰值点")

    ax1.set_title("种群数量时间演化")
    ax1.set_xlabel("t（月）")
    ax1.set_ylabel("数量")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2.plot(x, y, color="#2ca02c", linewidth=1.8, label="相轨线")
    ax2.scatter([X0], [Y0], color="black", s=45, marker="x", label="初始点")
    ax2.set_title("相图（x-y）")
    ax2.set_xlabel("兔子 x")
    ax2.set_ylabel("狐狸 y")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(t_end=220.0, n_grid=50000, method="RK45", rtol=1e-9, atol=1e-12, show_plot=True):
    t, x, y, sol = solve_system(
        t_end=t_end,
        n_grid=n_grid,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    x_stats = estimate_period_and_extrema(t, x)
    y_stats = estimate_period_and_extrema(t, y)
    print_report(t, x, y, sol, x_stats, y_stats, method=method)
    if show_plot:
        plot_result(t, x, y, x_stats, y_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.20 捕食者-被捕食者周期与极值分析")
    parser.add_argument("--t-end", type=float, default=220.0, help="积分终止时间（月），默认 220")
    parser.add_argument("--n-grid", type=int, default=50000, help="时间网格点数，默认 50000")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="积分方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-9, help="相对容差，默认 1e-9")
    parser.add_argument("--atol", type=float, default=1e-12, help="绝对容差，默认 1e-12")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        t_end=args.t_end,
        n_grid=max(1000, args.n_grid),
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )

