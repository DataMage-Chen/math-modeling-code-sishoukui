"""
习题 6.3：微分方程组的数值解与轨线绘制

方程组：
  x' = -x^3 - y,   x(0)=1
  y' =  x - y^3,   y(0)=0.5
  0 <= t <= 30

运行示例：
  python ch06/hw06_03/solution.py
  python ch06/hw06_03/solution.py --method DOP853 --num-points 3000
  python ch06/hw06_03/solution.py --t-end 40 --x0 1.2 --y0 0.3
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


def ode_system(_t, state):
    """题目给定的二维非线性自治系统。"""
    x_val, y_val = state
    dx = -x_val**3 - y_val
    dy = x_val - y_val**3
    return np.array([dx, dy], dtype=float)


def solve_system(t_start, t_end, x0, y0, num_points, method, rtol, atol):
    """调用 solve_ivp 求解，并返回网格解。"""
    if t_start >= t_end:
        raise ValueError("需要满足 t_start < t_end。")
    if num_points < 200:
        raise ValueError("num_points 建议不小于 200。")

    t_eval = np.linspace(t_start, t_end, num_points)
    init_state = np.array([x0, y0], dtype=float)

    sol = solve_ivp(
        ode_system,
        (t_start, t_end),
        init_state,
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"数值求解失败：{sol.message}")

    return sol


def print_report(sol, t_start, t_end, x0, y0, method, rtol, atol):
    """打印核心求解信息。"""
    x_series = sol.y[0]
    y_series = sol.y[1]

    print("=== 习题 6.3 求解结果 ===")
    print("方程组：")
    print("  x' = -x^3 - y")
    print("  y' =  x - y^3")
    print(f"初值: x({t_start:g})={x0}, y({t_start:g})={y0}")
    print(f"时间区间: [{t_start}, {t_end}]")
    print(f"方法: {method}, rtol={rtol:.1e}, atol={atol:.1e}")
    print(f"状态: success={sol.success}, message={sol.message}")
    print(f"统计: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

    print("\n末时刻状态：")
    print(f"  x({t_end:g}) = {x_series[-1]:.10f}")
    print(f"  y({t_end:g}) = {y_series[-1]:.10f}")

    print("\n范围统计：")
    print(f"  x(t) 范围: [{np.min(x_series):.10f}, {np.max(x_series):.10f}]")
    print(f"  y(t) 范围: [{np.min(y_series):.10f}, {np.max(y_series):.10f}]")


def plot_result(sol):
    """绘制 x(t), y(t) 时间曲线与相平面轨线。"""
    t_grid = sol.t
    x_series = sol.y[0]
    y_series = sol.y[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    ax1.plot(t_grid, x_series, color="#1f77b4", linewidth=2.0, label="x(t)")
    ax1.plot(t_grid, y_series, color="#d62728", linewidth=2.0, label="y(t)")
    ax1.set_title("习题 6.3：时间域解曲线")
    ax1.set_xlabel("t")
    ax1.set_ylabel("状态变量")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x_series, y_series, color="#2ca02c", linewidth=2.0, label="轨线")
    ax2.scatter([x_series[0]], [y_series[0]], color="#ff7f0e", s=55, marker="o", label="起点")
    ax2.scatter([x_series[-1]], [y_series[-1]], color="#9467bd", s=55, marker="s", label="终点")
    ax2.set_title("相平面轨线")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    t_start=0.0,
    t_end=30.0,
    x0=1.0,
    y0=0.5,
    num_points=2000,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
    show_plot=True,
):
    """主流程：数值求解 + 报告输出 + 绘图。"""
    sol = solve_system(
        t_start=t_start,
        t_end=t_end,
        x0=x0,
        y0=y0,
        num_points=num_points,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    print_report(sol, t_start, t_end, x0, y0, method, rtol, atol)

    if show_plot:
        plot_result(sol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.3：二维非线性系统的数值解")
    parser.add_argument("--t-start", type=float, default=0.0, help="起始时刻，默认 0")
    parser.add_argument("--t-end", type=float, default=30.0, help="终止时刻，默认 30")
    parser.add_argument("--x0", type=float, default=1.0, help="初值 x(0)，默认 1")
    parser.add_argument("--y0", type=float, default=0.5, help="初值 y(0)，默认 0.5")
    parser.add_argument("--num-points", type=int, default=2000, help="输出网格点数，默认 2000")
    parser.add_argument(
        "--method",
        type=str,
        default="RK45",
        choices=["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"],
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="相对误差容限")
    parser.add_argument("--atol", type=float, default=1e-10, help="绝对误差容限")
    parser.add_argument("--no-plot", action="store_true", help="只输出结果，不绘图")
    args = parser.parse_args()

    solve(
        t_start=args.t_start,
        t_end=args.t_end,
        x0=args.x0,
        y0=args.y0,
        num_points=args.num_points,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )
