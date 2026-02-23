"""
习题 6.4：竖直加热板自然对流方程组数值解

方程组（eta 为相似变量）：
  f''' + 3 f f'' - 2 (f')^2 + T = 0
  T'' + 2.1 f T' = 0

初值（eta=0）：
  f(0)=0, f'(0)=0, f''(0)=0.68, T(0)=1, T'(0)=-0.5

运行示例：
  python ch06/hw06_04/solution.py
  python ch06/hw06_04/solution.py --eta-max 12 --num-points 1800
  python ch06/hw06_04/solution.py --method DOP853 --rtol 1e-9 --atol 1e-11
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


def ode_system(_eta, state, beta):
    """
    把高阶系统改写成一阶系统：
      y1=f, y2=f', y3=f'', y4=T, y5=T'
    """
    y1, y2, y3, y4, y5 = state

    dy1 = y2
    dy2 = y3
    dy3 = -3.0 * y1 * y3 + 2.0 * y2**2 - y4
    dy4 = y5
    dy5 = -beta * y1 * y5
    return np.array([dy1, dy2, dy3, dy4, dy5], dtype=float)


def solve_system(
    eta_min,
    eta_max,
    num_points,
    beta,
    method,
    rtol,
    atol,
):
    """求解 IVP。"""
    if eta_min >= eta_max:
        raise ValueError("需要满足 eta_min < eta_max。")
    if num_points < 200:
        raise ValueError("num_points 建议不小于 200。")

    eta_eval = np.linspace(eta_min, eta_max, num_points)
    y0 = np.array([0.0, 0.0, 0.68, 1.0, -0.5], dtype=float)

    sol = solve_ivp(
        fun=lambda eta, y: ode_system(eta, y, beta=beta),
        t_span=(eta_min, eta_max),
        y0=y0,
        method=method,
        t_eval=eta_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"求解失败：{sol.message}")
    return sol


def print_report(sol, eta_max, beta, method, rtol, atol):
    """输出求解统计与末端值。"""
    f_val = sol.y[0]
    fp_val = sol.y[1]
    fpp_val = sol.y[2]
    t_val = sol.y[3]
    tp_val = sol.y[4]

    print("=== 习题 6.4 求解结果（竖直加热板自然对流） ===")
    print("方程组：")
    print("  f''' + 3 f f'' - 2 (f')^2 + T = 0")
    print(f"  T'' + {beta:g} f T' = 0")
    print("初值：f(0)=0, f'(0)=0, f''(0)=0.68, T(0)=1, T'(0)=-0.5")
    print(f"区间: [0, {eta_max}]")
    print(f"方法: {method}, rtol={rtol:.1e}, atol={atol:.1e}")
    print(f"统计: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

    print("\n末端 eta=eta_max 时：")
    print(f"  f      = {f_val[-1]:.10f}")
    print(f"  f'     = {fp_val[-1]:.10f}")
    print(f"  f''    = {fpp_val[-1]:.10f}")
    print(f"  T      = {t_val[-1]:.10f}")
    print(f"  T'     = {tp_val[-1]:.10f}")

    print("\n范围统计：")
    print(f"  f(eta)  范围: [{np.min(f_val):.10f}, {np.max(f_val):.10f}]")
    print(f"  T(eta)  范围: [{np.min(t_val):.10f}, {np.max(t_val):.10f}]")


def plot_result(sol):
    """绘制 f(eta)、T(eta) 曲线。"""
    eta_grid = sol.t
    f_val = sol.y[0]
    t_val = sol.y[3]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)

    ax1.plot(eta_grid, f_val, color="#1f77b4", linewidth=2.1)
    ax1.set_title("习题 6.4：速度函数 f(eta) 与温度函数 T(eta)")
    ax1.set_ylabel("f(eta)")
    ax1.grid(alpha=0.3)

    ax2.plot(eta_grid, t_val, color="#d62728", linewidth=2.1)
    ax2.set_xlabel("eta")
    ax2.set_ylabel("T(eta)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def solve(
    eta_min=0.0,
    eta_max=10.0,
    num_points=1500,
    beta=2.1,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
    show_plot=True,
):
    """主流程：数值求解 + 结果输出 + 绘图。"""
    sol = solve_system(
        eta_min=eta_min,
        eta_max=eta_max,
        num_points=num_points,
        beta=beta,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    print_report(sol, eta_max=eta_max, beta=beta, method=method, rtol=rtol, atol=atol)
    if show_plot:
        plot_result(sol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.4：竖直加热板自然对流方程组数值求解")
    parser.add_argument("--eta-min", type=float, default=0.0, help="eta 左端点，默认 0")
    parser.add_argument("--eta-max", type=float, default=10.0, help="eta 右端点，默认 10")
    parser.add_argument("--num-points", type=int, default=1500, help="输出网格点数，默认 1500")
    parser.add_argument("--beta", type=float, default=2.1, help="第二方程系数 beta，默认 2.1")
    parser.add_argument(
        "--method",
        type=str,
        default="RK45",
        choices=["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"],
        help="solve_ivp 积分方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="相对误差容限")
    parser.add_argument("--atol", type=float, default=1e-10, help="绝对误差容限")
    parser.add_argument("--no-plot", action="store_true", help="仅数值输出，不绘图")
    args = parser.parse_args()

    solve(
        eta_min=args.eta_min,
        eta_max=args.eta_max,
        num_points=args.num_points,
        beta=args.beta,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )
