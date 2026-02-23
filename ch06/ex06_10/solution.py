"""
例题 6.10：二阶非线性微分方程数值求解

方程：
  (1-x) y'' = (1/5)*sqrt(1 + (y')^2), 0<x<=1
初值：
  y(0)=0, y'(0)=0

运行示例：
  python ch06/ex06_10/solution.py
  python ch06/ex06_10/solution.py --method DOP853 --eps 1e-8
  python ch06/ex06_10/solution.py --no-reference --no-plot
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


def ode_rhs(x, state):
    """一阶系统右端：state=[y,p], p=y'。"""
    y_val, p_val = state
    _ = y_val  # y 在右端不显式出现，保留变量以保持可读性
    dy = p_val
    dp = np.sqrt(1.0 + p_val ** 2) / (5.0 * (1.0 - x))
    return [dy, dp]


def reference_solution(x):
    """
    参考解析式（用于误差校验）：
      y' = 0.5[(1-x)^(-1/5) - (1-x)^(1/5)]
      y  = 5/24 - 5/8*(1-x)^(4/5) + 5/12*(1-x)^(6/5)
    """
    x = np.asarray(x, dtype=float)
    u = 1.0 - x
    p_ref = 0.5 * (u ** (-0.2) - u ** 0.2)
    y_ref = 5.0 / 24.0 - 5.0 / 8.0 * (u ** 0.8) + 5.0 / 12.0 * (u ** 1.2)
    return y_ref, p_ref


def solve_numerical(method="RK45", eps=1e-6, n_grid=500, rtol=1e-9, atol=1e-12):
    """在 [0,1-eps] 上求数值解。"""
    if eps <= 0 or eps >= 1:
        raise ValueError("eps 必须满足 0 < eps < 1。")
    if n_grid < 20:
        raise ValueError("n_grid 建议不小于 20。")

    x0 = 0.0
    x1 = 1.0 - eps
    x_eval = np.linspace(x0, x1, n_grid)

    sol = solve_ivp(
        fun=ode_rhs,
        t_span=(x0, x1),
        y0=[0.0, 0.0],
        t_eval=x_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"数值积分失败：{sol.message}")

    y_num = sol.y[0]
    p_num = sol.y[1]
    return x_eval, y_num, p_num, sol


def print_report(x_eval, y_num, p_num, solver_result, method, eps, with_reference):
    """打印数值结果与可选误差指标。"""
    print("=== 例题 6.10 数值解结果 ===")
    print("方程: (1-x)*y'' = (1/5)*sqrt(1+(y')^2)")
    print("初值: y(0)=0, y'(0)=0")
    print(f"积分区间: [0, 1-eps], eps={eps}")
    print("\n求解器信息：")
    print(f"  method={method}")
    print(f"  nfev={solver_result.nfev}, njev={solver_result.njev}, nlu={solver_result.nlu}")

    print("\n末端值（x=1-eps）：")
    print(f"  x_end   = {x_eval[-1]:.10f}")
    print(f"  y(x_end)= {y_num[-1]:.12f}")
    print(f"  y'(x_end)= {p_num[-1]:.12f}")

    if with_reference:
        y_ref, p_ref = reference_solution(x_eval)
        err_y = y_num - y_ref
        err_p = p_num - p_ref

        print("\n与参考解析式对比误差：")
        print(f"  max|y_num-y_ref|   = {np.max(np.abs(err_y)):.12e}")
        print(f"  RMSE(y)            = {np.sqrt(np.mean(err_y ** 2)):.12e}")
        print(f"  max|yp_num-yp_ref| = {np.max(np.abs(err_p)):.12e}")
        print(f"  RMSE(y')           = {np.sqrt(np.mean(err_p ** 2)):.12e}")

    sample_points = np.array([0.0, 0.25, 0.5, 0.75, x_eval[-1]], dtype=float)
    y_sample = np.interp(sample_points, x_eval, y_num)
    p_sample = np.interp(sample_points, x_eval, p_num)
    print("\n样例点：")
    print("       x            y(x)          y'(x)")
    for xv, yv, pv in zip(sample_points, y_sample, p_sample):
        print(f"  {xv:10.6f}   {yv:12.8f}   {pv:12.8f}")


def plot_result(x_eval, y_num, p_num, with_reference):
    """绘制 y(x)、y'(x) 及可选误差曲线。"""
    if with_reference:
        y_ref, p_ref = reference_solution(x_eval)
    else:
        y_ref, p_ref = None, None

    if with_reference:
        fig, axes = plt.subplots(3, 1, figsize=(9.5, 9.0), sharex=True)
    else:
        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.8), sharex=True)

    axes[0].plot(x_eval, y_num, color="#1f77b4", linewidth=2.0, label="数值解 y(x)")
    if with_reference:
        axes[0].plot(x_eval, y_ref, color="#d62728", linewidth=1.8, linestyle="--", label="参考解 y_ref(x)")
    axes[0].set_ylabel("y")
    axes[0].set_title("例题 6.10 数值解")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(x_eval, p_num, color="#2ca02c", linewidth=2.0, label="数值解 y'(x)")
    if with_reference:
        axes[1].plot(x_eval, p_ref, color="#9467bd", linewidth=1.8, linestyle="--", label="参考解 y'_ref(x)")
    axes[1].set_ylabel("y'")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    if with_reference:
        err_y = y_num - y_ref
        axes[2].plot(x_eval, err_y, color="#ff7f0e", linewidth=1.8, label="误差 y_num-y_ref")
        axes[2].axhline(0.0, color="black", linewidth=1.0)
        axes[2].set_ylabel("误差")
        axes[2].set_xlabel("x")
        axes[2].grid(alpha=0.3)
        axes[2].legend()
    else:
        axes[1].set_xlabel("x")

    plt.tight_layout()
    plt.show()


def solve(method="RK45", eps=1e-6, n_grid=500, rtol=1e-9, atol=1e-12, with_reference=True, show_plot=True):
    x_eval, y_num, p_num, solver_result = solve_numerical(
        method=method,
        eps=eps,
        n_grid=n_grid,
        rtol=rtol,
        atol=atol,
    )
    print_report(
        x_eval=x_eval,
        y_num=y_num,
        p_num=p_num,
        solver_result=solver_result,
        method=method,
        eps=eps,
        with_reference=with_reference,
    )
    if show_plot:
        plot_result(x_eval, y_num, p_num, with_reference=with_reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.10 奇点附近二阶非线性方程数值求解")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--eps", type=float, default=1e-6, help="终点回避奇点的偏移量 eps，默认 1e-6")
    parser.add_argument("--n-grid", type=int, default=500, help="输出网格点数，默认 500")
    parser.add_argument("--rtol", type=float, default=1e-9, help="相对容差，默认 1e-9")
    parser.add_argument("--atol", type=float, default=1e-12, help="绝对容差，默认 1e-12")
    parser.add_argument("--no-reference", action="store_true", help="不计算参考解析式对比")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        method=args.method,
        eps=args.eps,
        n_grid=max(20, args.n_grid),
        rtol=args.rtol,
        atol=args.atol,
        with_reference=not args.no_reference,
        show_plot=not args.no_plot,
    )

