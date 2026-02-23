"""
习题 6.9：小船过河（追踪对岸固定目标点）

问题：
  河宽 d，水流速度 v1，船在静水中速度 v2，k=v1/v2。
  小船始终把船头指向对岸固定点 B（B 与起点 A 正对）。

任务：
  1) 给出航线方程解析解；
  2) 对 d=100, v1=1, v2=2 做数值求解，给出渡河时间、任意时刻位置并与解析解比较。

运行示例：
  python ch06/hw06_09/solution.py
  python ch06/hw06_09/solution.py --d 100 --v1 1 --v2 2 --num-points 1500
  python ch06/hw06_09/solution.py --query-times 0 10 20 30 40 50 60
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


def analytic_x_of_y(y, d, k):
    """
    解析航线（隐式变量替换后的显式形式）：
      x = u * sinh(k * ln(d/u)), u=d-y.
    """
    y_arr = np.asarray(y, dtype=float)
    u = np.maximum(d - y_arr, 1e-14)
    x = u * np.sinh(k * np.log(d / u))
    x = np.where(y_arr >= d, 0.0, x)
    return x


def analytic_t_of_y(y, d, v2, k):
    """
    解析时间关系 t(y)：
      t = d/(2v2) * [(1-r^(1-k))/(1-k) + (1-r^(1+k))/(1+k)],
      r=(d-y)/d.
    """
    y_arr = np.asarray(y, dtype=float)
    r = np.clip((d - y_arr) / d, 0.0, 1.0)
    term1 = (1.0 - r ** (1.0 - k)) / (1.0 - k)
    term2 = (1.0 - r ** (1.0 + k)) / (1.0 + k)
    return d / (2.0 * v2) * (term1 + term2)


def analytic_cross_time(d, v1, v2):
    """解析渡河时间 T=d/(v2*(1-k^2))，k=v1/v2。"""
    k = v1 / v2
    return d / (v2 * (1.0 - k**2))


def boat_ode(_t, state, d, v1, v2):
    """
    状态方程：
      x' = v1 - v2*x/r
      y' = v2*(d-y)/r
      r = sqrt(x^2 + (d-y)^2)
    """
    x_val, y_val = state
    r = np.hypot(x_val, d - y_val)
    if r < 1e-12:
        return np.array([0.0, 0.0], dtype=float)
    dx = v1 - v2 * x_val / r
    dy = v2 * (d - y_val) / r
    return np.array([dx, dy], dtype=float)


def make_reach_event(d):
    """创建到达对岸目标点 B 的事件函数（保留 terminal/direction 属性）。"""

    def event(_t, state):
        return state[1] - d

    event.terminal = True
    event.direction = 1
    return event


def make_near_target_event(d, r_tol):
    """
    创建“接近目标点”保护事件，避免在 r->0 附近反复缩步。
    当 sqrt(x^2+(d-y)^2)=r_tol 时停止积分。
    """

    def event(_t, state):
        return np.hypot(state[0], d - state[1]) - r_tol

    event.terminal = True
    event.direction = -1
    return event


def solve_numeric(d, v1, v2, num_points, method, rtol, atol, r_tol=1e-4):
    """数值积分并返回轨迹。"""
    t_ana = analytic_cross_time(d, v1, v2)
    t_end = 1.2 * t_ana
    t_eval = np.linspace(0.0, t_end, num_points)
    reach_event = make_reach_event(d=d)
    near_event = make_near_target_event(d=d, r_tol=r_tol)

    sol = solve_ivp(
        fun=lambda t, z: boat_ode(t, z, d=d, v1=v1, v2=v2),
        t_span=(0.0, t_end),
        y0=np.array([0.0, 0.0], dtype=float),
        method=method,
        t_eval=t_eval,
        events=[reach_event, near_event],
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"数值求解失败：{sol.message}")

    t_cross = float(sol.t[-1])
    stop_reason = "t_span_end"
    if sol.t_events:
        if len(sol.t_events[0]) > 0:
            t_cross = float(sol.t_events[0][0])
            stop_reason = "reach_B"
        elif len(sol.t_events) > 1 and len(sol.t_events[1]) > 0:
            t_cross = float(sol.t_events[1][0])
            stop_reason = "near_B"
    return sol, t_cross, stop_reason


def compare_trajectory(sol, d, v1, v2):
    """在同一 y 网格上对比数值轨迹与解析轨迹。"""
    k = v1 / v2
    y_num = sol.y[1]
    x_num = sol.y[0]

    y_max = min(d, float(np.max(y_num)))
    y_grid = np.linspace(0.0, max(1e-8, y_max), 600)
    x_num_interp = np.interp(y_grid, y_num, x_num)
    x_ana = analytic_x_of_y(y_grid, d=d, k=k)
    max_abs_err = float(np.max(np.abs(x_num_interp - x_ana)))
    rmse = float(np.sqrt(np.mean((x_num_interp - x_ana) ** 2)))

    return y_grid, x_num_interp, x_ana, max_abs_err, rmse


def analytic_position_at_times(t_query, d, v1, v2):
    """通过 t(y) 反插值得到解析位置 (x(t), y(t))。"""
    k = v1 / v2
    y_dense = np.linspace(0.0, d, 10000)
    t_dense = analytic_t_of_y(y_dense, d=d, v2=v2, k=k)
    y_q = np.interp(t_query, t_dense, y_dense)
    x_q = analytic_x_of_y(y_q, d=d, k=k)
    return x_q, y_q


def print_report(
    d,
    v1,
    v2,
    method,
    rtol,
    atol,
    t_cross_num,
    t_cross_ana,
    stop_reason,
    max_abs_err,
    rmse,
    query_times,
    x_num_q,
    y_num_q,
    x_ana_q,
    y_ana_q,
):
    """打印核心结果。"""
    k = v1 / v2
    print("=== 习题 6.9 求解结果（小船过河） ===")
    print(f"参数: d={d} m, v1={v1} m/s, v2={v2} m/s, k=v1/v2={k:.6f}")
    print(f"数值方法: {method}, rtol={rtol:.1e}, atol={atol:.1e}")
    print("解析航线: x = (d-y) * sinh(k * ln(d/(d-y)))")
    print("解析渡河时间: T = d / (v2*(1-k^2))")

    print("\n渡河时间比较：")
    print(f"  解析时间 T_ana = {t_cross_ana:.10f} s")
    print(f"  数值时间 T_num = {t_cross_num:.10f} s")
    print(f"  差值 T_num-T_ana = {t_cross_num - t_cross_ana:.6e} s")
    print(f"  数值停止原因 = {stop_reason}")

    print("\n航线对比误差（按 y 插值比较 x）：")
    print(f"  max|x_num-x_ana| = {max_abs_err:.6e} m")
    print(f"  RMSE             = {rmse:.6e} m")

    print("\n任意时刻位置（数值 vs 解析）：")
    print("  t(s)      x_num      y_num      x_ana      y_ana      |dx|       |dy|")
    for i, t_val in enumerate(query_times):
        dx = abs(x_num_q[i] - x_ana_q[i])
        dy = abs(y_num_q[i] - y_ana_q[i])
        print(
            f"  {t_val:6.2f}  "
            f"{x_num_q[i]:9.4f}  {y_num_q[i]:9.4f}  "
            f"{x_ana_q[i]:9.4f}  {y_ana_q[i]:9.4f}  "
            f"{dx:9.2e}  {dy:9.2e}"
        )


def plot_result(sol, d, t_cross_ana, t_cross_num, y_grid, x_ana):
    """绘制轨迹图和时程图。"""
    t = sol.t
    x_num = sol.y[0]
    y_num = sol.y[1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax1.plot(x_num, y_num, color="#1f77b4", linewidth=2.2, label="数值航线")
    ax1.plot(x_ana, y_grid, color="#d62728", linestyle="--", linewidth=1.9, label="解析航线")
    ax1.scatter([0.0], [0.0], color="#2ca02c", s=50, label="A(0,0)")
    ax1.scatter([0.0], [d], color="#9467bd", s=50, label="B(0,d)")
    ax1.set_title("小船航线：数值解与解析解")
    ax1.set_xlabel("x（顺流方向, m）")
    ax1.set_ylabel("y（横渡方向, m）")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(t, x_num, color="#ff7f0e", linewidth=2.0, label="x_num(t)")
    ax2.plot(t, y_num, color="#1f77b4", linewidth=2.0, label="y_num(t)")
    ax2.axvline(t_cross_ana, color="#d62728", linestyle="--", linewidth=1.4, label="T_ana")
    ax2.axvline(t_cross_num, color="#2ca02c", linestyle=":", linewidth=1.4, label="T_num")
    ax2.set_title("位置时程（数值解）")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("位置 (m)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    d=100.0,
    v1=1.0,
    v2=2.0,
    num_points=1200,
    method="DOP853",
    rtol=1e-7,
    atol=1e-9,
    r_tol=1e-4,
    query_times=None,
    show_plot=True,
):
    """主流程：解析式 + 数值解 + 比较。"""
    if d <= 0:
        raise ValueError("d 必须为正。")
    if v1 < 0 or v2 <= 0:
        raise ValueError("速度要求 v1>=0 且 v2>0。")

    k = v1 / v2
    if k >= 1.0:
        raise ValueError("该追踪策略下需满足 k=v1/v2<1 才能在有限时间到达 B。")
    if num_points < 300:
        raise ValueError("num_points 建议不小于 300。")

    sol, t_cross_num, stop_reason = solve_numeric(
        d=d, v1=v1, v2=v2, num_points=num_points, method=method, rtol=rtol, atol=atol, r_tol=r_tol
    )
    t_cross_ana = analytic_cross_time(d=d, v1=v1, v2=v2)

    y_grid, _x_num_interp, x_ana, max_abs_err, rmse = compare_trajectory(sol, d=d, v1=v1, v2=v2)

    if query_times is None or len(query_times) == 0:
        query_times = np.linspace(0.0, t_cross_ana, 6)
    else:
        query_times = np.asarray(query_times, dtype=float)
        query_times = np.clip(query_times, 0.0, t_cross_ana)

    x_num_q = np.interp(query_times, sol.t, sol.y[0])
    y_num_q = np.interp(query_times, sol.t, sol.y[1])
    x_ana_q, y_ana_q = analytic_position_at_times(query_times, d=d, v1=v1, v2=v2)

    print_report(
        d=d,
        v1=v1,
        v2=v2,
        method=method,
        rtol=rtol,
        atol=atol,
        t_cross_num=t_cross_num,
        t_cross_ana=t_cross_ana,
        stop_reason=stop_reason,
        max_abs_err=max_abs_err,
        rmse=rmse,
        query_times=query_times,
        x_num_q=x_num_q,
        y_num_q=y_num_q,
        x_ana_q=x_ana_q,
        y_ana_q=y_ana_q,
    )

    if show_plot:
        plot_result(sol, d=d, t_cross_ana=t_cross_ana, t_cross_num=t_cross_num, y_grid=y_grid, x_ana=x_ana)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.9：小船过河航线方程与数值比较")
    parser.add_argument("--d", type=float, default=100.0, help="河宽 d（m），默认 100")
    parser.add_argument("--v1", type=float, default=1.0, help="流速 v1（m/s），默认 1")
    parser.add_argument("--v2", type=float, default=2.0, help="静水船速 v2（m/s），默认 2")
    parser.add_argument("--num-points", type=int, default=1200, help="积分输出点数，默认 1200")
    parser.add_argument(
        "--method",
        type=str,
        default="DOP853",
        choices=["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"],
        help="solve_ivp 积分方法，默认 DOP853",
    )
    parser.add_argument("--rtol", type=float, default=1e-7, help="相对误差容限")
    parser.add_argument("--atol", type=float, default=1e-9, help="绝对误差容限")
    parser.add_argument("--r-tol-target", type=float, default=1e-4, help="接近目标点的保护半径")
    parser.add_argument(
        "--query-times",
        nargs="*",
        type=float,
        default=None,
        help="要输出位置的时刻列表（秒）",
    )
    parser.add_argument("--no-plot", action="store_true", help="仅输出文本结果，不绘图")
    args = parser.parse_args()

    solve(
        d=args.d,
        v1=args.v1,
        v2=args.v2,
        num_points=args.num_points,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        r_tol=args.r_tol_target,
        query_times=args.query_times,
        show_plot=not args.no_plot,
    )
