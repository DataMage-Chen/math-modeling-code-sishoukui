"""
例题 6.12：平面追击问题数值复现。

慢跑者轨迹：
  X(t)=10+20cos(t), Y(t)=20+15sin(t)
狗的运动：
  从原点出发，速度大小恒为 w，方向始终指向慢跑者

运行示例：
  python ch06/ex06_12/solution.py
  python ch06/ex06_12/solution.py --w-list 20 5 --t-end 120
  python ch06/ex06_12/solution.py --capture-tol 1e-3 --no-plot
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


def jogger_position(t):
    """慢跑者位置 R(t)=(X(t),Y(t))。"""
    t = np.asarray(t, dtype=float)
    x = 10.0 + 20.0 * np.cos(t)
    y = 20.0 + 15.0 * np.sin(t)
    return x, y


def jogger_speed(t):
    """慢跑者瞬时速度大小。"""
    t = np.asarray(t, dtype=float)
    vx = -20.0 * np.sin(t)
    vy = 15.0 * np.cos(t)
    return np.sqrt(vx ** 2 + vy ** 2)


def dog_rhs(t, state, w):
    """狗的追击微分方程。"""
    x, y = state
    xr, yr = jogger_position(t)
    dx = xr - x
    dy = yr - y
    dist = float(np.hypot(dx, dy))
    if dist < 1e-14:
        return [0.0, 0.0]
    return [w * dx / dist, w * dy / dist]


def simulate_for_w(w, t_end, capture_tol, method, rtol, atol, n_plot):
    """对给定 w 做一次追击仿真。"""
    if w <= 0:
        raise ValueError("w 必须为正数。")

    def capture_event(t, state):
        xr, yr = jogger_position(t)
        return np.hypot(xr - state[0], yr - state[1]) - capture_tol

    capture_event.terminal = True
    capture_event.direction = -1

    sol = solve_ivp(
        fun=lambda t, s: dog_rhs(t, s, w=w),
        t_span=(0.0, t_end),
        y0=[0.0, 0.0],
        method=method,
        rtol=rtol,
        atol=atol,
        dense_output=True,
        events=capture_event,
    )
    if not sol.success:
        raise RuntimeError(f"w={w} 积分失败：{sol.message}")

    captured = len(sol.t_events[0]) > 0
    t_stop = float(sol.t_events[0][0]) if captured else float(t_end)

    t_plot = np.linspace(0.0, t_stop, n_plot)
    dog_xy = sol.sol(t_plot)
    dog_x = dog_xy[0]
    dog_y = dog_xy[1]

    jog_x, jog_y = jogger_position(t_plot)
    dist = np.hypot(jog_x - dog_x, jog_y - dog_y)
    idx_min = int(np.argmin(dist))

    result = {
        "w": float(w),
        "captured": captured,
        "t_stop": t_stop,
        "t_plot": t_plot,
        "dog_x": dog_x,
        "dog_y": dog_y,
        "jog_x": jog_x,
        "jog_y": jog_y,
        "dist": dist,
        "t_min_dist": float(t_plot[idx_min]),
        "min_dist": float(dist[idx_min]),
        "solver_nfev": int(sol.nfev),
    }

    if captured:
        dog_cap = sol.y_events[0][0]
        jog_cap = jogger_position(t_stop)
        result.update(
            {
                "t_capture": t_stop,
                "dog_capture_xy": (float(dog_cap[0]), float(dog_cap[1])),
                "jog_capture_xy": (float(jog_cap[0]), float(jog_cap[1])),
            }
        )
    return result


def print_report(results, t_end, capture_tol, method):
    """打印求解结果。"""
    print("=== 例题 6.12 求解结果（平面追击） ===")
    print("慢跑者轨迹: X(t)=10+20cos(t), Y(t)=20+15sin(t)")
    print("狗初值: (0,0)，方向始终指向慢跑者")
    print(f"积分方法: {method}")
    print(f"时间上限: {t_end}, 捕获阈值: {capture_tol}")
    print("慢跑者速度范围理论上在 [15,20]（单位与 w 一致）")

    for res in results:
        w = res["w"]
        print(f"\n--- w = {w} ---")
        print(f"  求解函数评估次数 nfev = {res['solver_nfev']}")
        if res["captured"]:
            dcx, dcy = res["dog_capture_xy"]
            jcx, jcy = res["jog_capture_xy"]
            gap = np.hypot(dcx - jcx, dcy - jcy)
            print(f"  在 t = {res['t_capture']:.10f} 时捕获")
            print(f"  狗位置: ({dcx:.8f}, {dcy:.8f})")
            print(f"  人位置: ({jcx:.8f}, {jcy:.8f})")
            print(f"  捕获点距离校验: {gap:.3e}")
        else:
            print("  在给定时间上限内未捕获")
            print(f"  最小距离 = {res['min_dist']:.10f} (发生在 t={res['t_min_dist']:.6f})")


def plot_result(results, t_end, capture_tol):
    """绘制轨迹图与距离演化图。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))

    # 轨迹图
    t_runner = np.linspace(0.0, t_end, 2400)
    x_runner, y_runner = jogger_position(t_runner)
    ax1.plot(x_runner, y_runner, color="black", linestyle="--", linewidth=1.5, label="慢跑者轨迹")
    ax1.scatter([0.0], [0.0], color="#7f7f7f", s=45, zorder=5, label="狗起点")
    ax1.scatter([x_runner[0]], [y_runner[0]], color="#000000", s=45, zorder=5, label="慢跑者起点")

    colors = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        w = res["w"]
        ax1.plot(res["dog_x"], res["dog_y"], color=color, linewidth=2.0, label=f"狗轨迹 w={w}")
        if res["captured"]:
            dcx, dcy = res["dog_capture_xy"]
            ax1.scatter([dcx], [dcy], color=color, marker="x", s=70, zorder=6, label=f"捕获点 w={w}")

    ax1.set_title("平面追击轨迹")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.axis("equal")
    ax1.legend(loc="best", fontsize=9)

    # 距离图
    for i, res in enumerate(results):
        color = colors[i % len(colors)]
        ax2.plot(res["t_plot"], res["dist"], color=color, linewidth=1.8, label=f"w={res['w']}")
        if res["captured"]:
            ax2.scatter([res["t_capture"]], [capture_tol], color=color, marker="x", s=55, zorder=5)

    ax2.axhline(capture_tol, color="black", linestyle="--", linewidth=1.2, label="捕获阈值")
    ax2.set_title("狗-慢跑者距离随时间变化")
    ax2.set_xlabel("t")
    ax2.set_ylabel("距离")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    w_list=None,
    t_end=100.0,
    capture_tol=1e-2,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
    n_plot=3000,
    show_plot=True,
):
    if w_list is None:
        w_list = [20.0, 5.0]

    results = []
    for w in w_list:
        results.append(
            simulate_for_w(
                w=float(w),
                t_end=t_end,
                capture_tol=capture_tol,
                method=method,
                rtol=rtol,
                atol=atol,
                n_plot=n_plot,
            )
        )

    print_report(results, t_end=t_end, capture_tol=capture_tol, method=method)
    if show_plot:
        plot_result(results, t_end=t_end, capture_tol=capture_tol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.12 平面追击问题数值复现")
    parser.add_argument("--w-list", nargs="+", type=float, default=[20.0, 5.0], help="狗速率列表，默认 20 5")
    parser.add_argument("--t-end", type=float, default=100.0, help="积分时间上限，默认 100")
    parser.add_argument("--capture-tol", type=float, default=1e-2, help="捕获阈值，默认 1e-2")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="相对容差，默认 1e-8")
    parser.add_argument("--atol", type=float, default=1e-10, help="绝对容差，默认 1e-10")
    parser.add_argument("--n-plot", type=int, default=3000, help="每条轨迹绘图采样点数，默认 3000")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        w_list=args.w_list,
        t_end=args.t_end,
        capture_tol=args.capture_tol,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        n_plot=max(200, args.n_plot),
        show_plot=not args.no_plot,
    )

