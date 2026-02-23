"""
例题 6.14：阿波罗卫星轨道方程数值求解（按教材公式复现）。

运行示例：
  python ch06/ex06_14/solution.py
  python ch06/ex06_14/solution.py --t-end 100 --method RK45
  python ch06/ex06_14/solution.py --rtol 1e-7 --atol 1e-9 --no-plot
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


MU = 1.0 / 82.45
LAMBDA = 1.0 - MU


def rhs(_, state, mu, lam):
    """
    一阶系统右端，state=[x,vx,y,vy]。
    按教材给定式子实现（与常见 CR3BP 写法略有差异）。
    """
    x, vx, y, vy = state

    r1 = np.sqrt((x + mu) ** 2 + y ** 2)
    r2 = np.sqrt((x + lam) ** 2 + y ** 2)

    dx = vx
    dvx = 2.0 * vy + x - lam * (x + mu) / (r1 ** 3) - mu * (x - lam) / (r2 ** 3)
    dy = vy
    dvy = -2.0 * vx + y - lam * y / (r1 ** 3) - mu * y / (r2 ** 3)
    return [dx, dvx, dy, dvy]


def solve_orbit(t_end=100.0, method="RK45", rtol=1e-6, atol=1e-9, n_eval=12000):
    """数值积分并返回轨道结果。"""
    x0, vx0, y0, vy0 = 1.2, 0.0, 0.0, -1.0494
    t_eval = np.linspace(0.0, t_end, n_eval) if n_eval is not None else None

    sol = solve_ivp(
        fun=lambda t, s: rhs(t, s, mu=MU, lam=LAMBDA),
        t_span=(0.0, t_end),
        y0=[x0, vx0, y0, vy0],
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"积分失败：{sol.message}")

    x, vx, y, vy = sol.y
    r1 = np.sqrt((x + MU) ** 2 + y ** 2)
    r2 = np.sqrt((x + LAMBDA) ** 2 + y ** 2)

    return {
        "t": sol.t,
        "x": x,
        "vx": vx,
        "y": y,
        "vy": vy,
        "r1": r1,
        "r2": r2,
        "nfev": int(sol.nfev),
        "njev": int(sol.njev),
        "nlu": int(sol.nlu),
        "x0": x0,
        "vx0": vx0,
        "y0": y0,
        "vy0": vy0,
    }


def print_report(result, method, t_end, rtol, atol):
    """打印求解结果摘要。"""
    x, y = result["x"], result["y"]
    vx, vy = result["vx"], result["vy"]

    print("=== 例题 6.14 求解结果（按教材公式） ===")
    print(f"参数: mu={MU:.12f}, lambda={LAMBDA:.12f}")
    print(f"初值: x(0)={result['x0']}, x'(0)={result['vx0']}, y(0)={result['y0']}, y'(0)={result['vy0']}")
    print(f"时间区间: [0, {t_end}]")
    print(f"方法与容差: method={method}, rtol={rtol}, atol={atol}")
    print(f"求解器统计: nfev={result['nfev']}, njev={result['njev']}, nlu={result['nlu']}")

    print("\n末时刻状态：")
    print(f"  x(T)  = {x[-1]:.12f}")
    print(f"  y(T)  = {y[-1]:.12f}")
    print(f"  x'(T) = {vx[-1]:.12f}")
    print(f"  y'(T) = {vy[-1]:.12f}")

    print("\n轨道统计：")
    print(f"  x 范围: [{np.min(x):.8f}, {np.max(x):.8f}]")
    print(f"  y 范围: [{np.min(y):.8f}, {np.max(y):.8f}]")
    print(f"  min r1 = {np.min(result['r1']):.10f}")
    print(f"  min r2 = {np.min(result['r2']):.10f}")


def plot_orbit(result):
    """绘制轨迹图（贴近教材风格）。"""
    x, y = result["x"], result["y"]

    fig, ax = plt.subplots(figsize=(8.4, 6.6))
    ax.plot(x, y, color="black", linewidth=1.3, label="轨迹")
    ax.set_xlabel("x")
    ax.set_ylabel("y", rotation=0)
    ax.set_title("阿波罗卫星轨迹图")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.show()


def solve(t_end=100.0, method="RK45", rtol=1e-6, atol=1e-9, n_eval=12000, show_plot=True):
    result = solve_orbit(
        t_end=t_end,
        method=method,
        rtol=rtol,
        atol=atol,
        n_eval=n_eval,
    )
    print_report(result, method=method, t_end=t_end, rtol=rtol, atol=atol)
    if show_plot:
        plot_orbit(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.14 阿波罗卫星轨道数值求解（教材复现版）")
    parser.add_argument("--t-end", type=float, default=100.0, help="积分终止时间，默认 100（教材风格）")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45（对应 MATLAB ode45 风格）",
    )
    parser.add_argument("--rtol", type=float, default=1e-6, help="相对容差，默认 1e-6")
    parser.add_argument("--atol", type=float, default=1e-9, help="绝对容差，默认 1e-9")
    parser.add_argument(
        "--n-eval",
        type=int,
        default=12000,
        help="输出采样点数，默认 12000；若设为 0 则使用求解器自适应节点",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        t_end=args.t_end,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        n_eval=(None if args.n_eval <= 0 else max(200, args.n_eval)),
        show_plot=not args.no_plot,
    )

