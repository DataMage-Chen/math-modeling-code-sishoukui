"""
例题 6.11：Lorenz 模型混沌效应复现。

运行示例：
  python ch06/ex06_11/solution.py
  python ch06/ex06_11/solution.py --seed 2 --delta 1e-4
  python ch06/ex06_11/solution.py --x0 0.2 0.5 0.7 --method DOP853 --no-plot
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


def lorenz_rhs(_, state, sigma, rho, beta):
    """Lorenz 方程右端。"""
    x, y, z = state
    dx = sigma * (y - x)
    dy = rho * x - y - x * z
    dz = x * y - beta * z
    return [dx, dy, dz]


def simulate_lorenz(x0, t_end=80.0, n_points=6000, sigma=10.0, rho=28.0, beta=8.0 / 3.0, method="RK45", rtol=1e-8, atol=1e-10):
    """在统一时间网格上积分 Lorenz 系统。"""
    t_eval = np.linspace(0.0, t_end, n_points)
    sol = solve_ivp(
        fun=lambda t, s: lorenz_rhs(t, s, sigma=sigma, rho=rho, beta=beta),
        t_span=(0.0, t_end),
        y0=np.asarray(x0, dtype=float),
        method=method,
        t_eval=t_eval,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"Lorenz 数值积分失败：{sol.message}")
    return sol.t, sol.y


def print_report(t, xyz1, xyz2, sigma, rho, beta, x0, delta, method):
    """打印数值结果摘要。"""
    dx = xyz1[0] - xyz2[0]
    dy = xyz1[1] - xyz2[1]
    dz = xyz1[2] - xyz2[2]
    d_norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    print("=== 例题 6.11 求解结果（Lorenz 混沌效应） ===")
    print(f"参数: sigma={sigma}, rho={rho}, beta={beta}")
    print(f"积分方法: {method}")
    print(f"时间区间: [0, {t[-1]}], 网格点数: {t.size}")
    print(f"初值 X0 = [{x0[0]:.8f}, {x0[1]:.8f}, {x0[2]:.8f}]")
    print(f"扰动初值 X0* = X0 + [{delta}, {delta}, {delta}]")

    print("\n偏差统计：")
    print(f"  max|x-x*| = {np.max(np.abs(dx)):.12e}")
    print(f"  max|y-y*| = {np.max(np.abs(dy)):.12e}")
    print(f"  max|z-z*| = {np.max(np.abs(dz)):.12e}")
    print(f"  max||Δ||2 = {np.max(d_norm):.12e}")

    print("\n末时刻偏差：")
    print(f"  Δx(T) = {dx[-1]:.12e}")
    print(f"  Δy(T) = {dy[-1]:.12e}")
    print(f"  Δz(T) = {dz[-1]:.12e}")
    print(f"  ||Δ(T)||2 = {d_norm[-1]:.12e}")

    # 早期指数放大粗估（只用于现象说明）
    eps = 1e-16
    early_mask = (t >= 1.0) & (t <= min(20.0, t[-1])) & (d_norm > eps)
    if np.any(early_mask):
        tt = t[early_mask]
        ln_d = np.log(d_norm[early_mask])
        slope, intercept = np.polyfit(tt, ln_d, 1)
        _ = intercept
        print(f"\n早期 ln||Δ|| 线性拟合斜率（粗估 Lyapunov 指标）: {slope:.6f}")


def plot_result(t, xyz1, xyz2):
    """绘制三维相轨线与偏差曲线。"""
    dx = xyz1[0] - xyz2[0]
    dy = xyz1[1] - xyz2[1]
    dz = xyz1[2] - xyz2[2]
    d_norm = np.sqrt(dx ** 2 + dy ** 2 + dz ** 2)

    fig = plt.figure(figsize=(12.6, 5.2))

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot(xyz1[0], xyz1[1], xyz1[2], color="#1f77b4", linewidth=0.9, label="轨线 X(t)")
    ax1.set_title("Lorenz 三维轨线（蝴蝶吸引子）")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.set_zlabel("z")
    ax1.legend(loc="upper left")

    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(t, dx, color="#d62728", linestyle="-.", linewidth=1.6, label="x(t)-x*(t)")
    ax2.plot(t, d_norm, color="#2ca02c", linewidth=1.2, alpha=0.85, label="||Δ(t)||2")
    ax2.set_title("两条近初值轨线偏差演化")
    ax2.set_xlabel("t")
    ax2.set_ylabel("偏差")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    sigma=10.0,
    rho=28.0,
    beta=8.0 / 3.0,
    t_end=80.0,
    n_points=6000,
    seed=2,
    delta=1e-4,
    x0=None,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
    show_plot=True,
):
    if x0 is None:
        rng = np.random.default_rng(seed)
        x0 = rng.random(3)  # 复现 MATLAB 代码里 rand(3,1) 的思路
    else:
        x0 = np.asarray(x0, dtype=float)
        if x0.size != 3:
            raise ValueError("x0 必须是 3 维向量。")

    x0_perturb = x0 + delta

    t, xyz1 = simulate_lorenz(
        x0=x0,
        t_end=t_end,
        n_points=n_points,
        sigma=sigma,
        rho=rho,
        beta=beta,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    _, xyz2 = simulate_lorenz(
        x0=x0_perturb,
        t_end=t_end,
        n_points=n_points,
        sigma=sigma,
        rho=rho,
        beta=beta,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    print_report(t, xyz1, xyz2, sigma, rho, beta, x0, delta, method)
    if show_plot:
        plot_result(t, xyz1, xyz2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.11 Lorenz 混沌效应复现")
    parser.add_argument("--sigma", type=float, default=10.0, help="参数 sigma，默认 10")
    parser.add_argument("--rho", type=float, default=28.0, help="参数 rho，默认 28")
    parser.add_argument("--beta", type=float, default=8.0 / 3.0, help="参数 beta，默认 8/3")
    parser.add_argument("--t-end", type=float, default=80.0, help="积分终止时间 T，默认 80")
    parser.add_argument("--n-points", type=int, default=6000, help="时间采样点数，默认 6000")
    parser.add_argument("--seed", type=int, default=2, help="随机初值种子，默认 2")
    parser.add_argument("--delta", type=float, default=1e-4, help="初值扰动量，默认 1e-4")
    parser.add_argument("--x0", nargs=3, type=float, default=None, help="可选：手工给定初值 x0 y0 z0")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="相对容差，默认 1e-8")
    parser.add_argument("--atol", type=float, default=1e-10, help="绝对容差，默认 1e-10")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        sigma=args.sigma,
        rho=args.rho,
        beta=args.beta,
        t_end=args.t_end,
        n_points=max(200, args.n_points),
        seed=args.seed,
        delta=args.delta,
        x0=args.x0,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )

