"""
习题 6.8：有限时长广告下的销售量模型

建模思路：
  1) 无广告时，销售量 s(t) 的下降速度与 s(t) 成正比；
  2) 广告带来正向增量，强度与广告费 a(t) 成正比；
  3) 广告只作用于未饱和市场（饱和量 M）。

运行示例：
  python ch06/hw06_08/solution.py
  python ch06/hw06_08/solution.py --M 80000 --s0 4200 --k 0.5 --gamma 0.03 --ad-cost 3.5 --tau 10
  python ch06/hw06_08/solution.py --t-end 40 --num-points 1500 --no-plot
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


def build_ad_phase_solution(M, N0, s0, k, gamma, ad_cost):
    """
    广告期 0<=t<=tau 的解析解。

    模型：
      N'(t) = s(t)
      s'(t) = -k s(t) + gamma * ad_cost * (M - N(t))
    """
    if gamma * ad_cost < 0:
        raise ValueError("gamma*ad_cost 不能为负。")

    z0 = N0 - M
    delta = k**2 - 4.0 * gamma * ad_cost
    eps = 1e-12

    if delta > eps:
        regime = "过阻尼（指数双根）"
        sqrt_delta = np.sqrt(delta)
        r1 = (-k + sqrt_delta) / 2.0
        r2 = (-k - sqrt_delta) / 2.0
        c1 = (s0 - r2 * z0) / (r1 - r2)
        c2 = (r1 * z0 - s0) / (r1 - r2)

        def eval_func(t):
            t = np.asarray(t, dtype=float)
            e1 = np.exp(r1 * t)
            e2 = np.exp(r2 * t)
            z = c1 * e1 + c2 * e2
            s = c1 * r1 * e1 + c2 * r2 * e2
            n = M + z
            return n, s

        meta = {"delta": float(delta), "r1": float(r1), "r2": float(r2)}

    elif delta < -eps:
        regime = "欠阻尼（阻尼振荡）"
        omega = np.sqrt(4.0 * gamma * ad_cost - k**2) / 2.0
        a0 = z0
        b0 = (s0 + 0.5 * k * a0) / omega

        def eval_func(t):
            t = np.asarray(t, dtype=float)
            decay = np.exp(-0.5 * k * t)
            cos_term = np.cos(omega * t)
            sin_term = np.sin(omega * t)
            z = decay * (a0 * cos_term + b0 * sin_term)
            s = decay * (
                -0.5 * k * (a0 * cos_term + b0 * sin_term)
                - a0 * omega * sin_term
                + b0 * omega * cos_term
            )
            n = M + z
            return n, s

        meta = {"delta": float(delta), "omega": float(omega)}

    else:
        regime = "临界阻尼（重根）"
        r = -0.5 * k
        a0 = z0
        b0 = s0 - r * a0

        def eval_func(t):
            t = np.asarray(t, dtype=float)
            er = np.exp(r * t)
            z = (a0 + b0 * t) * er
            s = (b0 + r * (a0 + b0 * t)) * er
            n = M + z
            return n, s

        meta = {"delta": float(delta), "r": float(r)}

    return regime, meta, eval_func


def evaluate_piecewise_trajectory(t_grid, M, N0, s0, k, gamma, ad_cost, tau):
    """
    先用广告期解析解，再拼接停广告后的解析解：
      t > tau: s'=-k s, N'=s
    """
    t_grid = np.asarray(t_grid, dtype=float)
    if np.any(t_grid < 0):
        raise ValueError("t_grid 不能包含负时间。")

    regime, meta, ad_eval = build_ad_phase_solution(
        M=M,
        N0=N0,
        s0=s0,
        k=k,
        gamma=gamma,
        ad_cost=ad_cost,
    )

    n_traj = np.empty_like(t_grid)
    s_traj = np.empty_like(t_grid)
    ad_traj = np.where(t_grid <= tau, ad_cost, 0.0)

    mask_ad = t_grid <= tau
    if np.any(mask_ad):
        n_ad, s_ad = ad_eval(t_grid[mask_ad])
        n_traj[mask_ad] = n_ad
        s_traj[mask_ad] = s_ad

    n_tau, s_tau = ad_eval(np.array([tau], dtype=float))
    n_tau = float(n_tau[0])
    s_tau = float(s_tau[0])

    mask_post = ~mask_ad
    if np.any(mask_post):
        dt = t_grid[mask_post] - tau
        if abs(k) > 1e-12:
            e = np.exp(-k * dt)
            s_post = s_tau * e
            n_post = n_tau + (s_tau / k) * (1.0 - e)
        else:
            s_post = np.full_like(dt, s_tau)
            n_post = n_tau + s_tau * dt
        n_traj[mask_post] = n_post
        s_traj[mask_post] = s_post

    return {
        "regime": regime,
        "meta": meta,
        "t": t_grid,
        "N": n_traj,
        "s": s_traj,
        "a": ad_traj,
        "N_tau": n_tau,
        "s_tau": s_tau,
    }


def print_report(result, M, N0, s0, k, gamma, ad_cost, tau, t_end):
    """打印核心结果。"""
    n_end = float(result["N"][-1])
    s_end = float(result["s"][-1])
    n_tau = result["N_tau"]
    s_tau = result["s_tau"]

    if k > 0:
        n_inf = n_tau + s_tau / k
    else:
        n_inf = np.nan

    print("=== 习题 6.8 求解结果（有限时长广告） ===")
    print("模型：")
    print("  N'(t)=s(t)")
    print("  s'(t)=-k*s(t)+gamma*a(t)*(M-N(t))")
    print(f"  a(t) = {ad_cost} (0<=t<=tau), 0 (t>tau), tau={tau}")
    print("\n参数：")
    print(f"  M={M}, N0={N0}, s0={s0}")
    print(f"  k={k}, gamma={gamma}, ad_cost={ad_cost}, tau={tau}")
    print(f"  广告期解型: {result['regime']}, 判别式 delta={result['meta']['delta']:.8f}")

    print("\n阶段结果：")
    print(f"  t=tau 时: N(tau)={n_tau:.6f}, s(tau)={s_tau:.6f}")
    print(f"  t={t_end} 时: N(t_end)={n_end:.6f}, s(t_end)={s_end:.6f}")
    if k > 0:
        print(f"  停广告后极限销量 s(∞)=0, 极限拥有量 N(∞)={n_inf:.6f}")


def plot_result(result, M, tau):
    """绘制 a(t)、s(t)、N(t) 曲线。"""
    t = result["t"]
    s = result["s"]
    n = result["N"]
    ad = result["a"]

    fig, axes = plt.subplots(3, 1, figsize=(9.8, 8.6), sharex=True)

    axes[0].plot(t, ad, color="#9467bd", linewidth=2.0)
    axes[0].set_ylabel("a(t)")
    axes[0].set_title("习题 6.8：有限时长广告下的销售量变化")
    axes[0].grid(alpha=0.3)
    axes[0].axvline(tau, color="#444444", linestyle="--", linewidth=1.2)

    axes[1].plot(t, s, color="#1f77b4", linewidth=2.1, label="s(t)")
    axes[1].axvline(tau, color="#444444", linestyle="--", linewidth=1.2, label="广告停止时刻")
    axes[1].set_ylabel("销售量 s(t)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(t, n, color="#d62728", linewidth=2.1, label="N(t) 累计拥有量")
    axes[2].axhline(M, color="#2ca02c", linestyle=":", linewidth=1.6, label="饱和量 M")
    axes[2].axvline(tau, color="#444444", linestyle="--", linewidth=1.2)
    axes[2].set_xlabel("时间 t")
    axes[2].set_ylabel("拥有量 N(t)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def solve(
    M=100000.0,
    N0=0.0,
    s0=5000.0,
    k=0.6,
    gamma=0.02,
    ad_cost=4.0,
    tau=12.0,
    t_end=36.0,
    num_points=1200,
    show_plot=True,
):
    """主流程：按解析分段公式计算并绘图。"""
    if M <= 0:
        raise ValueError("M 必须为正。")
    if N0 < 0 or N0 > M:
        raise ValueError("N0 应满足 0<=N0<=M。")
    if s0 < 0:
        raise ValueError("s0 不能为负。")
    if k < 0:
        raise ValueError("k 不能为负。")
    if gamma < 0 or ad_cost < 0:
        raise ValueError("gamma 与 ad_cost 不能为负。")
    if tau < 0:
        raise ValueError("tau 不能为负。")
    if t_end <= 0 or t_end <= tau:
        raise ValueError("t_end 需大于 tau 且为正。")
    if num_points < 200:
        raise ValueError("num_points 建议不小于 200。")

    t_grid = np.linspace(0.0, t_end, num_points)
    result = evaluate_piecewise_trajectory(
        t_grid=t_grid,
        M=M,
        N0=N0,
        s0=s0,
        k=k,
        gamma=gamma,
        ad_cost=ad_cost,
        tau=tau,
    )
    print_report(
        result=result,
        M=M,
        N0=N0,
        s0=s0,
        k=k,
        gamma=gamma,
        ad_cost=ad_cost,
        tau=tau,
        t_end=t_end,
    )

    if show_plot:
        plot_result(result=result, M=M, tau=tau)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.8：有限时长广告下的销售量模型")
    parser.add_argument("--M", type=float, default=100000.0, help="市场饱和量 M，默认 100000")
    parser.add_argument("--N0", type=float, default=0.0, help="初始拥有量 N(0)，默认 0")
    parser.add_argument("--s0", type=float, default=5000.0, help="初始销售量 s(0)，默认 5000")
    parser.add_argument("--k", type=float, default=0.6, help="自然衰减系数 k，默认 0.6")
    parser.add_argument("--gamma", type=float, default=0.02, help="广告效率系数 gamma，默认 0.02")
    parser.add_argument("--ad-cost", type=float, default=4.0, help="广告费常数 a，默认 4.0")
    parser.add_argument("--tau", type=float, default=12.0, help="广告持续时间 tau，默认 12")
    parser.add_argument("--t-end", type=float, default=36.0, help="仿真终点时刻，默认 36")
    parser.add_argument("--num-points", type=int, default=1200, help="绘图采样点数，默认 1200")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(
        M=args.M,
        N0=args.N0,
        s0=args.s0,
        k=args.k,
        gamma=args.gamma,
        ad_cost=args.ad_cost,
        tau=args.tau,
        t_end=args.t_end,
        num_points=args.num_points,
        show_plot=not args.no_plot,
    )
