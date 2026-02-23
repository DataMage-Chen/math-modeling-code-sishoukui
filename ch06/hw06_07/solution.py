"""
习题 6.7：按原书推导重做过滤嘴吸烟模型

模型来源：教材中的 x-t 连续介质建模（q(x,t), w(x,t)）。
核心结果：
  Q = a * M * exp(-beta*l2/v) * phi(r)
  r = a' * b * l1 / v,  a' = 1-a,  phi(r) = (1-exp(-r))/r

运行示例：
  python ch06/hw06_07/solution.py
  python ch06/hw06_07/solution.py --M 12 --l1 60 --l2 30 --a 0.65 --b 0.35 --beta 1.2
  python ch06/hw06_07/solution.py --u 0.5 --v 40 --no-plot
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


def phi_func(r):
    """phi(r)=(1-exp(-r))/r，带 r->0 的稳定处理。"""
    r = np.asarray(r, dtype=float)
    out = np.empty_like(r)
    small = np.abs(r) < 1e-9
    out[small] = 1.0 - 0.5 * r[small]
    out[~small] = -np.expm1(-r[~small]) / r[~small]
    return out


def q_x0(x, a, h0, b, beta, l1, v):
    """
    式(6.11)：t=0 时毒物流率 q(x,0)=q(x)。
    x, l1 用 mm；v 用 mm/s；b,beta 用 1/s。
    """
    x = np.asarray(x, dtype=float)
    q_val = np.empty_like(x)
    in_tobacco = x <= l1
    q_val[in_tobacco] = a * h0 * np.exp(-b * x[in_tobacco] / v)
    q_val[~in_tobacco] = (
        a * h0 * np.exp(-b * l1 / v) * np.exp(-beta * (x[~in_tobacco] - l1) / v)
    )
    return q_val


def w_ut(t, w0, a, a_prime, b, u, v):
    """
    式(6.16) 在 x=ut 处的简化结果：
      w(ut,t) = w0/a' * (1 - a*exp(-a' b u t / v))
    """
    t = np.asarray(t, dtype=float)
    return (w0 / a_prime) * (1.0 - a * np.exp(-a_prime * b * u * t / v))


def q_l_t(t, a, w0, a_prime, b, beta, l1, l2, u, v):
    """
    按式(6.14)和式(6.16)构造 q(l,t)：
      q(l,t)=a*u*w(ut,t)*exp(-b(l1-ut)/v)*exp(-beta*l2/v)
    """
    wt = w_ut(t, w0=w0, a=a, a_prime=a_prime, b=b, u=u, v=v)
    return a * u * wt * np.exp(-b * (l1 - u * t) / v) * np.exp(-beta * l2 / v)


def q_l_t_closed_form(t, a, w0, a_prime, b, beta, l1, l2, u, v):
    """
    式(6.17) 的闭式 q(l,t)（用于和上式交叉校验）。
    """
    coef = (a * u * w0 / a_prime) * np.exp(-b * l1 / v) * np.exp(-beta * l2 / v)
    return coef * (np.exp(b * u * t / v) - a * np.exp(a * b * u * t / v))


def q_total_closed_form(M, a, b, beta, l1, l2, v):
    """
    式(6.19)/(6.21)：
      Q = a*M*exp(-beta*l2/v)*phi(a' b l1/v)
    """
    a_prime = 1.0 - a
    r = a_prime * b * l1 / v
    return a * M * np.exp(-beta * l2 / v) * float(phi_func(np.array([r]))[0])


def q_total_by_trapz(M, a, b, beta, l1, l2, u, v, n_grid=4000):
    """
    由 Q = ∫_0^{l1/u} q(l,t)dt 数值积分，和闭式结果做核对。
    """
    w0 = M / l1
    a_prime = 1.0 - a
    t_end = l1 / u
    t_grid = np.linspace(0.0, t_end, n_grid)
    q_grid = q_l_t(
        t_grid, a=a, w0=w0, a_prime=a_prime, b=b, beta=beta, l1=l1, l2=l2, u=u, v=v
    )
    q_grid_cf = q_l_t_closed_form(
        t_grid, a=a, w0=w0, a_prime=a_prime, b=b, beta=beta, l1=l1, l2=l2, u=u, v=v
    )
    q_err = float(np.max(np.abs(q_grid - q_grid_cf)))
    q_total_num = float(np.trapezoid(q_grid, t_grid))
    return q_total_num, t_grid, q_grid, q_err


def print_report(M, l1, l2, a, b, beta, u, v, q_closed, q_num, q_profile_match_err):
    """打印本题关键输出。"""
    a_prime = 1.0 - a
    r = a_prime * b * l1 / v
    phi = float(phi_func(np.array([r]))[0])
    q_no_filter = a * M * phi  # beta=0
    q_filter_like_tobacco = a * M * np.exp(-b * l2 / v) * phi  # beta=b

    ratio_1 = q_closed / q_no_filter if q_no_filter > 0 else np.nan
    ratio_2 = q_closed / q_filter_like_tobacco if q_filter_like_tobacco > 0 else np.nan
    ratio_theory = np.exp(-(beta - b) * l2 / v)

    print("=== 习题 6.7 求解结果（原书模型） ===")
    print("参数（单位采用 mm, s, mg）：")
    print(f"  烟草毒物总量 M = {M:.6f} mg")
    print(f"  烟草长度 l1 = {l1:.6f} mm, 过滤嘴长度 l2 = {l2:.6f} mm")
    print(f"  穿行比例 a = {a:.6f}, a' = {a_prime:.6f}")
    print(f"  烟草吸收率 b = {b:.6f} 1/s, 过滤嘴吸收率 beta = {beta:.6f} 1/s")
    print(f"  烟雾速度 v = {v:.6f} mm/s, 燃烧速度 u = {u:.6f} mm/s")
    print(f"  速度比 v/u = {v/u:.3f}（应远大于1）")

    print("\n中间量：")
    print(f"  r = a' * b * l1 / v = {r:.8f}")
    print(f"  phi(r) = (1-exp(-r))/r = {phi:.8f}")

    print("\n总吸入毒量 Q：")
    print(f"  闭式公式 Q_closed = {q_closed:.10f} mg")
    print(f"  数值积分 Q_num    = {q_num:.10f} mg")
    print(f"  差值 Q_num-Q_closed = {q_num - q_closed:.6e} mg")

    print("\n剖面一致性校验：")
    print(f"  max|q_l_t(式14+16) - q_l_t(式17)| = {q_profile_match_err:.6e}")

    print("\n对比量：")
    print(f"  无过滤嘴衰减(beta=0)时 Q0 = {q_no_filter:.10f} mg")
    print(f"  过滤嘴吸收率与烟草相同(beta=b)时 Q2 = {q_filter_like_tobacco:.10f} mg")
    print(f"  Q/Q0 = {ratio_1:.8f}")
    print(f"  Q/Q2 = {ratio_2:.8f}, 理论 e^[-(beta-b)l2/v] = {ratio_theory:.8f}")


def plot_result(M, l1, l2, a, b, beta, u, v, t_grid, q_grid):
    """绘图：q(l,t) 曲线 + Q 对 l2 的敏感性。"""
    a_prime = 1.0 - a

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2))

    ax1.plot(t_grid, q_grid, color="#1f77b4", linewidth=2.1)
    ax1.set_title("q(l,t) 随时间变化（一个吸烟周期）")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("q(l,t) (mg/s)")
    ax1.grid(alpha=0.3)

    l2_grid = np.linspace(0.0, max(50.0, l2 * 2.0), 200)
    q_current = a * M * np.exp(-beta * l2_grid / v) * phi_func(a_prime * b * l1 / v)
    q_beta_eq_b = a * M * np.exp(-b * l2_grid / v) * phi_func(a_prime * b * l1 / v)
    q_no_decay = a * M * phi_func(a_prime * b * l1 / v)

    ax2.plot(l2_grid, q_current, color="#d62728", linewidth=2.1, label="当前 beta")
    ax2.plot(l2_grid, q_beta_eq_b, color="#2ca02c", linewidth=1.9, linestyle="--", label="beta=b")
    ax2.axhline(q_no_decay, color="#444444", linewidth=1.4, linestyle=":", label="beta=0")
    ax2.axvline(l2, color="#9467bd", linewidth=1.2, linestyle="-.", label="当前 l2")
    ax2.set_title("总吸入毒量 Q 对过滤嘴长度 l2 的敏感性")
    ax2.set_xlabel("l2 (mm)")
    ax2.set_ylabel("Q (mg)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    M=10.0,
    l1=60.0,
    l2=30.0,
    a=0.65,
    b=0.35,
    beta=1.20,
    u=0.5,
    v=40.0,
    show_plot=True,
):
    """主流程：按原书公式计算并核对。"""
    if M <= 0:
        raise ValueError("M 必须为正。")
    if l1 <= 0 or l2 < 0:
        raise ValueError("l1 必须为正，l2 不能为负。")
    if not (0 < a < 1):
        raise ValueError("a 必须在 (0,1) 内。")
    if b <= 0 or beta < 0:
        raise ValueError("b 必须为正，beta 不能为负。")
    if u <= 0 or v <= 0:
        raise ValueError("u 和 v 必须为正。")
    if v <= 10 * u:
        print("警告：当前参数下 v 并未显著大于 u，可能偏离原书近似前提。")

    q_closed = q_total_closed_form(M=M, a=a, b=b, beta=beta, l1=l1, l2=l2, v=v)
    q_num, t_grid, q_grid, q_profile_match_err = q_total_by_trapz(
        M=M, a=a, b=b, beta=beta, l1=l1, l2=l2, u=u, v=v
    )

    print_report(
        M=M,
        l1=l1,
        l2=l2,
        a=a,
        b=b,
        beta=beta,
        u=u,
        v=v,
        q_closed=q_closed,
        q_num=q_num,
        q_profile_match_err=q_profile_match_err,
    )

    if show_plot:
        plot_result(M=M, l1=l1, l2=l2, a=a, b=b, beta=beta, u=u, v=v, t_grid=t_grid, q_grid=q_grid)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.7：按原书推导重做过滤嘴模型")
    parser.add_argument("--M", type=float, default=10.0, help="烟草总毒物量 M（mg）")
    parser.add_argument("--l1", type=float, default=60.0, help="烟草长度 l1（mm）")
    parser.add_argument("--l2", type=float, default=30.0, help="过滤嘴长度 l2（mm）")
    parser.add_argument("--a", type=float, default=0.65, help="毒物随烟雾穿行比例 a（0~1）")
    parser.add_argument("--b", type=float, default=0.35, help="烟草吸收率 b（1/s）")
    parser.add_argument("--beta", type=float, default=1.20, help="过滤嘴吸收率 beta（1/s）")
    parser.add_argument("--u", type=float, default=0.5, help="燃烧速度 u（mm/s）")
    parser.add_argument("--v", type=float, default=40.0, help="烟雾穿行速度 v（mm/s）")
    parser.add_argument("--no-plot", action="store_true", help="只输出数值，不绘图")
    args = parser.parse_args()

    solve(
        M=args.M,
        l1=args.l1,
        l2=args.l2,
        a=args.a,
        b=args.b,
        beta=args.beta,
        u=args.u,
        v=args.v,
        show_plot=not args.no_plot,
    )
