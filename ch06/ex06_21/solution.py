"""
例题 6.21：圆桶下沉速度安全性分析（线性阻力 + 二次阻力）。

运行示例：
  python ch06/ex06_21/solution.py
  python ch06/ex06_21/solution.py --depth 90 --v-safe 12.2 --no-plot
  python ch06/ex06_21/solution.py --g 9.81 --k 0.6
"""

import argparse
import math

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


def net_force(m, rho, v_obj, g):
    """有效下拉常量 F0 = (m-rho*V)*g。"""
    return (m - rho * v_obj) * g


def linear_model_params(m, k, f0):
    """线性阻力模型参数。"""
    beta = k / m
    v_inf = f0 / k
    return beta, v_inf


def linear_velocity(t, beta, v_inf):
    """线性阻力速度。"""
    t = np.asarray(t, dtype=float)
    return v_inf * (1.0 - np.exp(-beta * t))


def linear_displacement(t, beta, v_inf):
    """线性阻力位移。"""
    t = np.asarray(t, dtype=float)
    return v_inf * (t - (1.0 - np.exp(-beta * t)) / beta)


def solve_time_for_depth_linear(depth, beta, v_inf):
    """求线性模型下 s(t)=depth 的时间（单调方程二分）。"""
    if depth < 0:
        raise ValueError("depth 必须非负。")
    if depth == 0:
        return 0.0

    def f(t):
        return float(linear_displacement(t, beta=beta, v_inf=v_inf) - depth)

    lo, hi = 0.0, 1.0
    while f(hi) < 0:
        hi *= 2.0
        if hi > 1e7:
            raise RuntimeError("线性模型深度方程求解区间扩展失败。")

    for _ in range(120):
        mid = 0.5 * (lo + hi)
        if f(mid) >= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def quadratic_model_params(m, k, f0):
    """二次阻力模型参数。"""
    v_t = math.sqrt(f0 / k)
    alpha = math.sqrt(f0 * k) / m
    return alpha, v_t


def quadratic_velocity(t, alpha, v_t):
    """二次阻力速度 v=v_t*tanh(alpha*t)。"""
    t = np.asarray(t, dtype=float)
    return v_t * np.tanh(alpha * t)


def quadratic_displacement(t, m, k, alpha):
    """二次阻力位移 s=(m/k)ln(cosh(alpha*t))。"""
    t = np.asarray(t, dtype=float)
    return (m / k) * np.log(np.cosh(alpha * t))


def quadratic_time_for_depth(depth, m, k, alpha):
    """二次阻力模型下 s(t)=depth 的显式反解。"""
    if depth < 0:
        raise ValueError("depth 必须非负。")
    if depth == 0:
        return 0.0
    return float(np.arccosh(np.exp(k * depth / m)) / alpha)


def report_results(
    m,
    v_obj,
    rho,
    k,
    g,
    depth,
    v_safe,
    beta,
    v_inf,
    t_depth_lin,
    v_depth_lin,
    alpha,
    v_t,
    t_depth_quad,
    v_depth_quad,
    t_safe_quad,
    s_safe_quad,
):
    """打印结果报告。"""
    f0 = net_force(m, rho, v_obj, g)
    print("=== 例题 6.21 求解结果 ===")
    print(f"参数: m={m}, V={v_obj}, rho={rho}, k={k}, g={g}")
    print(f"安全阈值速度: v_safe={v_safe} m/s, 海底深度: {depth} m")
    print(f"有效下拉常量 F0=(m-rho*V)g = {f0:.12f} N")

    print("\n(1) 线性阻力模型 R=kv：")
    print(f"  速度表达式: v(t)=v_inf*(1-exp(-beta*t))")
    print(f"  beta={beta:.12f}, v_inf={v_inf:.12f} m/s")
    print(f"  到达 {depth}m 的时间 t_depth={t_depth_lin:.10f} s")
    print(f"  海底撞击速度 v(depth)={v_depth_lin:.10f} m/s")
    print(f"  与阈值比较: v(depth) {'<=' if v_depth_lin <= v_safe else '>'} {v_safe}")
    if v_depth_lin <= v_safe:
        print("  结论：按线性阻力模型，该处理方法可认为安全。")
    else:
        print("  结论：按线性阻力模型，撞击速度超阈值，存在风险。")

    print("\n(2) 高速二次阻力模型 R=kv^2（由 k->k*v）：")
    print("  速度解析式: v(t)=v_t*tanh(alpha*t)")
    print("  位移解析式: s(t)=(m/k)*ln(cosh(alpha*t))")
    print(f"  alpha={alpha:.12f}, v_t={v_t:.12f} m/s")
    print(f"  到达 {depth}m 的时间 t_depth={t_depth_quad:.10f} s")
    print(f"  该时刻速度 v(depth)={v_depth_quad:.10f} m/s")

    if np.isfinite(t_safe_quad):
        print(f"  为保证 v<= {v_safe} m/s：")
        print(f"    t 不应超过 t_safe={t_safe_quad:.10f} s")
        print(f"    s 不应超过 s_safe={s_safe_quad:.10f} m")
    else:
        print("  v_safe >= v_t，速度上限在该模型下天然满足（无需额外 t,s 限制）。")


def plot_curves(
    depth,
    v_safe,
    t_max,
    beta,
    v_inf,
    m,
    k,
    alpha,
    v_t,
):
    """绘制线性/二次模型的 v-t 与 s-t 曲线。"""
    t = np.linspace(0.0, t_max, 1200)

    v_lin = linear_velocity(t, beta=beta, v_inf=v_inf)
    s_lin = linear_displacement(t, beta=beta, v_inf=v_inf)

    v_quad = quadratic_velocity(t, alpha=alpha, v_t=v_t)
    s_quad = quadratic_displacement(t, m=m, k=k, alpha=alpha)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.8, 5.2))

    ax1.plot(t, v_lin, color="#1f77b4", linewidth=1.8, label="线性阻力 v(t)")
    ax1.plot(t, v_quad, color="#d62728", linewidth=1.8, linestyle="--", label="二次阻力 v(t)")
    ax1.axhline(v_safe, color="black", linestyle=":", linewidth=1.2, label="安全阈值")
    ax1.set_title("速度-时间关系")
    ax1.set_xlabel("t (s)")
    ax1.set_ylabel("v (m/s)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(t, s_lin, color="#1f77b4", linewidth=1.8, label="线性阻力 s(t)")
    ax2.plot(t, s_quad, color="#d62728", linewidth=1.8, linestyle="--", label="二次阻力 s(t)")
    ax2.axhline(depth, color="black", linestyle=":", linewidth=1.2, label="海底深度")
    ax2.set_title("位移-时间关系")
    ax2.set_xlabel("t (s)")
    ax2.set_ylabel("s (m)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    m=239.46,
    v_obj=0.2058,
    rho=1035.71,
    k=0.6,
    g=9.8,
    depth=90.0,
    v_safe=12.2,
    show_plot=True,
):
    f0 = net_force(m=m, rho=rho, v_obj=v_obj, g=g)
    if f0 <= 0:
        raise RuntimeError("F0<=0，表示物体不会下沉（或上浮），请检查参数。")
    if k <= 0:
        raise RuntimeError("k 必须为正。")

    # (1) 线性阻力
    beta, v_inf = linear_model_params(m=m, k=k, f0=f0)
    t_depth_lin = solve_time_for_depth_linear(depth=depth, beta=beta, v_inf=v_inf)
    v_depth_lin = float(linear_velocity(t_depth_lin, beta=beta, v_inf=v_inf))

    # (2) 二次阻力（高速近似）
    alpha, v_t = quadratic_model_params(m=m, k=k, f0=f0)
    t_depth_quad = quadratic_time_for_depth(depth=depth, m=m, k=k, alpha=alpha)
    v_depth_quad = float(quadratic_velocity(t_depth_quad, alpha=alpha, v_t=v_t))

    if v_safe < v_t:
        t_safe_quad = float(np.arctanh(v_safe / v_t) / alpha)
        s_safe_quad = float(quadratic_displacement(t_safe_quad, m=m, k=k, alpha=alpha))
    else:
        t_safe_quad = np.inf
        s_safe_quad = np.inf

    report_results(
        m=m,
        v_obj=v_obj,
        rho=rho,
        k=k,
        g=g,
        depth=depth,
        v_safe=v_safe,
        beta=beta,
        v_inf=v_inf,
        t_depth_lin=t_depth_lin,
        v_depth_lin=v_depth_lin,
        alpha=alpha,
        v_t=v_t,
        t_depth_quad=t_depth_quad,
        v_depth_quad=v_depth_quad,
        t_safe_quad=t_safe_quad,
        s_safe_quad=s_safe_quad,
    )

    if show_plot:
        t_plot_max = max(t_depth_lin, t_depth_quad, (t_safe_quad if np.isfinite(t_safe_quad) else 0.0)) * 1.2
        t_plot_max = max(t_plot_max, 5.0)
        plot_curves(
            depth=depth,
            v_safe=v_safe,
            t_max=t_plot_max,
            beta=beta,
            v_inf=v_inf,
            m=m,
            k=k,
            alpha=alpha,
            v_t=v_t,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.21 圆桶下沉速度安全性分析")
    parser.add_argument("--m", type=float, default=239.46, help="圆桶质量，默认 239.46 kg")
    parser.add_argument("--V", type=float, default=0.2058, help="圆桶体积，默认 0.2058 m^3")
    parser.add_argument("--rho", type=float, default=1035.71, help="海水密度，默认 1035.71 kg/m^3")
    parser.add_argument("--k", type=float, default=0.6, help="阻力参数，默认 0.6")
    parser.add_argument("--g", type=float, default=9.8, help="重力加速度，默认 9.8")
    parser.add_argument("--depth", type=float, default=90.0, help="海底深度，默认 90 m")
    parser.add_argument("--v-safe", type=float, default=12.2, help="安全阈值速度，默认 12.2 m/s")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        m=args.m,
        v_obj=args.V,
        rho=args.rho,
        k=args.k,
        g=args.g,
        depth=args.depth,
        v_safe=args.v_safe,
        show_plot=not args.no_plot,
    )

