"""
习题 6.5：半球形容器底部小孔放水模型

题意：高为 1 m 的半球形容器，底部有面积为 1 cm^2 的小孔，
初始盛满水，求水位高度 h(t)（水面到孔口中心的距离）随时间变化规律。

运行示例：
  python ch06/hw06_05/solution.py
  python ch06/hw06_05/solution.py --radius 1 --hole-area-cm2 1 --cd 1.0
  python ch06/hw06_05/solution.py --g 9.81 --num-points 1200 --no-plot
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


def cross_section_area(h, radius):
    """半球在高度 h 处的横截面积 A(h) = pi(2Rh-h^2)。"""
    h = np.asarray(h, dtype=float)
    return np.pi * (2.0 * radius * h - h**2)


def t_of_h_implicit(h, radius, hole_area_m2, g, cd):
    """
    基于体积守恒 + 托里拆利定律得到的隐式关系 t(h)。

    dV/dt = -Cd * a * sqrt(2gh)
    dV/dh = A(h) = pi(2Rh-h^2)
    =>
    dt/dh = -A(h) / (Cd*a*sqrt(2gh))
    """
    h = np.asarray(h, dtype=float)
    k = np.pi / (cd * hole_area_m2 * np.sqrt(2.0 * g))
    term1 = (4.0 * radius / 3.0) * (radius ** 1.5 - h ** 1.5)
    term2 = (2.0 / 5.0) * (radius ** 2.5 - h ** 2.5)
    return k * (term1 - term2)


def build_ode(radius, hole_area_m2, g, cd):
    """构造 h'(t) 右端函数。"""

    def rhs(_t, y):
        h_val = float(y[0])
        if h_val <= 0.0:
            return np.array([0.0], dtype=float)

        area = float(cross_section_area(h_val, radius))
        outflow = cd * hole_area_m2 * np.sqrt(2.0 * g * h_val)
        dh_dt = -outflow / area
        return np.array([dh_dt], dtype=float)

    return rhs


def empty_event(_t, y):
    """水位触底事件：h=0。"""
    return y[0]


empty_event.terminal = True
empty_event.direction = -1


def solve_drain(radius, hole_area_cm2, g, cd, num_points, method, rtol, atol):
    """求解放水过程 ODE。"""
    if radius <= 0:
        raise ValueError("radius 必须为正。")
    if hole_area_cm2 <= 0:
        raise ValueError("hole_area_cm2 必须为正。")
    if g <= 0:
        raise ValueError("g 必须为正。")
    if cd <= 0:
        raise ValueError("cd 必须为正。")
    if num_points < 200:
        raise ValueError("num_points 建议不小于 200。")

    hole_area_m2 = hole_area_cm2 * 1e-4
    h0 = radius  # 半球高=半径，题中“盛满水”对应 h(0)=R

    # 用隐式公式先估算排空时间，给数值积分设置上限
    t_empty_theory = float(
        t_of_h_implicit(
            h=np.array([0.0]),
            radius=radius,
            hole_area_m2=hole_area_m2,
            g=g,
            cd=cd,
        )[0]
    )
    t_span_end = t_empty_theory * 1.2
    t_eval = np.linspace(0.0, t_span_end, num_points)

    rhs = build_ode(radius=radius, hole_area_m2=hole_area_m2, g=g, cd=cd)
    sol = solve_ivp(
        rhs,
        (0.0, t_span_end),
        y0=np.array([h0], dtype=float),
        method=method,
        t_eval=t_eval,
        events=empty_event,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"求解失败：{sol.message}")

    if sol.t_events and len(sol.t_events[0]) > 0:
        t_empty_num = float(sol.t_events[0][0])
    else:
        t_empty_num = float(sol.t[-1])

    return {
        "sol": sol,
        "hole_area_m2": hole_area_m2,
        "t_empty_theory": t_empty_theory,
        "t_empty_num": t_empty_num,
    }


def print_report(result, radius, hole_area_cm2, g, cd, method, rtol, atol):
    """打印模型与结果。"""
    sol = result["sol"]
    h_series = sol.y[0]
    t_series = sol.t
    t_empty_theory = result["t_empty_theory"]
    t_empty_num = result["t_empty_num"]

    print("=== 习题 6.5 求解结果（半球容器放水） ===")
    print(f"几何参数: 半径 R = {radius:.6f} m（半球高度同为 {radius:.6f} m）")
    print(f"小孔面积: a = {hole_area_cm2:.6f} cm^2 = {result['hole_area_m2']:.10f} m^2")
    print(f"重力加速度: g = {g:.6f} m/s^2")
    print(f"流量系数: Cd = {cd:.6f}")
    print(f"数值方法: {method}, rtol={rtol:.1e}, atol={atol:.1e}")
    print(f"积分统计: nfev={sol.nfev}, njev={sol.njev}, nlu={sol.nlu}")

    print("\n变化规律（隐式表达）：")
    print(
        "  t(h) = pi/(Cd*a*sqrt(2g)) * [ (4R/3)*(R^(3/2)-h^(3/2)) - (2/5)*(R^(5/2)-h^(5/2)) ]"
    )
    print(f"\n理论排空时间（h=0）: T_theory = {t_empty_theory:.6f} s")
    print(f"数值事件时间（h=0）: T_num    = {t_empty_num:.6f} s")
    print(f"两者差值: T_num - T_theory = {t_empty_num - t_empty_theory:.6e} s")

    print("\n关键时刻样本：")
    sample_times = np.linspace(0.0, min(t_empty_num, t_series[-1]), 6)
    h_interp = np.interp(sample_times, t_series, h_series)
    for t_val, h_val in zip(sample_times, h_interp):
        print(f"  t={t_val:10.4f} s -> h={h_val:10.6f} m")


def plot_result(result, radius):
    """绘制 h(t) 曲线。"""
    sol = result["sol"]
    t_series = sol.t
    h_series = sol.y[0]
    t_empty_num = result["t_empty_num"]

    plt.figure(figsize=(9.2, 5.6))
    plt.plot(t_series, h_series, color="#1f77b4", linewidth=2.2, label="数值解 h(t)")
    plt.axvline(t_empty_num, color="#d62728", linestyle="--", linewidth=1.5, label="排空时刻")
    plt.title("习题 6.5：半球容器放水过程中的水位变化")
    plt.xlabel("时间 t (s)")
    plt.ylabel("水位高度 h (m)")
    plt.ylim(bottom=-0.02 * radius, top=1.02 * radius)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(
    radius=1.0,
    hole_area_cm2=1.0,
    g=9.81,
    cd=1.0,
    num_points=1000,
    method="RK45",
    rtol=1e-8,
    atol=1e-10,
    show_plot=True,
):
    """主流程：建模、求解、输出、绘图。"""
    result = solve_drain(
        radius=radius,
        hole_area_cm2=hole_area_cm2,
        g=g,
        cd=cd,
        num_points=num_points,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    print_report(
        result=result,
        radius=radius,
        hole_area_cm2=hole_area_cm2,
        g=g,
        cd=cd,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if show_plot:
        plot_result(result, radius=radius)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.5：半球容器底孔放水模型")
    parser.add_argument("--radius", type=float, default=1.0, help="半球半径（m），默认 1")
    parser.add_argument("--hole-area-cm2", type=float, default=1.0, help="小孔面积（cm^2），默认 1")
    parser.add_argument("--g", type=float, default=9.81, help="重力加速度（m/s^2），默认 9.81")
    parser.add_argument("--cd", type=float, default=1.0, help="流量系数 Cd，默认 1.0")
    parser.add_argument("--num-points", type=int, default=1000, help="输出网格点数，默认 1000")
    parser.add_argument(
        "--method",
        type=str,
        default="RK45",
        choices=["RK23", "RK45", "DOP853", "Radau", "BDF", "LSODA"],
        help="solve_ivp 积分方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-8, help="相对误差容限")
    parser.add_argument("--atol", type=float, default=1e-10, help="绝对误差容限")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(
        radius=args.radius,
        hole_area_cm2=args.hole_area_cm2,
        g=args.g,
        cd=args.cd,
        num_points=args.num_points,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )
