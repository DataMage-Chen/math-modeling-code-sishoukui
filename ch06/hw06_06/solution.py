"""
习题 6.6：黄灯时长与“进退两难区”建模

问题：司机接近路口时，黄灯过短会出现“既难以安全停车，又难以及时通过”的区间。
本代码基于简化动力学模型给出“最小合理黄灯时长”。

运行示例：
  python ch06/hw06_06/solution.py
  python ch06/hw06_06/solution.py --v0-kmh 60 --reaction-time 1.2 --decel 4.5
  python ch06/hw06_06/solution.py --accel 0 --intersection-width 18 --vehicle-length 4.8
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


def kmh_to_ms(v_kmh):
    """速度 km/h 转 m/s。"""
    return v_kmh / 3.6


def stop_distance(v0, reaction_time, decel):
    """
    安全停车所需距离（到停止线）：
      d_stop = v0 * t_r + v0^2 / (2b)
    """
    return v0 * reaction_time + (v0**2) / (2.0 * decel)


def pass_distance(yellow_time, v0, accel, clear_distance):
    """
    黄灯期间可清空路口的最大初始距离（到停止线）：
      d_pass = v0*T + 0.5*a*T^2 - D_clear
    其中 D_clear = 路口宽度 + 车长。
    """
    return v0 * yellow_time + 0.5 * accel * yellow_time**2 - clear_distance


def min_yellow_time(v0, reaction_time, decel, accel, clear_distance):
    """
    消除两难区的最小黄灯时长：令 d_pass(T)=d_stop。
    求解：
      0.5*a*T^2 + v0*T - (d_stop + D_clear) = 0
    """
    d_stop = stop_distance(v0, reaction_time, decel)
    rhs = d_stop + clear_distance

    if abs(accel) < 1e-12:
        return rhs / v0

    a_q = 0.5 * accel
    b_q = v0
    c_q = -rhs
    delta = b_q**2 - 4.0 * a_q * c_q
    if delta < 0:
        raise ValueError("给定参数下二次方程无实根，请检查加速度与参数设置。")

    root1 = (-b_q + math.sqrt(delta)) / (2.0 * a_q)
    root2 = (-b_q - math.sqrt(delta)) / (2.0 * a_q)
    positive_roots = [r for r in (root1, root2) if r > 0]
    if not positive_roots:
        raise ValueError("未得到正黄灯时长根，请检查参数设置。")
    return min(positive_roots)


def zone_length(yellow_time, v0, reaction_time, decel, accel, clear_distance):
    """两难区长度 = max(0, d_stop - d_pass(T))。"""
    d_stop = stop_distance(v0, reaction_time, decel)
    d_pass = pass_distance(yellow_time, v0, accel, clear_distance)
    return max(0.0, d_stop - d_pass), d_stop, d_pass


def print_report(
    v0_kmh,
    reaction_time,
    decel,
    accel,
    intersection_width,
    vehicle_length,
    policy_min_yellow,
    t_star,
    t_recommend,
):
    """输出关键结果。"""
    v0 = kmh_to_ms(v0_kmh)
    clear_distance = intersection_width + vehicle_length
    z_len_star, d_stop, d_pass_star = zone_length(
        yellow_time=t_star,
        v0=v0,
        reaction_time=reaction_time,
        decel=decel,
        accel=accel,
        clear_distance=clear_distance,
    )
    z_len_rec, _, d_pass_rec = zone_length(
        yellow_time=t_recommend,
        v0=v0,
        reaction_time=reaction_time,
        decel=decel,
        accel=accel,
        clear_distance=clear_distance,
    )

    print("=== 习题 6.6 求解结果（黄灯时长） ===")
    print(f"来车速度: v0 = {v0_kmh:.2f} km/h = {v0:.6f} m/s")
    print(f"驾驶员反应时间: t_r = {reaction_time:.3f} s")
    print(f"可接受减速度: b = {decel:.3f} m/s^2")
    print(f"黄灯期可用加速度: a = {accel:.3f} m/s^2")
    print(f"路口净通行距离: D_clear = 路口宽度+车长 = {clear_distance:.3f} m")

    print("\n判据距离：")
    print(f"  安全停车阈值 d_stop = {d_stop:.6f} m")
    print(f"  在 T*= {t_star:.6f} s 下可通过阈值 d_pass(T*) = {d_pass_star:.6f} m")
    print(f"  两难区长度 L_zone(T*) = {z_len_star:.6e} m（理论上应接近 0）")

    print("\n黄灯时长建议：")
    print(f"  纯模型最小黄灯时长 T* = {t_star:.6f} s")
    print(f"  规则下限（可调）T_min_policy = {policy_min_yellow:.6f} s")
    print(f"  推荐黄灯时长 T_rec = max(T*, T_min_policy) = {t_recommend:.6f} s")
    print(f"  对应两难区长度 L_zone(T_rec) = {z_len_rec:.6f} m")
    print(f"  对应可通过阈值 d_pass(T_rec) = {d_pass_rec:.6f} m")


def plot_result(v0, reaction_time, decel, accel, clear_distance, t_star, t_recommend):
    """绘制 d_stop 与 d_pass(T) 关系，直观看两难区是否消除。"""
    t_grid = np.linspace(0.5, max(8.0, t_recommend * 1.6), 500)
    d_stop = stop_distance(v0, reaction_time, decel)
    d_pass_grid = pass_distance(t_grid, v0, accel, clear_distance)
    zone_grid = np.maximum(0.0, d_stop - d_pass_grid)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.2))

    ax1.plot(t_grid, d_pass_grid, linewidth=2.0, color="#1f77b4", label="可通过阈值 d_pass(T)")
    ax1.axhline(d_stop, color="#d62728", linestyle="--", linewidth=1.8, label="停车阈值 d_stop")
    ax1.axvline(t_star, color="#2ca02c", linestyle=":", linewidth=1.8, label="T*（消除两难区）")
    ax1.axvline(
        t_recommend,
        color="#9467bd",
        linestyle="-.",
        linewidth=1.8,
        label="T_rec（推荐）",
    )
    ax1.set_title("阈值距离与黄灯时长关系")
    ax1.set_xlabel("黄灯时长 T (s)")
    ax1.set_ylabel("距离阈值 (m)")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(t_grid, zone_grid, linewidth=2.0, color="#ff7f0e")
    ax2.axvline(t_star, color="#2ca02c", linestyle=":", linewidth=1.8, label="T*")
    ax2.axvline(t_recommend, color="#9467bd", linestyle="-.", linewidth=1.8, label="T_rec")
    ax2.set_title("两难区长度 L_zone(T)")
    ax2.set_xlabel("黄灯时长 T (s)")
    ax2.set_ylabel("L_zone (m)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(
    v0_kmh=50.0,
    reaction_time=1.0,
    decel=4.0,
    accel=2.0,
    intersection_width=20.0,
    vehicle_length=5.0,
    policy_min_yellow=3.0,
    show_plot=True,
):
    """主流程：计算最小黄灯时长并展示结果。"""
    if decel <= 0:
        raise ValueError("decel 必须为正。")
    if reaction_time < 0:
        raise ValueError("reaction_time 不能为负。")
    if intersection_width <= 0 or vehicle_length <= 0:
        raise ValueError("intersection_width 与 vehicle_length 必须为正。")
    if policy_min_yellow < 0:
        raise ValueError("policy_min_yellow 不能为负。")

    v0 = kmh_to_ms(v0_kmh)
    clear_distance = intersection_width + vehicle_length

    t_star = min_yellow_time(
        v0=v0,
        reaction_time=reaction_time,
        decel=decel,
        accel=accel,
        clear_distance=clear_distance,
    )
    t_recommend = max(t_star, policy_min_yellow)

    print_report(
        v0_kmh=v0_kmh,
        reaction_time=reaction_time,
        decel=decel,
        accel=accel,
        intersection_width=intersection_width,
        vehicle_length=vehicle_length,
        policy_min_yellow=policy_min_yellow,
        t_star=t_star,
        t_recommend=t_recommend,
    )

    if show_plot:
        plot_result(
            v0=v0,
            reaction_time=reaction_time,
            decel=decel,
            accel=accel,
            clear_distance=clear_distance,
            t_star=t_star,
            t_recommend=t_recommend,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.6：黄灯时长与两难区分析")
    parser.add_argument("--v0-kmh", type=float, default=50.0, help="来车速度（km/h），默认 50")
    parser.add_argument("--reaction-time", type=float, default=1.0, help="反应时间 t_r（s），默认 1.0")
    parser.add_argument("--decel", type=float, default=4.0, help="可接受减速度 b（m/s^2），默认 4.0")
    parser.add_argument("--accel", type=float, default=2.0, help="黄灯期可用加速度 a（m/s^2），默认 2.0")
    parser.add_argument(
        "--intersection-width",
        type=float,
        default=20.0,
        help="路口宽度（m），默认 20",
    )
    parser.add_argument("--vehicle-length", type=float, default=5.0, help="车长（m），默认 5")
    parser.add_argument(
        "--policy-min-yellow",
        type=float,
        default=3.0,
        help="工程规则下限黄灯时长（s），默认 3.0",
    )
    parser.add_argument("--no-plot", action="store_true", help="只输出结果，不绘图")
    args = parser.parse_args()

    solve(
        v0_kmh=args.v0_kmh,
        reaction_time=args.reaction_time,
        decel=args.decel,
        accel=args.accel,
        intersection_width=args.intersection_width,
        vehicle_length=args.vehicle_length,
        policy_min_yellow=args.policy_min_yellow,
        show_plot=not args.no_plot,
    )
