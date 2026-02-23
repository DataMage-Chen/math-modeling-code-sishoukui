"""
例题 5.5：用三次样条插值估计位移 S = ∫ v(t) dt。

运行：
  python ch05/ex05_05/solution.py
  python ch05/ex05_05/solution.py --bc not-a-knot
  python ch05/ex05_05/solution.py --bc natural --show-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline
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


T_DATA = np.array([0.15, 0.16, 0.17, 0.18], dtype=float)
V_DATA = np.array([3.5, 1.5, 2.5, 2.8], dtype=float)


def solve(bc_type="not-a-knot", show_plot=False):
    spline = CubicSpline(T_DATA, V_DATA, bc_type=bc_type)

    # 插值校验
    fit_err = np.max(np.abs(spline(T_DATA) - V_DATA))

    # 样条积分位移
    t0, t1 = float(T_DATA[0]), float(T_DATA[-1])
    s_spline = float(spline.integrate(t0, t1))

    # 作为参考：梯形法积分（兼容新旧 NumPy）
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    s_trapz = float(trapz_fn(V_DATA, T_DATA))

    print("=== 例题 5.5 求解结果 ===")
    print(f"样条边界条件: {bc_type}")
    print(f"节点插值校验 max|s(t_i)-v_i| = {fit_err:.3e}")
    print(f"样条积分位移 S = ∫[{t0:.2f},{t1:.2f}] v(t)dt = {s_spline:.10f}")
    print(f"梯形法参考值 S_trapz = {s_trapz:.10f}")
    print(f"两者差值 S - S_trapz = {s_spline - s_trapz:.10f}")

    print("\n分段三次多项式系数（每段按 a*(t-ti)^3+b*(t-ti)^2+c*(t-ti)+d）：")
    # CubicSpline.c 形状为 (4, n-1)，对应降幂系数
    for i in range(len(T_DATA) - 1):
        ti = T_DATA[i]
        a, b, c, d = spline.c[:, i]
        print(
            f"  段[{T_DATA[i]:.2f},{T_DATA[i+1]:.2f}] (ti={ti:.2f}): "
            f"a={a:.10f}, b={b:.10f}, c={c:.10f}, d={d:.10f}"
        )

    if show_plot:
        t_dense = np.linspace(t0, t1, 500)
        v_dense = spline(t_dense)

        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.plot(t_dense, v_dense, color="#1f78b4", linewidth=2.0, label="三次样条 v(t)")
        ax.scatter(T_DATA, V_DATA, color="black", zorder=3, label="观测点")
        ax.fill_between(t_dense, 0, v_dense, color="#a6cee3", alpha=0.35, label="位移面积")
        ax.set_title("例5.5 速度样条曲线与积分面积")
        ax.set_xlabel("t")
        ax.set_ylabel("v(t)")
        ax.grid(alpha=0.3)
        ax.legend()
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.5 三次样条积分求位移")
    parser.add_argument(
        "--bc",
        choices=["natural", "not-a-knot"],
        default="not-a-knot",
        help="三次样条边界条件，默认 not-a-knot",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="显示样条曲线与积分面积图",
    )
    args = parser.parse_args()
    solve(bc_type=args.bc, show_plot=args.show_plot)
