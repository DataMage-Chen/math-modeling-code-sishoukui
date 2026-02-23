"""
习题 5.3：线性插值与三次样条插值

运行示例：
  python ch05/hw05_03/solution.py
  python ch05/hw05_03/solution.py --queries 750 770 --bc-type natural
  python ch05/hw05_03/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline, interp1d
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


T_DATA = np.array([700, 720, 740, 760, 780], dtype=float)
V_DATA = np.array([0.0977, 0.1218, 0.1406, 0.1551, 0.1664], dtype=float)


def build_interpolators(bc_type):
    """构造线性插值与三次样条插值函数。"""
    linear_interp = interp1d(T_DATA, V_DATA, kind="linear")
    cubic_interp = CubicSpline(T_DATA, V_DATA, bc_type=bc_type)
    return linear_interp, cubic_interp


def print_report(queries, v_lin, v_cubic):
    """打印查询点结果。"""
    print("=== 习题 5.3 求解结果 ===")
    print("已知数据：")
    for t, v in zip(T_DATA, V_DATA):
        print(f"  T={t:.0f}, V={v:.4f}")

    print("\n查询温度点估计：")
    for t, vl, vc in zip(queries, v_lin, v_cubic):
        diff = abs(float(vl) - float(vc))
        print(
            f"  T={float(t):.1f}: "
            f"线性插值 V={float(vl):.8f}, "
            f"三次样条 V={float(vc):.8f}, "
            f"|差值|={diff:.8e}"
        )


def plot_result(linear_interp, cubic_interp, grid_points=600):
    """绘制原始数据、线性插值曲线和三次样条曲线。"""
    t_dense = np.linspace(float(T_DATA.min()), float(T_DATA.max()), grid_points)
    v_lin_dense = linear_interp(t_dense)
    v_cubic_dense = cubic_interp(t_dense)

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    ax.scatter(T_DATA, V_DATA, color="#1f77b4", s=48, zorder=4, label="原始数据点")
    ax.plot(t_dense, v_lin_dense, color="#ff7f0e", linewidth=2.0, label="线性插值函数")
    ax.plot(t_dense, v_cubic_dense, color="#2ca02c", linewidth=2.0, linestyle="--", label="三次样条插值函数")

    ax.set_title("习题 5.3：线性插值与三次样条插值对比")
    ax.set_xlabel("温度 T")
    ax.set_ylabel("体积 V")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(queries=None, bc_type="not-a-knot", show_plot=True):
    if queries is None:
        queries = [750.0, 770.0]

    queries = np.array(queries, dtype=float)
    t_min = float(T_DATA.min())
    t_max = float(T_DATA.max())
    if np.any(queries < t_min) or np.any(queries > t_max):
        raise ValueError(f"查询温度必须在 [{t_min}, {t_max}] 范围内。")

    linear_interp, cubic_interp = build_interpolators(bc_type=bc_type)
    v_lin = linear_interp(queries)
    v_cubic = cubic_interp(queries)

    print_report(queries, v_lin, v_cubic)
    if show_plot:
        plot_result(linear_interp, cubic_interp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.3 线性插值与三次样条插值")
    parser.add_argument(
        "--queries",
        nargs="+",
        type=float,
        default=[750.0, 770.0],
        help="查询温度点，默认 750 770",
    )
    parser.add_argument(
        "--bc-type",
        type=str,
        choices=["not-a-knot", "natural"],
        default="not-a-knot",
        help="三次样条边界条件，默认 not-a-knot",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        queries=args.queries,
        bc_type=args.bc_type,
        show_plot=not args.no_plot,
    )

