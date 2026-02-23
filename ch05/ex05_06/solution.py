"""
例题 5.6：根据边界测量数据估计国土面积与周长。

运行：
  python ch05/ex05_06/solution.py
  python ch05/ex05_06/solution.py --bc not-a-knot --show-plot
  python ch05/ex05_06/solution.py --bc natural
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import quad
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


# 表 5.5 数据（单位：mm）
X_DATA = np.array(
    [
        7.0, 10.5, 13.0, 17.5, 34.0, 40.5, 44.5, 48.0, 56.0,
        61.0, 68.5, 76.5, 80.5, 91.0, 96.0, 101.0, 104.0, 106.5,
        111.5, 118.0, 123.5, 136.5, 142.0, 146.0, 150.0, 157.0, 158.0,
    ],
    dtype=float,
)
Y1_DATA = np.array(
    [
        44, 45, 47, 50, 50, 38, 30, 30, 34,
        36, 34, 41, 45, 46, 43, 37, 33, 28,
        32, 65, 55, 54, 52, 50, 66, 66, 68,
    ],
    dtype=float,
)
Y2_DATA = np.array(
    [
        44, 59, 70, 72, 93, 100, 110, 110, 110,
        117, 118, 116, 118, 118, 121, 124, 121, 121,
        121, 122, 116, 83, 81, 82, 86, 85, 68,
    ],
    dtype=float,
)

SCALE_LEN = 40.0 / 18.0  # km / mm
SCALE_AREA = SCALE_LEN * SCALE_LEN  # km^2 / mm^2
TRUE_AREA = 41288.0  # km^2


def trapezoid_integral(y, x):
    """兼容新旧 NumPy 的梯形积分。"""
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(y, x))


def linear_baseline(x, y1, y2):
    """折线（分段线性）基准结果。"""
    area_map = trapezoid_integral(y2 - y1, x)

    dx = np.diff(x)
    l_bottom = float(np.sum(np.hypot(dx, np.diff(y1))))
    l_top = float(np.sum(np.hypot(dx, np.diff(y2))))
    l_left = float(abs(y2[0] - y1[0]))
    l_right = float(abs(y2[-1] - y1[-1]))
    perim_map = l_bottom + l_top + l_left + l_right

    return area_map, perim_map


def spline_method(x, y1, y2, bc_type):
    """三次样条方法。"""
    s1 = CubicSpline(x, y1, bc_type=bc_type)
    s2 = CubicSpline(x, y2, bc_type=bc_type)
    x0, x1 = float(x[0]), float(x[-1])

    area_map = float(s2.integrate(x0, x1) - s1.integrate(x0, x1))

    def integrand_bottom(t):
        d = float(s1(t, 1))
        return float(np.sqrt(1.0 + d * d))

    def integrand_top(t):
        d = float(s2(t, 1))
        return float(np.sqrt(1.0 + d * d))

    # 在样条分段节点上拆分积分，减少 quad 在全区间上的舍入误差警告
    l_bottom = 0.0
    l_top = 0.0
    for i in range(len(x) - 1):
        a, b = float(x[i]), float(x[i + 1])
        l_bottom += float(quad(integrand_bottom, a, b, limit=100)[0])
        l_top += float(quad(integrand_top, a, b, limit=100)[0])

    l_left = float(abs(y2[0] - y1[0]))
    l_right = float(abs(y2[-1] - y1[-1]))
    perim_map = l_bottom + l_top + l_left + l_right

    return area_map, perim_map, s1, s2


def print_report(name, area_map, perim_map):
    """打印指定方法的面积与周长结果。"""
    area_km2 = area_map * SCALE_AREA
    perim_km = perim_map * SCALE_LEN
    abs_err = area_km2 - TRUE_AREA
    rel_err = abs(abs_err) / TRUE_AREA * 100.0

    print(f"\n--- {name} ---")
    print(f"地图面积估计: {area_map:.10f} mm^2")
    print(f"实际面积估计: {area_km2:.10f} km^2")
    print(f"与精确面积差: {abs_err:.10f} km^2")
    print(f"面积相对误差: {rel_err:.6f}%")
    print(f"地图周长估计: {perim_map:.10f} mm")
    print(f"实际周长估计: {perim_km:.10f} km")


def plot_boundaries(x, y1, y2, s1, s2):
    """绘制南北边界曲线及面积填充。"""
    x_dense = np.linspace(float(x[0]), float(x[-1]), 2000)
    y1_dense = np.asarray(s1(x_dense), dtype=float)
    y2_dense = np.asarray(s2(x_dense), dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.plot(x_dense, y1_dense, color="#1f78b4", linewidth=2.0, label="南边界样条 y1(x)")
    ax.plot(x_dense, y2_dense, color="#e31a1c", linewidth=2.0, label="北边界样条 y2(x)")
    ax.scatter(x, y1, color="#1f78b4", s=20, alpha=0.9)
    ax.scatter(x, y2, color="#e31a1c", s=20, alpha=0.9)
    ax.fill_between(x_dense, y1_dense, y2_dense, color="#a6cee3", alpha=0.35, label="国土截面区域")

    ax.set_title("例5.6 边界样条与面积估计区域（地图坐标，单位 mm）")
    ax.set_xlabel("x (mm)")
    ax.set_ylabel("y (mm)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(bc_type="not-a-knot", show_plot=False):
    if not np.all(np.diff(X_DATA) > 0):
        raise ValueError("x 数据必须严格递增。")
    if np.any(Y2_DATA < Y1_DATA):
        raise ValueError("检测到 y2<y1，数据不满足北边界在上方的假设。")

    print("=== 例题 5.6 求解结果 ===")
    print(f"数据点数: {len(X_DATA)}")
    print(f"比例尺: 18 mm = 40 km, 长度换算系数 = {SCALE_LEN:.10f} km/mm")
    print(f"面积精确值（用于比较）: {TRUE_AREA:.2f} km^2")

    area_lin, perim_lin = linear_baseline(X_DATA, Y1_DATA, Y2_DATA)
    print_report("分段线性基准", area_lin, perim_lin)

    area_sp, perim_sp, s1, s2 = spline_method(X_DATA, Y1_DATA, Y2_DATA, bc_type=bc_type)
    print(f"\n三次样条边界条件: {bc_type}")
    print_report("三次样条方法", area_sp, perim_sp)

    if show_plot:
        plot_boundaries(X_DATA, Y1_DATA, Y2_DATA, s1, s2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.6：边界插值求面积与周长")
    parser.add_argument(
        "--bc",
        choices=["not-a-knot", "natural"],
        default="not-a-knot",
        help="三次样条边界条件，默认 not-a-knot",
    )
    parser.add_argument(
        "--show-plot",
        action="store_true",
        help="显示边界样条图与面积填充",
    )
    args = parser.parse_args()
    solve(bc_type=args.bc, show_plot=args.show_plot)
