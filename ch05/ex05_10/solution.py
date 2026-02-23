"""
例题 5.10：由 5 个观测点求小行星椭圆轨道方程。

运行：
  python ch05/ex05_10/solution.py
  python ch05/ex05_10/solution.py --no-plot
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


X_DATA = np.array([5.764, 6.286, 6.759, 7.168, 7.408], dtype=float)
Y_DATA = np.array([0.648, 1.202, 1.823, 2.526, 3.360], dtype=float)


def fit_orbit_coefficients(x, y):
    """
    求解参数 a1..a5，使：
      a1*x^2 + a2*x*y + a3*y^2 + a4*x + a5*y + 1 = 0
    """
    m = np.column_stack([x * x, x * y, y * y, x, y])
    rhs = -np.ones_like(x)

    # n=5 时理论上可精确解；若后续扩展更多点，也能最小二乘求解
    if m.shape[0] == 5 and np.linalg.matrix_rank(m) == 5:
        coeff = np.linalg.solve(m, rhs)
        mode = "exact_solve"
    else:
        coeff, *_ = np.linalg.lstsq(m, rhs, rcond=None)
        mode = "least_squares"
    return coeff, mode


def conic_value(x, y, coeff):
    """计算隐式函数 F(x,y)。"""
    a1, a2, a3, a4, a5 = coeff
    return a1 * x * x + a2 * x * y + a3 * y * y + a4 * x + a5 * y + 1.0


def conic_center(coeff):
    """计算二次曲线中心（若可解）。"""
    a1, a2, a3, a4, a5 = coeff
    mat = np.array([[2 * a1, a2], [a2, 2 * a3]], dtype=float)
    vec = -np.array([a4, a5], dtype=float)
    if abs(np.linalg.det(mat)) < 1e-14:
        return None
    c = np.linalg.solve(mat, vec)
    return float(c[0]), float(c[1])


def print_report(coeff, mode):
    """打印参数与校验信息。"""
    a1, a2, a3, a4, a5 = coeff
    residual = conic_value(X_DATA, Y_DATA, coeff)
    max_res = float(np.max(np.abs(residual)))
    delta = float(a2 * a2 - 4 * a1 * a3)
    center = conic_center(coeff)

    print("=== 例题 5.10 求解结果 ===")
    print(f"求解方式: {mode}")
    print("拟合轨道方程：")
    print(
        "  "
        f"{a1:.12f}*x^2 + {a2:.12f}*x*y + {a3:.12f}*y^2 + "
        f"{a4:.12f}*x + {a5:.12f}*y + 1 = 0"
    )
    print("参数：")
    print(f"  a1 = {a1:.12f}")
    print(f"  a2 = {a2:.12f}")
    print(f"  a3 = {a3:.12f}")
    print(f"  a4 = {a4:.12f}")
    print(f"  a5 = {a5:.12f}")

    print("观测点代入残差 F(x_i,y_i)：")
    for i, r in enumerate(residual, start=1):
        print(f"  点{i}: {r:.3e}")
    print(f"最大残差 max|F| = {max_res:.3e}")

    print(f"判别量 Δ = a2^2 - 4*a1*a3 = {delta:.12f}")
    print("曲线类型判别：")
    print("  Δ < 0 => 椭圆型" if delta < 0 else "  Δ >= 0 => 非椭圆型（需复核）")

    if center is not None:
        print(f"轨道中心（由二次项求得）约为: ({center[0]:.6f}, {center[1]:.6f})")


def plot_orbit(coeff):
    """绘制隐式曲线 F(x,y)=0 与观测点。"""
    # 先给一个较宽的计算窗口，避免轨道在绘图区边界被截断
    x_span = float(np.max(X_DATA) - np.min(X_DATA))
    y_span = float(np.max(Y_DATA) - np.min(Y_DATA))
    x_margin = max(0.8, 1.6 * x_span)
    y_margin = max(0.8, 1.6 * y_span)
    x_min, x_max = float(np.min(X_DATA) - x_margin), float(np.max(X_DATA) + x_margin)
    y_min, y_max = float(np.min(Y_DATA) - y_margin), float(np.max(Y_DATA) + y_margin)

    xx = np.linspace(x_min, x_max, 600)
    yy = np.linspace(y_min, y_max, 600)
    xg, yg = np.meshgrid(xx, yy)
    fg = conic_value(xg, yg, coeff)

    fig, ax = plt.subplots(figsize=(8.5, 6))
    contour = ax.contour(xg, yg, fg, levels=[0.0], colors=["#d62728"], linewidths=2.0)
    ax.clabel(contour, inline=True, fontsize=8, fmt={0.0: "轨道 F=0"})
    ax.scatter(X_DATA, Y_DATA, c="#1f77b4", s=45, zorder=3, label="观测点")

    # 根据实际等值线顶点自适应坐标边界，避免图像被裁剪
    try:
        paths = contour.collections[0].get_paths()
        verts = np.vstack([p.vertices for p in paths if p.vertices.size > 0])
        vx_min, vy_min = np.min(verts, axis=0)
        vx_max, vy_max = np.max(verts, axis=0)
        pad = 0.08 * max(vx_max - vx_min, vy_max - vy_min)
        ax.set_xlim(vx_min - pad, vx_max + pad)
        ax.set_ylim(vy_min - pad, vy_max + pad)
    except Exception:
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    center = conic_center(coeff)
    if center is not None:
        ax.scatter([center[0]], [center[1]], c="black", marker="x", s=70, label="轨道中心")

    ax.set_title("例5.10 小行星轨道拟合曲线")
    ax.set_xlabel("x (天文单位)")
    ax.set_ylabel("y (天文单位)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(show_plot=True):
    coeff, mode = fit_orbit_coefficients(X_DATA, Y_DATA)
    print_report(coeff, mode)
    if show_plot:
        plot_orbit(coeff)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.10 小行星椭圆轨道拟合")
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="不显示轨道曲线图",
    )
    args = parser.parse_args()
    solve(show_plot=not args.no_plot)
