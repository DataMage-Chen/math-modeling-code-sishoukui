"""
例题 5.20（续例 5.17）：
拟合五阶傅里叶级数，并求其与 y=2x-1 的交点。

运行：
  python ch05/ex05_20/solution.py
  python ch05/ex05_20/solution.py --order 5 --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import brentq, minimize_scalar
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


X_DATA = np.array([0.81, 0.91, 0.13, 0.91, 0.63, 0.098, 0.28, 0.55, 0.96, 0.96, 0.16, 0.97, 0.96], dtype=float)
Y_DATA = np.array([0.17, 0.12, 0.16, 0.0035, 0.37, 0.082, 0.34, 0.56, 0.15, -0.046, 0.17, -0.091, -0.071], dtype=float)


def line_value(x):
    """目标直线 y = 2x - 1。"""
    return 2.0 * x - 1.0


def build_design_matrix(x, w, order):
    """构造固定 w 下的傅里叶线性设计矩阵。"""
    cols = [np.ones_like(x)]
    for k in range(1, order + 1):
        cols.append(np.cos(k * w * x))
        cols.append(np.sin(k * w * x))
    return np.column_stack(cols)


def fit_given_w(x, y, w, order):
    """固定 w 时，对系数做线性最小二乘。"""
    design = build_design_matrix(x, w, order)
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    y_hat = design @ coef
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    return coef, y_hat, sse


def fourier_value(x, w, coef, order):
    """计算傅里叶级数函数值。"""
    x = np.asarray(x, dtype=float)
    y = np.full_like(x, fill_value=float(coef[0]), dtype=float)
    idx = 1
    for k in range(1, order + 1):
        ak = float(coef[idx])
        bk = float(coef[idx + 1])
        y += ak * np.cos(k * w * x) + bk * np.sin(k * w * x)
        idx += 2
    return y


def search_best_w(x, y, order, w_min=0.1, w_max=100.0, grid_n=3000, refine_top=8):
    """
    搜索最优 w：
    1) 粗网格扫描找候选
    2) 对前若干候选做局部一维精细化
    """
    ws = np.linspace(w_min, w_max, grid_n)
    sse_list = np.empty_like(ws)

    for i, w in enumerate(ws):
        _, _, sse = fit_given_w(x, y, w, order)
        sse_list[i] = sse

    idx_sorted = np.argsort(sse_list)
    candidate_ids = []
    for idx in idx_sorted:
        if all(abs(idx - j) > 2 for j in candidate_ids):
            candidate_ids.append(int(idx))
        if len(candidate_ids) >= refine_top:
            break

    best = None
    for idx in candidate_ids:
        i_l = max(0, idx - 2)
        i_r = min(grid_n - 1, idx + 2)
        lo = float(ws[i_l])
        hi = float(ws[i_r])
        if hi <= lo:
            continue

        def obj(w):
            return fit_given_w(x, y, w, order)[2]

        res = minimize_scalar(obj, bounds=(lo, hi), method="bounded", options={"xatol": 1e-10})
        w_opt = float(res.x)
        coef, y_hat, sse = fit_given_w(x, y, w_opt, order)
        if best is None or sse < best["sse"]:
            best = {"w": w_opt, "coef": coef, "y_hat": y_hat, "sse": sse}

    if best is None:
        raise RuntimeError("未找到可行的 w。")
    return best


def find_intersections(w, coef, order, left, right, n_grid=5000, tol=1e-8):
    """在区间 [left,right] 内求 f(x)=2x-1 的交点。"""
    xg = np.linspace(left, right, n_grid)
    yg = fourier_value(xg, w, coef, order)
    g = yg - line_value(xg)

    roots = []
    for i in range(n_grid - 1):
        x1, x2 = float(xg[i]), float(xg[i + 1])
        g1, g2 = float(g[i]), float(g[i + 1])

        if abs(g1) < tol:
            roots.append(x1)
        if g1 * g2 < 0:
            try:
                r = float(brentq(lambda t: float(fourier_value([t], w, coef, order)[0] - line_value(t)), x1, x2))
                roots.append(r)
            except ValueError:
                pass

    if abs(float(g[-1])) < tol:
        roots.append(float(xg[-1]))

    # 去重
    roots = sorted(roots)
    unique_roots = []
    for r in roots:
        if not unique_roots or abs(r - unique_roots[-1]) > 1e-5:
            unique_roots.append(r)

    return [(r, float(line_value(r))) for r in unique_roots]


def metrics(y, y_hat):
    """计算拟合指标。"""
    residual = y - y_hat
    sse = float(np.dot(residual, residual))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.dot(y - np.mean(y), y - np.mean(y)))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def print_report(order, best, intersections):
    """打印拟合参数与交点。"""
    w = best["w"]
    coef = best["coef"]
    sse, rmse, r2 = metrics(Y_DATA, best["y_hat"])

    print("=== 例题 5.20 拟合结果 ===")
    print(f"傅里叶阶数: {order}")
    print(f"最优频率参数 w = {w:.10f}")
    print("系数：")
    print(f"  a0 = {coef[0]:.10f}")
    idx = 1
    for k in range(1, order + 1):
        print(f"  a{k} = {coef[idx]:.10f}, b{k} = {coef[idx+1]:.10f}")
        idx += 2

    print("拟合指标：")
    print(f"  SSE  = {sse:.10f}")
    print(f"  RMSE = {rmse:.10f}")
    print(f"  R^2  = {r2:.10f}")

    if intersections:
        print("\n与 y=2x-1 的交点（在数据区间内）：")
        for i, (xr, yr) in enumerate(intersections, start=1):
            print(f"  交点{i}: x={xr:.10f}, y={yr:.10f}")
    else:
        print("\n在数据区间内未检测到与 y=2x-1 的交点。")


def plot_result(order, best, intersections, left, right):
    """绘制数据散点、傅里叶拟合曲线、直线及交点。"""
    w = best["w"]
    coef = best["coef"]

    x_dense = np.linspace(left, right, 2000)
    y_fit = fourier_value(x_dense, w, coef, order)
    y_line = line_value(x_dense)

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=45, label="原始数据")
    ax.plot(x_dense, y_fit, color="#d62728", linewidth=2.0, label=f"{order}阶傅里叶拟合")
    ax.plot(x_dense, y_line, color="#2ca02c", linewidth=1.8, linestyle="--", label="y=2x-1")

    if intersections:
        x_root = [p[0] for p in intersections]
        y_root = [p[1] for p in intersections]
        ax.scatter(x_root, y_root, color="black", marker="x", s=70, label="交点", zorder=4)

    ax.set_title("例5.20 五阶傅里叶拟合与直线交点")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(order=5, show_plot=True):
    left = float(np.min(X_DATA))
    right = float(np.max(X_DATA))

    best = search_best_w(X_DATA, Y_DATA, order=order, w_min=0.1, w_max=100.0, grid_n=3000, refine_top=10)
    intersections = find_intersections(
        w=best["w"],
        coef=best["coef"],
        order=order,
        left=left,
        right=right,
        n_grid=8000,
    )

    print_report(order, best, intersections)
    if show_plot:
        plot_result(order, best, intersections, left, right)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.20 五阶傅里叶拟合与交点求解")
    parser.add_argument("--order", type=int, default=5, help="傅里叶阶数，默认 5")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()
    solve(order=args.order, show_plot=not args.no_plot)
