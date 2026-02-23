"""
例题 6.9（续例 6.3）：
求微分方程 y'=-2y+2x^2+2x, y(0)=1 在 [0,0.5] 的数值解，
并与符号解在同图对比。

运行示例：
  python ch06/ex06_09/solution.py
  python ch06/ex06_09/solution.py --method DOP853 --rtol 1e-10 --atol 1e-12
  python ch06/ex06_09/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
    from scipy.integrate import solve_ivp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy scipy matplotlib sympy"
    ) from exc


# 让 matplotlib 尽量正确显示中文和负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def rhs(x, y):
    """方程右端：y' = -2y + 2x^2 + 2x。"""
    return -2.0 * y + 2.0 * x ** 2 + 2.0 * x


def symbolic_exact_solution():
    """用 SymPy 求符号解并返回可数值计算函数。"""
    x = sp.symbols("x", real=True)
    y = sp.Function("y")

    ode = sp.Eq(sp.diff(y(x), x), -2 * y(x) + 2 * x ** 2 + 2 * x)
    sol = sp.dsolve(ode, ics={y(0): 1})
    y_expr = sp.simplify(sol.rhs)
    y_func = sp.lambdify(x, y_expr, modules="numpy")

    return x, y_expr, y_func


def solve_numerical(method, rtol, atol, n_grid):
    """求数值解，并返回评价网格上的值。"""
    x0, x1 = 0.0, 0.5
    x_eval = np.linspace(x0, x1, n_grid)

    sol = solve_ivp(
        fun=rhs,
        t_span=(x0, x1),
        y0=[1.0],
        t_eval=x_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        raise RuntimeError(f"数值求解失败：{sol.message}")

    return x_eval, sol.y[0], sol


def print_report(x_symbol, y_expr, x_eval, y_num, y_exact, solver_result, method):
    """打印解析解、数值解精度与样例点。"""
    err = y_num - y_exact
    max_abs_err = float(np.max(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))

    # 用符号方式再校验一次解析解是否满足方程
    residual_expr = sp.simplify(sp.diff(y_expr, x_symbol) + 2 * y_expr - (2 * x_symbol ** 2 + 2 * x_symbol))

    print("=== 例题 6.9 求解结果（数值解 vs 符号解） ===")
    print("方程: y' = -2y + 2x^2 + 2x, y(0)=1, x∈[0,0.5]")
    print(f"符号解: y(x) = {y_expr}")
    print(f"符号解校验残差: {residual_expr}")
    print("\n求解器信息：")
    print(f"  method={method}")
    print(f"  nfev={solver_result.nfev}, njev={solver_result.njev}, nlu={solver_result.nlu}")
    print("\n误差指标（数值解 - 符号解）：")
    print(f"  max|error| = {max_abs_err:.12e}")
    print(f"  RMSE       = {rmse:.12e}")

    sample_points = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5], dtype=float)
    y_num_sample = np.interp(sample_points, x_eval, y_num)
    y_exact_sample = np.interp(sample_points, x_eval, y_exact)
    print("\n样例点对比：")
    print("      x        y_num         y_exact       error")
    for xv, yn, ye in zip(sample_points, y_num_sample, y_exact_sample):
        print(f"  {xv:6.2f}   {yn:12.9f}   {ye:12.9f}   {yn - ye: .3e}")


def plot_result(x_eval, y_num, y_exact):
    """同图绘制数值解与符号解。"""
    err = y_num - y_exact

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)

    ax1.plot(x_eval, y_exact, color="#1f77b4", linewidth=2.0, label="符号解")
    ax1.plot(x_eval, y_num, color="#d62728", linewidth=1.8, linestyle="--", label="数值解")
    ax1.set_title("例题 6.9：数值解与符号解对比")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x_eval, err, color="#2ca02c", linewidth=1.8, label="误差 y_num - y_exact")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("x")
    ax2.set_ylabel("误差")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(method="RK45", rtol=1e-9, atol=1e-12, n_grid=301, show_plot=True):
    x_symbol, y_expr, y_exact_func = symbolic_exact_solution()
    x_eval, y_num, solver_result = solve_numerical(method=method, rtol=rtol, atol=atol, n_grid=n_grid)
    y_exact = np.asarray(y_exact_func(x_eval), dtype=float)

    print_report(x_symbol, y_expr, x_eval, y_num, y_exact, solver_result, method=method)
    if show_plot:
        plot_result(x_eval, y_num, y_exact)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.9 数值解与符号解对比")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-9, help="相对误差容差，默认 1e-9")
    parser.add_argument("--atol", type=float, default=1e-12, help="绝对误差容差，默认 1e-12")
    parser.add_argument("--n-grid", type=int, default=301, help="对比网格点数，默认 301")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        n_grid=max(20, args.n_grid),
        show_plot=not args.no_plot,
    )
