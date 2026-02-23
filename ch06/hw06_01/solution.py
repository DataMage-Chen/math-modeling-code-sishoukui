"""
习题 6.1：线性常微分方程的符号解与积分曲线绘制

题目：y' - y = sin(x)，分别取初值 y(0)=1,2,3,4，
在同一窗口绘制区间 -2<=x<=4 上的四条积分曲线。

运行示例：
  python ch06/hw06_01/solution.py
  python ch06/hw06_01/solution.py --ics 1 2 3 4 --x-min -2 --x-max 4
  python ch06/hw06_01/solution.py --ics 0 1.5 3 5 --num-points 1200
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy matplotlib sympy"
    ) from exc


# 让 matplotlib 尽量正确显示中文和负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def solve_symbolic(initial_values):
    """求解通解，并根据给定初值列表得到对应特解。"""
    x = sp.symbols("x", real=True)
    y = sp.Function("y")

    ode = sp.Eq(sp.diff(y(x), x) - y(x), sp.sin(x))
    general_rhs = sp.simplify(sp.dsolve(ode).rhs)

    # 从通解中自动提取积分常数（通常为 C1）
    constants = sorted(
        (s for s in general_rhs.free_symbols if s.name.startswith("C")),
        key=lambda s: s.name,
    )
    if not constants:
        raise RuntimeError("未能从通解中提取积分常数。")
    c_symbol = constants[0]

    particular_solutions = {}
    for y0 in initial_values:
        c_value = sp.solve(sp.Eq(general_rhs.subs(x, 0), y0), c_symbol)[0]
        y_expr = sp.simplify(general_rhs.subs(c_symbol, c_value))
        particular_solutions[y0] = y_expr

    return x, ode, general_rhs, particular_solutions


def print_report(ode, general_rhs, particular_solutions):
    """打印方程、通解和各初值对应的特解。"""
    print("=== 习题 6.1 求解结果 ===")
    print(f"微分方程: {sp.sstr(ode.lhs)} = {sp.sstr(ode.rhs)}")
    print(f"通解: y(x) = {sp.sstr(general_rhs)}")
    print("不同初值下的特解：")
    for y0, expr in particular_solutions.items():
        print(f"  y(0)={y0:g}: y(x) = {sp.sstr(expr)}")


def plot_solutions(x_symbol, particular_solutions, x_min, x_max, num_points):
    """在同一窗口绘制多条积分曲线。"""
    x_grid = np.linspace(x_min, x_max, num_points)

    plt.figure(figsize=(9, 6))
    for y0, expr in particular_solutions.items():
        y_func = sp.lambdify(x_symbol, expr, "numpy")
        y_grid = y_func(x_grid)
        plt.plot(x_grid, y_grid, linewidth=2.0, label=f"y(0)={y0:g}")

    plt.title("习题 6.1：方程 y' - y = sin(x) 的积分曲线")
    plt.xlabel("x")
    plt.ylabel("y(x)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(initial_values, x_min=-2.0, x_max=4.0, num_points=800, show_plot=True):
    """主流程：符号求解 + 报告输出 + 曲线绘制。"""
    if x_min >= x_max:
        raise ValueError("需要满足 x_min < x_max。")
    if num_points < 100:
        raise ValueError("num_points 建议不小于 100。")

    x_symbol, ode, general_rhs, particular_solutions = solve_symbolic(initial_values)
    print_report(ode, general_rhs, particular_solutions)

    if show_plot:
        plot_solutions(
            x_symbol=x_symbol,
            particular_solutions=particular_solutions,
            x_min=x_min,
            x_max=x_max,
            num_points=num_points,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.1：线性微分方程符号解与积分曲线绘图")
    parser.add_argument(
        "--ics",
        nargs="*",
        type=float,
        default=[1.0, 2.0, 3.0, 4.0],
        help="初值 y(0) 列表，默认 1 2 3 4",
    )
    parser.add_argument("--x-min", type=float, default=-2.0, help="绘图区间左端点，默认 -2")
    parser.add_argument("--x-max", type=float, default=4.0, help="绘图区间右端点，默认 4")
    parser.add_argument("--num-points", type=int, default=800, help="绘图采样点数，默认 800")
    parser.add_argument("--no-plot", action="store_true", help="仅输出符号解，不绘图")
    args = parser.parse_args()

    solve(
        initial_values=args.ics,
        x_min=args.x_min,
        x_max=args.x_max,
        num_points=args.num_points,
        show_plot=not args.no_plot,
    )
