"""
习题 6.2：Bessel 方程（n=1/2）的符号解与数值解

题目：
  x^2 y'' + x y' + (x^2 - n^2) y = 0, n = 1/2
  y(pi/2)=2, y'(pi/2)=-2/pi

运行示例：
  python ch06/hw06_02/solution.py
  python ch06/hw06_02/solution.py --x-min 0.2 --x-max 20 --num-points 1500
  python ch06/hw06_02/solution.py --rtol 1e-9 --atol 1e-11 --no-plot
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


def symbolic_solution():
    """给出 n=1/2 时的通解和满足初值条件的解析解。"""
    x = sp.symbols("x", positive=True, real=True)
    n = sp.Rational(1, 2)
    c1, c2 = sp.symbols("C1 C2", real=True)

    # n=1/2 时可写成 sin/cos 与 x^{-1/2} 的线性组合
    y_general = (c1 * sp.sin(x) + c2 * sp.cos(x)) / sp.sqrt(x)
    y_general_prime = sp.diff(y_general, x)

    x0 = sp.pi / 2
    conditions = [
        sp.Eq(y_general.subs(x, x0), 2),
        sp.Eq(y_general_prime.subs(x, x0), -sp.Rational(2, 1) / sp.pi),
    ]
    constant_solution = sp.solve(conditions, [c1, c2], dict=True)[0]
    y_exact = sp.simplify(y_general.subs(constant_solution))

    # 与 BesselJ(1/2, x) 的等价表示
    y_bessel = sp.simplify(sp.pi * sp.besselj(sp.Rational(1, 2), x))

    # 校验方程残差
    residual = sp.simplify(
        x**2 * sp.diff(y_exact, x, 2) + x * sp.diff(y_exact, x) + (x**2 - n**2) * y_exact
    )

    return {
        "x": x,
        "n": n,
        "x0": x0,
        "c1": c1,
        "c2": c2,
        "y_general": sp.simplify(y_general),
        "constants": constant_solution,
        "y_exact": y_exact,
        "y_bessel": y_bessel,
        "residual": residual,
    }


def build_rhs(n_value):
    """把二阶方程改写成一阶系统以便数值积分。"""

    def rhs(x, state):
        y_val, dy_val = state
        # y'' + (1/x)y' + (1 - n^2/x^2)y = 0
        ddy_val = -(dy_val / x) - (1.0 - (n_value**2) / (x**2)) * y_val
        return np.array([dy_val, ddy_val], dtype=float)

    return rhs


def numerical_solution(x_min, x_max, num_points, rtol, atol):
    """从 x0=pi/2 向左右两侧积分，得到数值解。"""
    x0 = float(np.pi / 2.0)
    n_value = 0.5

    if not (0.0 < x_min < x0 < x_max):
        raise ValueError("需要满足 0 < x_min < pi/2 < x_max（x=0 为奇点）。")
    if num_points < 200:
        raise ValueError("num_points 建议不小于 200。")

    left_count = max(2, int(num_points * (x0 - x_min) / (x_max - x_min)))
    right_count = max(2, num_points - left_count + 1)

    x_eval_left = np.linspace(x0, x_min, left_count)
    x_eval_right = np.linspace(x0, x_max, right_count)
    y0 = np.array([2.0, -2.0 / np.pi], dtype=float)

    rhs = build_rhs(n_value)
    sol_left = solve_ivp(
        rhs,
        (x0, x_min),
        y0,
        method="RK45",
        t_eval=x_eval_left,
        rtol=rtol,
        atol=atol,
    )
    sol_right = solve_ivp(
        rhs,
        (x0, x_max),
        y0,
        method="RK45",
        t_eval=x_eval_right,
        rtol=rtol,
        atol=atol,
    )

    if not sol_left.success:
        raise RuntimeError(f"左侧积分失败：{sol_left.message}")
    if not sol_right.success:
        raise RuntimeError(f"右侧积分失败：{sol_right.message}")

    x_left = sol_left.t[::-1]
    y_left = sol_left.y[0][::-1]
    x_right = sol_right.t[1:]
    y_right = sol_right.y[0][1:]

    x_all = np.concatenate([x_left, x_right])
    y_all = np.concatenate([y_left, y_right])

    return {
        "x": x_all,
        "y": y_all,
        "left_nfev": sol_left.nfev,
        "right_nfev": sol_right.nfev,
    }


def print_report(symbolic_result, numeric_result, error_stats):
    """打印解析解和数值解对比信息。"""
    x0 = symbolic_result["x0"]
    constants = symbolic_result["constants"]
    c1 = symbolic_result["c1"]
    c2 = symbolic_result["c2"]

    print("=== 习题 6.2 求解结果（Bessel 方程，n=1/2） ===")
    print("方程: x^2*y'' + x*y' + (x^2 - 1/4)*y = 0")
    print(f"初值: y(pi/2)=2, y'(pi/2)=-2/pi（x0={float(sp.N(x0)):.10f}）")
    print(f"通解（n=1/2）: y(x) = {sp.sstr(symbolic_result['y_general'])}")
    print(
        "由初值解得常数: "
        f"C1={sp.sstr(sp.simplify(constants[c1]))}, "
        f"C2={sp.sstr(sp.simplify(constants[c2]))}"
    )
    print(f"解析特解: y(x) = {sp.sstr(symbolic_result['y_exact'])}")
    print(f"Bessel 等价形式: y(x) = {sp.sstr(symbolic_result['y_bessel'])}")
    print(f"方程残差校验（应为0）: {sp.sstr(symbolic_result['residual'])}")

    print("\n数值积分统计：")
    print(f"  左侧积分 nfev = {numeric_result['left_nfev']}")
    print(f"  右侧积分 nfev = {numeric_result['right_nfev']}")
    print("\n数值解与解析解误差：")
    print(f"  max|y_num-y_exact| = {error_stats['max_abs_err']:.6e}")
    print(f"  RMSE               = {error_stats['rmse']:.6e}")


def plot_result(x_grid, y_exact, y_num, abs_err):
    """绘制解析解、数值解及误差曲线。"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    ax1.plot(x_grid, y_exact, color="#1f77b4", linewidth=2.1, label="解析解")
    ax1.plot(x_grid, y_num, color="#d62728", linestyle="--", linewidth=1.8, label="数值解 RK45")
    ax1.set_ylabel("y(x)")
    ax1.set_title("习题 6.2：Bessel 方程（n=1/2）解析解与数值解对比")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x_grid, abs_err, color="#2ca02c", linewidth=1.8)
    ax2.set_xlabel("x")
    ax2.set_ylabel("|误差|")
    ax2.set_title("绝对误差曲线")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def solve(x_min=0.2, x_max=20.0, num_points=1200, rtol=1e-8, atol=1e-10, show_plot=True):
    """主流程：符号解 + 数值解 + 误差统计 + 画图。"""
    symbolic_result = symbolic_solution()
    numeric_result = numerical_solution(
        x_min=x_min,
        x_max=x_max,
        num_points=num_points,
        rtol=rtol,
        atol=atol,
    )

    x_symbol = symbolic_result["x"]
    y_exact_expr = symbolic_result["y_exact"]
    y_exact_func = sp.lambdify(x_symbol, y_exact_expr, "numpy")

    x_grid = numeric_result["x"]
    y_num = numeric_result["y"]
    y_exact = y_exact_func(x_grid)
    abs_err = np.abs(y_num - y_exact)

    error_stats = {
        "max_abs_err": float(np.max(abs_err)),
        "rmse": float(np.sqrt(np.mean((y_num - y_exact) ** 2))),
    }
    print_report(symbolic_result, numeric_result, error_stats)

    if show_plot:
        plot_result(x_grid, y_exact, y_num, abs_err)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题 6.2：Bessel 方程符号解与数值解")
    parser.add_argument("--x-min", type=float, default=0.2, help="绘图区间左端点，需 > 0，默认 0.2")
    parser.add_argument("--x-max", type=float, default=20.0, help="绘图区间右端点，默认 20")
    parser.add_argument("--num-points", type=int, default=1200, help="绘图与误差评估网格点数，默认 1200")
    parser.add_argument("--rtol", type=float, default=1e-8, help="solve_ivp 相对误差容限")
    parser.add_argument("--atol", type=float, default=1e-10, help="solve_ivp 绝对误差容限")
    parser.add_argument("--no-plot", action="store_true", help="仅输出文本结果，不绘图")
    args = parser.parse_args()

    solve(
        x_min=args.x_min,
        x_max=args.x_max,
        num_points=args.num_points,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )
