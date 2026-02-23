"""
例题 6.3：求解初值问题 y'=-2y+2x^2+2x, y(0)=1。

运行示例：
  python ch06/ex06_03/solution.py
  python ch06/ex06_03/solution.py --samples -1 0 1 2
"""

import argparse

try:
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install sympy"
    ) from exc


def solve_symbolic():
    """用符号法求解并返回关键表达式。"""
    x = sp.symbols("x", real=True)
    y = sp.Function("y")

    ode = sp.Eq(sp.diff(y(x), x), -2 * y(x) + 2 * x ** 2 + 2 * x)
    sol = sp.dsolve(ode, ics={y(0): 1})
    y_expr = sp.simplify(sol.rhs)

    # 方程与初值校验
    residual = sp.simplify(sp.diff(y_expr, x) + 2 * y_expr - (2 * x ** 2 + 2 * x))
    y0 = sp.simplify(y_expr.subs(x, 0))

    return x, y_expr, residual, y0


def print_report(x, y_expr, residual, y0, samples):
    """打印解析解与校验结果。"""
    print("=== 例题 6.3 解析解结果 ===")
    print("方程: y' = -2y + 2x^2 + 2x, y(0)=1")
    print(f"解析解: y(x) = {y_expr}")
    print("\n校验：")
    print(f"  初值 y(0) = {y0}")
    print(f"  代回残差 y'+2y-(2x^2+2x) = {residual}")

    if samples:
        print("\n样例函数值：")
        for sx in samples:
            yv = sp.N(y_expr.subs(x, float(sx)))
            print(f"  y({sx}) = {float(yv):.10f}")


def solve(samples=None):
    x, y_expr, residual, y0 = solve_symbolic()
    print_report(x, y_expr, residual, y0, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.3 一阶线性微分方程解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 x 的 y(x) 值",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

