"""
例题 6.4：求解二阶线性微分方程
y'' - 2y' + y = e^x, y(0)=1, y'(0)=-1。

运行示例：
  python ch06/ex06_04/solution.py
  python ch06/ex06_04/solution.py --samples 0 0.5 1 2
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
    """符号求解并做校验。"""
    x = sp.symbols("x", real=True)
    y = sp.Function("y")

    ode = sp.Eq(sp.diff(y(x), x, 2) - 2 * sp.diff(y(x), x) + y(x), sp.exp(x))
    sol = sp.dsolve(ode, ics={y(0): 1, sp.diff(y(x), x).subs(x, 0): -1})
    y_expr = sp.simplify(sol.rhs)
    y_prime = sp.simplify(sp.diff(y_expr, x))

    residual = sp.simplify(sp.diff(y_expr, x, 2) - 2 * sp.diff(y_expr, x) + y_expr - sp.exp(x))
    y0 = sp.simplify(y_expr.subs(x, 0))
    yp0 = sp.simplify(y_prime.subs(x, 0))

    return x, y_expr, y_prime, residual, y0, yp0


def print_report(x, y_expr, y_prime, residual, y0, yp0, samples):
    """打印结果。"""
    print("=== 例题 6.4 解析解结果 ===")
    print("方程: y'' - 2y' + y = e^x")
    print("初值: y(0)=1, y'(0)=-1")
    print(f"解析解: y(x) = {y_expr}")

    print("\n校验：")
    print(f"  y(0)  = {y0}")
    print(f"  y'(0) = {yp0}")
    print(f"  代回残差 y''-2y'+y-e^x = {residual}")

    if samples:
        print("\n样例函数值：")
        for sx in samples:
            yv = sp.N(y_expr.subs(x, float(sx)))
            ypv = sp.N(y_prime.subs(x, float(sx)))
            print(f"  x={sx:8.3f}: y={float(yv):12.8f}, y'={float(ypv):12.8f}")


def solve(samples=None):
    x, y_expr, y_prime, residual, y0, yp0 = solve_symbolic()
    print_report(x, y_expr, y_prime, residual, y0, yp0, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.4 二阶线性微分方程解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 x 的 y(x)、y'(x)",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

