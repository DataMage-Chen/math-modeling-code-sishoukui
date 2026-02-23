"""
例题 6.7：线性系统初值问题解析求解。

运行示例：
  python ch06/ex06_07/solution.py
  python ch06/ex06_07/solution.py --samples 0 0.5 1 2
"""

import argparse

try:
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install sympy"
    ) from exc


def build_solution():
    """构造解析解并做符号校验。"""
    t = sp.symbols("t", real=True)

    a = sp.Matrix(
        [
            [1, 0, 0],
            [2, 1, -2],
            [3, 2, 1],
        ]
    )
    b = sp.Matrix([0, 0, sp.exp(t) * sp.cos(2 * t)])
    x0 = sp.Matrix([0, 1, 1])

    x1 = sp.Integer(0)
    x2 = sp.exp(t) * (sp.cos(2 * t) - sp.sin(2 * t) - sp.Rational(1, 2) * t * sp.sin(2 * t))
    x3 = sp.exp(t) * ((sp.Rational(2, 1) + t) / 2 * sp.cos(2 * t) + sp.Rational(5, 4) * sp.sin(2 * t))
    x = sp.Matrix([sp.simplify(x1), sp.simplify(x2), sp.simplify(x3)])

    residual = sp.simplify(sp.diff(x, t) - a * x - b)
    x_at_0 = sp.simplify(x.subs(t, 0))

    return {
        "t": t,
        "a": a,
        "b": b,
        "x0": x0,
        "x": x,
        "residual": residual,
        "x_at_0": x_at_0,
    }


def print_report(res, samples):
    """打印解析解与校验信息。"""
    t = res["t"]
    x = res["x"]

    print("=== 例题 6.7 解析解结果 ===")
    print("系统: x'(t) = A x(t) + b(t)")
    print("A =")
    print(res["a"])
    print(f"b(t) = {res['b']}")
    print(f"初值 x(0) = {res['x0']}")

    print("\n分量解析解：")
    print(f"  x1(t) = {x[0]}")
    print(f"  x2(t) = {x[1]}")
    print(f"  x3(t) = {x[2]}")

    print("\n校验：")
    print(f"  x(0) = {res['x_at_0']}")
    print(f"  残差 x'(t)-Ax(t)-b(t) = {res['residual']}")

    if samples:
        print("\n样例数值：")
        for sv in samples:
            xv = sp.N(x.subs(t, float(sv)))
            x1v, x2v, x3v = [float(xv[i]) for i in range(3)]
            print(f"  t={sv:8.3f} -> x=[{x1v:12.8f}, {x2v:12.8f}, {x3v:12.8f}]")


def solve(samples=None):
    res = build_solution()
    print_report(res, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.7 线性系统初值问题解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 t 的 x(t) 数值",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

