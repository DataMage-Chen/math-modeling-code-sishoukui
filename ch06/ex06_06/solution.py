"""
例题 6.6：线性常系数微分方程组柯西问题解析求解。

运行示例：
  python ch06/ex06_06/solution.py
  python ch06/ex06_06/solution.py --samples 0 0.5 1 2
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
    """构造解析解并做符号校验。"""
    t = sp.symbols("t", real=True)

    a = sp.Matrix(
        [
            [3, -1, 1],
            [2, 0, -1],
            [1, -1, 2],
        ]
    )
    x0 = sp.Matrix([1, 1, 1])

    # 由特征分解推得的显式分量解
    x1 = sp.Rational(1, 6) - sp.Rational(1, 2) * sp.exp(2 * t) + sp.Rational(4, 3) * sp.exp(3 * t)
    x2 = sp.Rational(5, 6) - sp.Rational(1, 2) * sp.exp(2 * t) + sp.Rational(2, 3) * sp.exp(3 * t)
    x3 = sp.Rational(1, 3) + sp.Rational(2, 3) * sp.exp(3 * t)
    x_expr = sp.Matrix([sp.simplify(x1), sp.simplify(x2), sp.simplify(x3)])

    # 用矩阵指数形式求一份解用于交叉验证
    x_mat_exp = sp.simplify((a * t).exp() * x0)
    diff_solution = sp.simplify(x_expr - x_mat_exp)

    # 初值与方程残差校验
    x_at_0 = sp.simplify(x_expr.subs(t, 0))
    residual = sp.simplify(sp.diff(x_expr, t) - a * x_expr)

    # 特征信息（用于报告）
    char_poly = sp.factor(a.charpoly(sp.symbols("lam")).as_expr())
    eigen_data = a.eigenvects()

    return {
        "t": t,
        "a": a,
        "x0": x0,
        "x_expr": x_expr,
        "x_mat_exp": x_mat_exp,
        "diff_solution": diff_solution,
        "x_at_0": x_at_0,
        "residual": residual,
        "char_poly": char_poly,
        "eigen_data": eigen_data,
    }


def print_report(res, samples):
    """打印求解结果与校验信息。"""
    t = res["t"]
    x_expr = res["x_expr"]

    print("=== 例题 6.6 解析解结果 ===")
    print("系统: dx/dt = A x, x(0)=[1,1,1]^T")
    print("A =")
    print(res["a"])
    print(f"\n特征多项式: det(A-lambda I) = {res['char_poly']}")
    print("特征值及特征向量（SymPy 输出）：")
    for lam, mult, vecs in res["eigen_data"]:
        print(f"  lambda={lam}, 代数重数={mult}, 向量={vecs}")

    print("\n分量解析解：")
    print(f"  x1(t) = {x_expr[0]}")
    print(f"  x2(t) = {x_expr[1]}")
    print(f"  x3(t) = {x_expr[2]}")

    print("\n校验：")
    print(f"  初值 x(0) = {res['x_at_0']}")
    print(f"  方程残差 dx/dt - A x = {res['residual']}")
    print(f"  与 exp(At)x0 差值 = {res['diff_solution']}")

    if samples:
        print("\n样例数值：")
        for sv in samples:
            xv = sp.N(x_expr.subs(t, float(sv)))
            x1v, x2v, x3v = [float(xv[i]) for i in range(3)]
            print(f"  t={sv:8.3f} -> x=[{x1v:12.8f}, {x2v:12.8f}, {x3v:12.8f}]")


def solve(samples=None):
    res = solve_symbolic()
    print_report(res, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.6 线性系统解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 t 的 x(t) 数值",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

