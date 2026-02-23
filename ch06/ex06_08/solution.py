"""
例题 6.8：常微分方程组求解（混合初边值条件）。

方程组：
  f'' + g = 3
  g' + f' = 1
条件：
  f'(1)=0, f(0)=0, g(0)=0

运行示例：
  python ch06/ex06_08/solution.py
  python ch06/ex06_08/solution.py --samples 0 0.5 1 2
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
    """符号推导解析解并校验。"""
    t = sp.symbols("t", real=True)
    c1, c2 = sp.symbols("c1 c2", real=True)

    # 通解形式
    f_general = c1 * sp.exp(t) + c2 * sp.exp(-t) + t - 3
    g_general = 3 - c1 * sp.exp(t) - c2 * sp.exp(-t)

    # 由 f(0)=0, f'(1)=0 求常数（g(0)=0 与 f(0)=0 等价，这里作为额外校验）
    equations = [
        sp.Eq(f_general.subs(t, 0), 0),
        sp.Eq(sp.diff(f_general, t).subs(t, 1), 0),
    ]
    sol = sp.solve(equations, [c1, c2], dict=True)[0]

    f_expr = sp.simplify(f_general.subs(sol))
    g_expr = sp.simplify(g_general.subs(sol))

    # 方程残差
    r1 = sp.simplify(sp.diff(f_expr, t, 2) + g_expr - 3)
    r2 = sp.simplify(sp.diff(g_expr, t) + sp.diff(f_expr, t) - 1)

    # 条件校验
    f0 = sp.simplify(f_expr.subs(t, 0))
    g0 = sp.simplify(g_expr.subs(t, 0))
    fp1 = sp.simplify(sp.diff(f_expr, t).subs(t, 1))

    return {
        "t": t,
        "c1": sp.simplify(sol[c1]),
        "c2": sp.simplify(sol[c2]),
        "f_expr": f_expr,
        "g_expr": g_expr,
        "r1": r1,
        "r2": r2,
        "f0": f0,
        "g0": g0,
        "fp1": fp1,
    }


def print_report(res, samples):
    """打印解析解与校验结果。"""
    t = res["t"]
    f_expr = res["f_expr"]
    g_expr = res["g_expr"]

    print("=== 例题 6.8 解析解结果 ===")
    print("方程组:")
    print("  f'' + g = 3")
    print("  g' + f' = 1")
    print("条件: f'(1)=0, f(0)=0, g(0)=0")
    print(f"\n常数: C1={res['c1']}, C2={res['c2']}")
    print(f"f(t) = {f_expr}")
    print(f"g(t) = {g_expr}")

    print("\n校验：")
    print(f"  f''+g-3 = {res['r1']}")
    print(f"  g'+f'-1 = {res['r2']}")
    print(f"  f(0)    = {res['f0']}")
    print(f"  g(0)    = {res['g0']}")
    print(f"  f'(1)   = {res['fp1']}")

    if samples:
        print("\n样例数值：")
        for sv in samples:
            fv = sp.N(f_expr.subs(t, float(sv)))
            gv = sp.N(g_expr.subs(t, float(sv)))
            print(f"  t={sv:8.3f} -> f(t)={float(fv):12.8f}, g(t)={float(gv):12.8f}")


def solve(samples=None):
    res = solve_symbolic()
    print_report(res, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.8 常微分方程组解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 t 的 f(t)、g(t)",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

