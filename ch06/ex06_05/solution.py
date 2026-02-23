"""
例题 6.5：四阶线性常系数微分方程解析求解。

运行示例：
  python ch06/ex06_05/solution.py
  python ch06/ex06_05/solution.py --samples 0 0.5 1 2
"""

import argparse

try:
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install sympy"
    ) from exc


def manual_solution_expr(t):
    """手工推导得到的解析式。"""
    return (
        -sp.Rational(7, 3) * sp.exp(-t)
        + sp.Rational(9, 2) * sp.exp(-2 * t)
        - sp.Rational(14, 5) * sp.exp(-3 * t)
        + sp.Rational(19, 30) * sp.exp(-4 * t)
        - sp.Rational(1, 5) * sp.exp(-t) * sp.sin(t)
    )


def solve_symbolic():
    """使用 sympy.dsolve 求解，并与手工结果对比。"""
    t = sp.symbols("t", real=True)
    y = sp.Function("y")

    u = sp.exp(-t) * sp.cos(t)
    rhs = sp.diff(u, t, 2)

    ode = sp.Eq(
        sp.diff(y(t), t, 4)
        + 10 * sp.diff(y(t), t, 3)
        + 35 * sp.diff(y(t), t, 2)
        + 50 * sp.diff(y(t), t)
        + 24 * y(t),
        rhs,
    )

    sol = sp.dsolve(
        ode,
        ics={
            y(0): 0,
            sp.diff(y(t), t).subs(t, 0): -1,
            sp.diff(y(t), t, 2).subs(t, 0): 1,
            sp.diff(y(t), t, 3).subs(t, 0): 1,
        },
    )
    y_sym = sp.simplify(sol.rhs)
    y_manual = sp.simplify(manual_solution_expr(t))

    # 校验：符号解与手工解等价
    diff_expr = sp.simplify(sp.expand(y_sym - y_manual))

    # 初值校验
    y0 = sp.simplify(y_manual.subs(t, 0))
    yp0 = sp.simplify(sp.diff(y_manual, t).subs(t, 0))
    ypp0 = sp.simplify(sp.diff(y_manual, t, 2).subs(t, 0))
    yppp0 = sp.simplify(sp.diff(y_manual, t, 3).subs(t, 0))

    # 方程残差校验
    residual = sp.simplify(
        sp.diff(y_manual, t, 4)
        + 10 * sp.diff(y_manual, t, 3)
        + 35 * sp.diff(y_manual, t, 2)
        + 50 * sp.diff(y_manual, t)
        + 24 * y_manual
        - rhs
    )

    return {
        "t": t,
        "u": u,
        "rhs": sp.simplify(rhs),
        "y_sym": y_sym,
        "y_manual": y_manual,
        "diff_expr": diff_expr,
        "y0": y0,
        "yp0": yp0,
        "ypp0": ypp0,
        "yppp0": yppp0,
        "residual": residual,
    }


def print_report(res, samples):
    """打印解与校验信息。"""
    t = res["t"]
    y_manual = res["y_manual"]

    print("=== 例题 6.5 解析解结果 ===")
    print(f"输入信号: u(t) = {res['u']}")
    print(f"右端项:   u''(t) = {res['rhs']}")
    print("方程: y'''' + 10y''' + 35y'' + 50y' + 24y = u''(t)")
    print("初值: y(0)=0, y'(0)=-1, y''(0)=1, y'''(0)=1")
    print(f"\n手工解析解: y(t) = {y_manual}")
    print(f"SymPy 解析解: y(t) = {res['y_sym']}")
    print(f"两者差值(应为0): {res['diff_expr']}")

    print("\n校验：")
    print(f"  y(0)     = {res['y0']}")
    print(f"  y'(0)    = {res['yp0']}")
    print(f"  y''(0)   = {res['ypp0']}")
    print(f"  y'''(0)  = {res['yppp0']}")
    print(f"  代回残差 = {res['residual']}")

    if samples:
        print("\n样例函数值：")
        for sv in samples:
            yv = sp.N(y_manual.subs(t, float(sv)))
            print(f"  t={sv:8.3f} -> y(t)={float(yv):12.8f}")


def solve(samples=None):
    res = solve_symbolic()
    print_report(res, samples or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.5 四阶线性微分方程解析求解")
    parser.add_argument(
        "--samples",
        nargs="*",
        type=float,
        default=[0.0, 1.0, 2.0],
        help="可选：输出指定 t 的 y(t) 值",
    )
    args = parser.parse_args()
    solve(samples=args.samples)

