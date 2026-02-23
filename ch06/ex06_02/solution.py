"""
例题 6.2：目标跟踪问题（追击曲线）解析求解。

运行示例：
  python ch06/ex06_02/solution.py
  python ch06/ex06_02/solution.py --samples 6
"""

import argparse

try:
    import sympy as sp
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install sympy"
    ) from exc


def symbolic_solution():
    """返回追击曲线和命中信息的解析表达式。"""
    x = sp.symbols("x", real=True)
    u = 1 - x

    # p(x)=dy/dx
    p_expr = sp.simplify((u ** (-sp.Rational(1, 5)) - u ** (sp.Rational(1, 5))) / 2)

    # 由 y(0)=0 积分得到 y(x)
    y_expr = sp.simplify(
        sp.Rational(5, 24)
        - sp.Rational(5, 8) * u ** (sp.Rational(4, 5))
        + sp.Rational(5, 12) * u ** (sp.Rational(6, 5))
    )

    # 目标乙舰的竖直坐标 s(x)=y+p*(1-x)
    s_expr = sp.simplify(y_expr + p_expr * (1 - x))

    y_hit = sp.simplify(sp.limit(y_expr, x, 1, dir="-"))
    s_hit = sp.simplify(sp.limit(s_expr, x, 1, dir="-"))

    v0 = sp.symbols("v0", positive=True, real=True)
    t_hit = sp.simplify(s_hit / v0)

    return {
        "x": x,
        "p_expr": p_expr,
        "y_expr": y_expr,
        "s_expr": s_expr,
        "y_hit": y_hit,
        "s_hit": s_hit,
        "t_hit": t_hit,
    }


def print_report(res, samples):
    """打印解析解与样例点。"""
    x = res["x"]
    p_expr = res["p_expr"]
    y_expr = res["y_expr"]
    s_expr = res["s_expr"]

    print("=== 例题 6.2 解析解结果（目标跟踪） ===")
    print("乙舰轨迹: x=1, y=s=v0*t")
    print("速度关系: |v_导弹| = 5 v0")
    print("\n导弹轨迹切线斜率：")
    print(f"  dy/dx = {sp.simplify(p_expr)}")
    print("\n导弹轨迹方程：")
    print(f"  y(x) = {sp.simplify(y_expr)}")
    print("\n乙舰对应竖直坐标（便于校验）：")
    print(f"  s(x) = {sp.simplify(s_expr)}")
    print("\n命中条件（x->1^-）结果：")
    print(f"  命中点导弹坐标 y_hit = {res['y_hit']} = {float(sp.N(res['y_hit'])):.10f}")
    print(f"  乙舰行驶距离   L     = {res['s_hit']} = {float(sp.N(res['s_hit'])):.10f}")
    print(f"  命中时间 t_hit = {res['t_hit']}")
    print("  即 t_hit = 5/(24*v0)")

    if samples > 1:
        print("\n轨迹样例点（x, y(x), s(x), 竖直间距 s-y）：")
        for i in range(samples):
            xv = i / (samples - 1)
            if xv >= 1.0:
                xv = 1.0 - 1e-10
            yv = float(sp.N(y_expr.subs(x, xv)))
            sv = float(sp.N(s_expr.subs(x, xv)))
            print(f"  x={xv:.6f}, y={yv:.8f}, s={sv:.8f}, s-y={sv - yv:.8f}")


def solve(samples=6):
    res = symbolic_solution()
    print_report(res, samples=samples)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.2 目标跟踪问题解析求解")
    parser.add_argument("--samples", type=int, default=6, help="打印轨迹样例点数量，默认 6")
    args = parser.parse_args()

    solve(samples=max(2, args.samples))

