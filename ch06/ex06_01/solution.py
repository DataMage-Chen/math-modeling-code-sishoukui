"""
例题 6.1：牛顿冷却模型解析求解。

运行示例：
  python ch06/ex06_01/solution.py
  python ch06/ex06_01/solution.py --query-minutes 0 20 40 60
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
    """用符号法推导解析式并计算达到 30°C 的时间。"""
    t = sp.symbols("t", real=True)
    k = sp.symbols("k", positive=True, real=True)
    c1 = sp.symbols("c1", real=True)

    # 牛顿冷却方程：du/dt = -k(u-20)，其通解形式写作 20 + c1*exp(-k*t)
    u_expr = 20 + c1 * sp.exp(-k * t)

    # 由 u(0)=100, u(20)=60 解出 c1、k
    equations = [
        sp.Eq(u_expr.subs(t, 0), 100),
        sp.Eq(u_expr.subs(t, 20), 60),
    ]
    sol = sp.solve(equations, [c1, k], dict=True)[0]

    u_final = sp.simplify(u_expr.subs(sol))
    t_30 = sp.solve(sp.Eq(u_final, 30), t)[0]

    return {
        "t_symbol": t,
        "k": sp.simplify(sol[k]),
        "c1": sp.simplify(sol[c1]),
        "u_expr": u_final,
        "t_30": sp.simplify(t_30),
    }


def print_report(result, query_minutes):
    """打印解析式和关键结果。"""
    t = result["t_symbol"]
    u_expr = result["u_expr"]
    k = result["k"]
    t_30 = result["t_30"]

    print("=== 例题 6.1 解析解结果（牛顿冷却） ===")
    print("模型方程: du/dt = -k(u-20)")
    print("初值条件: u(0)=100, u(20)=60")
    print(f"求得冷却系数: k = {sp.simplify(k)} ≈ {float(sp.N(k)):.10f} (min^-1)")
    print(f"温度解析式: u(t) = {sp.simplify(u_expr)}")
    print(f"达到 30°C 的时间: t = {sp.simplify(t_30)} min ≈ {float(sp.N(t_30)):.6f} min")
    print(f"相对 t=20min 再经过: {float(sp.N(t_30 - 20)):.6f} min")

    if query_minutes:
        print("\n指定时刻温度：")
        for tm in query_minutes:
            uv = sp.N(u_expr.subs(t, float(tm)))
            print(f"  t={tm:8.3f} min -> u={float(uv):10.6f} °C")


def solve(query_minutes=None):
    result = solve_symbolic()
    print_report(result, query_minutes=query_minutes or [])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.1 牛顿冷却定律解析求解")
    parser.add_argument(
        "--query-minutes",
        nargs="*",
        type=float,
        default=[0, 20, 60],
        help="可选：需要输出温度的时刻（min）",
    )
    args = parser.parse_args()
    solve(query_minutes=args.query_minutes)
