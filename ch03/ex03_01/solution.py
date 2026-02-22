"""
例题 3.1：非线性规划求解（凸 QCP）。

运行：
  python ch03/ex03_01/solution.py
"""

from gurobipy import GRB, Model


def bisection_root():
    """
    对方程 2*x^3 + 3*x - 2 = 0 做二分求根（区间 [0,1]）。
    用于对最优解做一维消元后的数值校验。
    """

    def f(x):
        return 2 * x * x * x + 3 * x - 2

    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if f(mid) > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def solve():
    model = Model("ex03_01_nlp")
    model.Params.OutputFlag = 0

    x1 = model.addVar(lb=0.0, name="x1")
    x2 = model.addVar(lb=0.0, name="x2")

    model.addConstr(-x1 + x2 - 2 <= 0, name="g1")
    model.addConstr(x1 * x1 - x2 + 1 <= 0, name="g2")

    model.setObjective(x1 * x1 + x2 * x2 - 4 * x1 + 4, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    f_star = model.ObjVal

    g1_val = -x1_star + x2_star - 2
    g2_val = x1_star * x1_star - x2_star + 1

    print("=== 例题 3.1 求解结果 ===")
    print(f"最优目标值 f*: {f_star:.10f}")
    print(f"x1*: {x1_star:.10f}")
    print(f"x2*: {x2_star:.10f}")
    print("约束校验：")
    print(f"  g1(x*) = {g1_val:.10e} <= 0")
    print(f"  g2(x*) = {g2_val:.10e} <= 0")

    # 一维消元校验：最优点应在 g2 活跃边界 x2 = x1^2 + 1 上
    x1_check = bisection_root()
    x2_check = x1_check * x1_check + 1
    f_check = x1_check * x1_check + x2_check * x2_check - 4 * x1_check + 4

    print("一维消元数值校验：")
    print(f"  x1_check: {x1_check:.10f}")
    print(f"  x2_check: {x2_check:.10f}")
    print(f"  f_check : {f_check:.10f}")
    print(f"  |x1*-x1_check| = {abs(x1_star - x1_check):.3e}")
    print(f"  |f*-f_check|  = {abs(f_star - f_check):.3e}")


if __name__ == "__main__":
    solve()
