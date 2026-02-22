"""
例题 3.7：求非线性规划。

运行：
  python ch03/ex03_07/solution.py
"""

import math

from gurobipy import GRB, Model


def constraint_slack(constr):
    """兼容线性约束(Constr)与二次约束(QConstr)的松弛量读取。"""
    try:
        return constr.getAttr("QCSlack")
    except Exception:
        return constr.getAttr("Slack")


def reduced_grid_check(num_points=100000):
    """
    用等式约束消元后做一维网格校验（近似）：
      w = x3^2, x2 = 3 - 2w, x1 = 2 - x2^2, x3 = sqrt(w), w in [0, 1.5]
    """
    tol = 1e-8
    best = None

    for i in range(num_points + 1):
        w = 1.5 * i / num_points
        x3 = math.sqrt(w)
        x2 = 3 - 2 * w
        x1 = 2 - x2 * x2

        c1 = x1 * x1 - x2 + w
        c2 = x1 + x2 * x2 + x3 * w - 20

        if x1 < -tol or x2 < -tol or c1 < -tol or c2 > tol:
            continue

        obj = x1 * x1 + x2 * x2 + w + 8
        if best is None or obj < best["obj"]:
            best = {
                "obj": obj,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "w": w,
                "c1": c1,
                "c2": c2,
            }
    return best


def solve():
    model = Model("ex03_07_nlp")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x1 = model.addVar(lb=0.0, name="x1")
    x2 = model.addVar(lb=0.0, name="x2")
    x3 = model.addVar(lb=0.0, name="x3")
    w = model.addVar(lb=0.0, name="w")  # w = x3^2

    c1 = model.addConstr(x1 * x1 - x2 + w >= 0, name="c1")
    c2 = model.addConstr(x1 + x2 * x2 + x3 * w <= 20, name="c2")
    c3 = model.addConstr(-x1 - x2 * x2 + 2 == 0, name="c3")
    c4 = model.addConstr(x2 + 2 * w == 3, name="c4")
    c5 = model.addConstr(w == x3 * x3, name="def_w")

    model.setObjective(x1 * x1 + x2 * x2 + w + 8, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    x3_star = x3.X
    w_star = w.X
    f_star = model.ObjVal

    print("=== 例题 3.7 求解结果 ===")
    print(f"最优目标值 f*: {f_star:.10f}")
    print(f"x1*: {x1_star:.10f}")
    print(f"x2*: {x2_star:.10f}")
    print(f"x3*: {x3_star:.10f}")
    print(f"w* (=x3^2): {w_star:.10f}")

    print("约束校验：")
    print(f"  c1: x1^2 - x2 + w = {x1_star * x1_star - x2_star + w_star:.10f} >= 0")
    print(
        f"  c2: x1 + x2^2 + x3*w = "
        f"{x1_star + x2_star * x2_star + x3_star * w_star:.10f} <= 20"
    )
    print(f"  c3: -x1 - x2^2 + 2 = {-x1_star - x2_star * x2_star + 2:.10e}")
    print(f"  c4: x2 + 2w - 3 = {x2_star + 2 * w_star - 3:.10e}")
    print(f"  c5: w - x3^2 = {w_star - x3_star * x3_star:.10e}")
    print("求解器松弛量/残差（供参考）：")
    print(f"  c1 slack = {constraint_slack(c1):.10e}")
    print(f"  c2 slack = {constraint_slack(c2):.10e}")
    print(f"  c3 slack = {constraint_slack(c3):.10e}")
    print(f"  c4 slack = {constraint_slack(c4):.10e}")
    print(f"  c5 slack = {constraint_slack(c5):.10e}")

    check = reduced_grid_check()
    if check is not None:
        print("一维消元网格校验（近似）：")
        print(f"  f_check ≈ {check['obj']:.10f}")
        print(f"  (x1,x2,x3) ≈ ({check['x1']:.10f}, {check['x2']:.10f}, {check['x3']:.10f})")
        print(f"  |f*-f_check| = {abs(f_star - check['obj']):.3e}")


if __name__ == "__main__":
    solve()
