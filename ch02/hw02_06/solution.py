"""
习题 2.6：整数线性规划求解。

运行：
  python ch02/hw02_06/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("hw02_06_ilp")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")
    x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")
    x3 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x3")
    x4 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x4")
    x5 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x5")

    model.addConstr(x1 + x2 + x5 >= 30, name="c1")
    model.addConstr(x3 + x4 >= 30, name="c2")
    model.addConstr(3 * x1 + 2 * x3 <= 120, name="c3")
    model.addConstr(3 * x2 + 2 * x4 + x5 <= 48, name="c4")

    model.setObjective(20 * x1 + 90 * x2 + 80 * x3 + 70 * x4 + 30 * x5, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    print("=== 习题 2.6 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最小目标值: {model.ObjVal:.6f}")
    print("最优整数解：")
    print(f"  x1 = {int(round(x1.X))}")
    print(f"  x2 = {int(round(x2.X))}")
    print(f"  x3 = {int(round(x3.X))}")
    print(f"  x4 = {int(round(x4.X))}")
    print(f"  x5 = {int(round(x5.X))}")

    print("约束校验：")
    print(f"  x1+x2+x3 = {x1.X + x2.X + x3.X:.6f} >= 30")
    print(f"  x3+x4 = {x3.X + x4.X:.6f} >= 30")
    print(f"  3x1+2x3 = {3*x1.X + 2*x3.X:.6f} <= 120")
    print(f"  3x2+2x4+x5 = {3*x2.X + 2*x4.X + x5.X:.6f} <= 48")


if __name__ == "__main__":
    solve()
