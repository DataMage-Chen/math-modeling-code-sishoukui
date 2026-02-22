"""
习题 1.1：线性规划求解。

运行：
  python ch01/hw01_01/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("hw01_01")
    model.Params.OutputFlag = 0

    x1 = model.addVar(lb=0.0, name="x1")
    x2 = model.addVar(lb=0.0, name="x2")
    x3 = model.addVar(lb=0.0, name="x3")

    model.addConstr(x1 - 2 * x2 + x3 <= 11, name="c1")
    model.addConstr(-4 * x1 + x2 + 2 * x3 >= 3, name="c2")
    model.addConstr(-2 * x1 + x3 == 1, name="c3")

    model.setObjective(3 * x1 - x2 - x3, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.1 求解结果 ===")
        print(f"最优目标值 z: {model.ObjVal:.6g}")
        print(f"x1: {x1.X:.6g}")
        print(f"x2: {x2.X:.6g}")
        print(f"x3: {x3.X:.6g}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

