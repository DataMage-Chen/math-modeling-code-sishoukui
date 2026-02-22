"""
习题 2.1：先将非线性 0-1 规划线性化，再求解。

原问题：
  max z = x1 + x1*x2 - x3
  s.t. -2x1 + 3x2 + x3 <= 3
       x1, x2, x3 ∈ {0,1}

线性化：
  令 y = x1*x2（y 也取 0-1），则
  max z = x1 + y - x3
  s.t. y <= x1
       y <= x2
       y >= x1 + x2 - 1
       其余约束不变。

运行：
  python ch02/hw02_01/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("hw02_01")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x1 = model.addVar(vtype=GRB.BINARY, name="x1")
    x2 = model.addVar(vtype=GRB.BINARY, name="x2")
    x3 = model.addVar(vtype=GRB.BINARY, name="x3")
    y = model.addVar(vtype=GRB.BINARY, name="y")

    model.addConstr(-2 * x1 + 3 * x2 + x3 <= 3, name="c1")
    model.addConstr(y <= x1, name="lin1")
    model.addConstr(y <= x2, name="lin2")
    model.addConstr(y >= x1 + x2 - 1, name="lin3")

    model.setObjective(x1 + y - x3, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_val = int(round(x1.X))
    x2_val = int(round(x2.X))
    x3_val = int(round(x3.X))
    y_val = int(round(y.X))

    nonlinear_obj = x1_val + x1_val * x2_val - x3_val

    print("=== 习题 2.1 求解结果（线性化后 MILP） ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print("最优变量：")
    print(f"  x1 = {x1_val}")
    print(f"  x2 = {x2_val}")
    print(f"  x3 = {x3_val}")
    print(f"  y  = {y_val}")
    print("校验：")
    print(f"  x1*x2 = {x1_val * x2_val}")
    print(f"  y 与 x1*x2 是否一致: {y_val == x1_val * x2_val}")
    print(f"  代回原非线性目标 z = x1 + x1*x2 - x3 = {nonlinear_obj}")


if __name__ == "__main__":
    solve()
