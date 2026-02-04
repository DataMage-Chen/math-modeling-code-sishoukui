"""
调用方法：
  python ch01/ex01_03/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("ex01_03")
    model.Params.OutputFlag = 0

    x1 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x1")
    x2 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x2")
    x3 = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name="x3")

    model.addConstr(x1 + x2 + x3 == 7, name="eq")
    model.addConstr(2 * x1 - 5 * x2 + x3 >= 10, name="geq")
    model.addConstr(x1 + 3 * x2 + x3 <= 12, name="leq")

    model.setObjective(2 * x1 + 3 * x2 - 5 * x3, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal objective: {model.ObjVal:.6g}")
        print(f"x1: {x1.X:.6g}")
        print(f"x2: {x2.X:.6g}")
        print(f"x3: {x3.X:.6g}")
    else:
        print(f"Optimization ended with status: {model.status}")


if __name__ == "__main__":
    solve()
