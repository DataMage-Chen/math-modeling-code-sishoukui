"""
调用方法：
  python ch01/ex01_06_07/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("ex01_06_07")
    model.Params.OutputFlag = 0

    x = model.addVars(4, lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="x")
    u = model.addVars(4, lb=0.0, vtype=GRB.CONTINUOUS, name="u")

    for i in range(4):
        model.addConstr(x[i] <= u[i], name=f"abs_pos_{i}")
        model.addConstr(-x[i] <= u[i], name=f"abs_neg_{i}")

    model.addConstr(x[0] - x[1] - x[2] + x[3] <= -2, name="c1")
    model.addConstr(x[0] - x[1] + x[2] - 3 * x[3] <= -1, name="c2")
    model.addConstr(x[0] - x[1] - 2 * x[2] + 3 * x[3] <= -0.5, name="c3")

    weights = [1, 2, 3, 4]
    model.setObjective(
        sum(weights[i] * u[i] for i in range(4)), GRB.MINIMIZE
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Minimum objective: {model.ObjVal:.6g}")
        for i in range(4):
            print(f"x{i + 1}: {x[i].X:.6g}")
    else:
        print(f"Optimization ended with status: {model.status}")


if __name__ == "__main__":
    solve()
