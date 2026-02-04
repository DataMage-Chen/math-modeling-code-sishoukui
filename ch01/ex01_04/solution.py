"""
调用方法：
  python ch01/ex01_04/solution.py
"""

from gurobipy import GRB, Model


def solve():
    months = [1, 2, 3, 4]
    demand = {1: 15, 2: 10, 3: 20, 4: 12}  # unit: 100 m^2
    lengths = [1, 2, 3, 4]
    cost = {1: 2800, 2: 4500, 3: 6000, 4: 7300}  # yuan per 100 m^2 for full term

    model = Model("ex01_04")
    model.Params.OutputFlag = 0

    x = {}
    for t in months:
        for k in lengths:
            if t + k - 1 <= 4:
                x[t, k] = model.addVar(
                    lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{t}_{k}"
                )

    for m in months:
        model.addConstr(
            sum(x[t, k] for (t, k) in x if t <= m <= t + k - 1) >= demand[m],
            name=f"demand_{m}",
        )

    model.setObjective(
        sum(cost[k] * x[t, k] for (t, k) in x), GRB.MINIMIZE
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Minimum cost: {model.ObjVal:.6g} (yuan)")
        for t, k in sorted(x):
            if x[t, k].X > 1e-6:
                print(f"Start month {t}, length {k}: {x[t, k].X:.6g} (100 m^2)")
    else:
        print(f"Optimization ended with status: {model.status}")


if __name__ == "__main__":
    solve()
