"""
调用方法：
  python ch01/ex01_05/solution.py
"""

from gurobipy import GRB, Model


def solve():
    origins = ["A1", "A2", "A3", "A4", "A5", "A6"]
    destinations = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8"]

    supply = {"A1": 60, "A2": 55, "A3": 51, "A4": 43, "A5": 41, "A6": 52}
    demand = {"B1": 35, "B2": 37, "B3": 22, "B4": 32, "B5": 41, "B6": 32, "B7": 43, "B8": 38}

    cost = {
        "A1": {"B1": 6, "B2": 2, "B3": 6, "B4": 7, "B5": 4, "B6": 2, "B7": 5, "B8": 9},
        "A2": {"B1": 4, "B2": 9, "B3": 5, "B4": 3, "B5": 8, "B6": 5, "B7": 8, "B8": 2},
        "A3": {"B1": 5, "B2": 2, "B3": 1, "B4": 9, "B5": 7, "B6": 4, "B7": 3, "B8": 3},
        "A4": {"B1": 7, "B2": 6, "B3": 7, "B4": 3, "B5": 9, "B6": 2, "B7": 7, "B8": 1},
        "A5": {"B1": 2, "B2": 3, "B3": 9, "B4": 5, "B5": 7, "B6": 2, "B7": 6, "B8": 5},
        "A6": {"B1": 5, "B2": 5, "B3": 2, "B4": 2, "B5": 8, "B6": 1, "B7": 4, "B8": 3},
    }

    model = Model("ex01_05")
    model.Params.OutputFlag = 0

    x = {}
    for i in origins:
        for j in destinations:
            x[i, j] = model.addVar(lb=0.0, vtype=GRB.CONTINUOUS, name=f"x_{i}_{j}")

    for i in origins:
        model.addConstr(
            sum(x[i, j] for j in destinations) <= supply[i], name=f"supply_{i}"
        )

    for j in destinations:
        model.addConstr(
            sum(x[i, j] for i in origins) == demand[j], name=f"demand_{j}"
        )

    model.setObjective(
        sum(cost[i][j] * x[i, j] for i in origins for j in destinations),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Minimum cost: {model.ObjVal:.6g}")
        for i in origins:
            for j in destinations:
                if x[i, j].X > 1e-6:
                    print(f"{i} -> {j}: {x[i, j].X:.6g}")
    else:
        print(f"Optimization ended with status: {model.status}")


if __name__ == "__main__":
    solve()
