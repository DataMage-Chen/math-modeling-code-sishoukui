from gurobipy import GRB, Model


def solve(integer=False):
    model = Model("ex01_01")
    model.Params.OutputFlag = 0

    vtype = GRB.INTEGER if integer else GRB.CONTINUOUS
    x = model.addVar(lb=0.0, vtype=vtype, name="x")
    y = model.addVar(lb=0.0, vtype=vtype, name="y")

    model.addConstr(2 * x + y <= 10, name="A_hours")
    model.addConstr(x + y <= 8, name="B_hours")
    model.addConstr(y <= 7, name="C_hours")

    model.setObjective(4 * x + 3 * y, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print(f"Optimal profit: {model.ObjVal:.6g} (thousand yuan)")
        print(f"x (machine A): {x.X:.6g}")
        print(f"y (machine B): {y.X:.6g}")
    else:
        print(f"Optimization ended with status: {model.status}")


if __name__ == "__main__":
    solve(integer=False)
