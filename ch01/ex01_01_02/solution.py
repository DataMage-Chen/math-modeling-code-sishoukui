"""
调用方法：
  python ch01/ex01_01/solution.py
默认变量为连续变量；若要求整数，则设置 integer=True，例如：
  python ch01/ex01_01/solution.py --integer=True
本题两者结果一样。
"""

import argparse

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
    def _parse_bool(value):
        if isinstance(value, bool):
            return value
        value = str(value).strip().lower()
        if value in {"1", "true", "t", "yes", "y"}:
            return True
        if value in {"0", "false", "f", "no", "n"}:
            return False
        raise argparse.ArgumentTypeError(
            "integer must be a boolean (e.g., True/False, 1/0, yes/no)"
        )

    parser = argparse.ArgumentParser(description="Solve ex01_01 with Gurobi.")
    parser.add_argument(
        "--integer",
        type=_parse_bool,
        default=False,
        help="Use integer variables (default: False for continuous).",
    )
    args = parser.parse_args()
    solve(integer=args.integer)
