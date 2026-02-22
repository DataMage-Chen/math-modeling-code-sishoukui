"""
习题 1.2：带绝对值目标的线性规划。

参数说明：
- --linear：强制使用绝对值线性化约束；
            不加时优先尝试 Gurobi 的绝对值一般约束。

运行示例：
- python ch01/hw01_02/solution.py
- python ch01/hw01_02/solution.py --linear
"""

import argparse

from gurobipy import GRB, Model


def solve(use_genconstr=True):
    model = Model("hw01_02")
    model.Params.OutputFlag = 0

    x = model.addVars(4, lb=-GRB.INFINITY, name="x")
    u = model.addVars(4, lb=0.0, name="u")

    can_use_gen = hasattr(model, "addGenConstrAbs")
    use_gen = use_genconstr and can_use_gen

    if use_gen:
        for i in range(4):
            model.addGenConstrAbs(u[i], x[i], name=f"abs_{i + 1}")
    else:
        for i in range(4):
            model.addConstr(x[i] <= u[i], name=f"abs_pos_{i + 1}")
            model.addConstr(-x[i] <= u[i], name=f"abs_neg_{i + 1}")

    model.addConstr(x[0] - x[1] - x[2] + x[3] == 0, name="c1")
    model.addConstr(x[0] - x[1] + x[2] - 3 * x[3] == 1, name="c2")
    model.addConstr(x[0] - x[1] - 2 * x[2] + 3 * x[3] == -0.5, name="c3")

    weights = [1, 2, 3, 4]
    model.setObjective(sum(weights[i] * u[i] for i in range(4)), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.2 求解结果 ===")
        print(f"最优目标值 z: {model.ObjVal:.6g}")
        for i in range(4):
            print(f"x{i + 1}: {x[i].X:.6g}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 1.2。")
    parser.add_argument(
        "--linear",
        action="store_true",
        help="强制使用绝对值线性化约束。",
    )
    args = parser.parse_args()
    solve(use_genconstr=not args.linear)

