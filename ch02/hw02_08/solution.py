"""
习题 2.8：标准指派问题（0-1 线性规划）。

运行：
  python ch02/hw02_08/solution.py
"""

from gurobipy import GRB, Model, quicksum


def build_data():
    cost = [
        [6, 7, 5, 8, 9, 10],
        [6, 3, 7, 9, 3, 8],
        [8, 11, 12, 6, 7, 9],
        [9, 7, 5, 4, 7, 6],
        [5, 8, 9, 6, 10, 7],
        [9, 8, 7, 6, 5, 9],
    ]
    n = len(cost)
    return cost, n


def solve():
    cost, n = build_data()

    model = Model("hw02_08_assignment")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")

    for i in range(n):
        model.addConstr(quicksum(x[i, j] for j in range(n)) == 1, name=f"row_{i+1}")
    for j in range(n):
        model.addConstr(quicksum(x[i, j] for i in range(n)) == 1, name=f"col_{j+1}")

    model.setObjective(
        quicksum(cost[i][j] * x[i, j] for i in range(n) for j in range(n)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    assignment = []
    total = 0.0
    for i in range(n):
        for j in range(n):
            if x[i, j].X > 0.5:
                assignment.append((i + 1, j + 1, cost[i][j]))
                total += cost[i][j]
                break

    print("=== 习题 2.8 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最小总成本: {model.ObjVal:.6f}")
    print("最优指派方案（执行者 -> 任务, 成本）：")
    for i, j, c in assignment:
        print(f"  {i} -> {j}, 成本 {c}")
    print(f"成本合计（校验）: {total:.6f}")


if __name__ == "__main__":
    solve()
