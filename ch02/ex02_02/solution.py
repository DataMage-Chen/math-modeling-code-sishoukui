"""
例题 2.2：标准指派问题（最小费用）。

运行：
  python ch02/ex02_02/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    # 原题未给具体数值，这里先定义一组示例成本矩阵 c_ij
    cost_matrix = [
        [9, 11, 14, 11, 7],
        [6, 15, 13, 13, 10],
        [12, 13, 6, 8, 8],
        [11, 9, 10, 12, 9],
        [7, 12, 14, 10, 14],
    ]

    people = ["人员1", "人员2", "人员3", "人员4", "人员5"]
    tasks = ["任务1", "任务2", "任务3", "任务4", "任务5"]
    size = len(cost_matrix)

    model = Model("ex02_02_assignment")
    model.Params.OutputFlag = 0

    x = model.addVars(size, size, vtype=GRB.BINARY, name="x")

    for i in range(size):
        model.addConstr(quicksum(x[i, j] for j in range(size)) == 1, name=f"person_{i}")
    for j in range(size):
        model.addConstr(quicksum(x[i, j] for i in range(size)) == 1, name=f"task_{j}")

    model.setObjective(
        quicksum(cost_matrix[i][j] * x[i, j] for i in range(size) for j in range(size)),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 例题 2.2 求解结果 ===")
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
        print("最优指派方案：")

        total_cost = 0.0
        for i in range(size):
            for j in range(size):
                if x[i, j].X > 0.5:
                    cost = cost_matrix[i][j]
                    total_cost += cost
                    print(f"  {people[i]} -> {tasks[j]}，费用 = {cost}")

        print(f"总费用校验: {total_cost:.10f}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

