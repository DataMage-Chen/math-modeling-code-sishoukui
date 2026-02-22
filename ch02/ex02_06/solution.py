"""
例题 2.6：带容量约束的门店装修任务分配（整数规划）。

运行：
  python ch02/ex02_06/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    companies = ["A", "B", "C", "D"]
    stores = [1, 2, 3, 4, 5]

    # 装修费用（单位：万元），cost[i][j] 表示公司 i 装修门店 j 的费用
    cost = {
        ("A", 1): 15.0,
        ("A", 2): 13.8,
        ("A", 3): 12.5,
        ("A", 4): 11.0,
        ("A", 5): 14.3,
        ("B", 1): 14.5,
        ("B", 2): 14.0,
        ("B", 3): 13.2,
        ("B", 4): 10.5,
        ("B", 5): 15.0,
        ("C", 1): 13.8,
        ("C", 2): 13.0,
        ("C", 3): 12.8,
        ("C", 4): 11.3,
        ("C", 5): 14.6,
        ("D", 1): 14.7,
        ("D", 2): 13.6,
        ("D", 3): 13.0,
        ("D", 4): 11.6,
        ("D", 5): 14.0,
    }

    model = Model("ex02_06_assignment_capacity")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x = model.addVars(companies, stores, vtype=GRB.BINARY, name="x")

    for store in stores:
        model.addConstr(
            quicksum(x[company, store] for company in companies) == 1,
            name=f"assign_store_{store}",
        )

    for company in companies:
        model.addConstr(
            quicksum(x[company, store] for store in stores) <= 2,
            name=f"cap_{company}",
        )

    model.setObjective(
        quicksum(cost[company, store] * x[company, store] for company in companies for store in stores),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 例题 2.6 求解结果 ===")
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")

        print("门店分配方案：")
        total_cost = 0.0
        for store in stores:
            for company in companies:
                if x[company, store].X > 0.5:
                    current_cost = cost[company, store]
                    total_cost += current_cost
                    print(f"  门店{store} -> 公司{company}，费用 = {current_cost:.6g} 万元")

        print("公司任务量统计：")
        for company in companies:
            load = sum(1 for store in stores if x[company, store].X > 0.5)
            print(f"  公司{company}: {load} 个门店")

        print(f"总费用校验: {total_cost:.10f} 万元")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

