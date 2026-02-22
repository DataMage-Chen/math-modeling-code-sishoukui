"""
习题 2.9：运输优化。

运行：
  python ch02/hw02_09/solution.py
"""

from gurobipy import GRB, Model, quicksum


def build_data():
    # 成本矩阵：15 个用户 x 8 个配送中心
    costs = [
        [390.6, 618.5, 553.0, 442.0, 113.1, 5.2, 1217.7, 1011.0],
        [370.8, 636.0, 440.0, 401.8, 25.6, 113.1, 1172.4, 894.5],
        [876.3, 1098.6, 497.6, 779.8, 903.0, 1003.3, 907.2, 40.1],
        [745.4, 1037.0, 305.9, 725.7, 445.7, 531.4, 1376.4, 768.1],
        [144.5, 354.6, 624.7, 238.0, 290.7, 269.4, 993.2, 974.0],
        [200.2, 242.0, 691.5, 173.4, 560.0, 589.7, 661.8, 855.7],
        [235.0, 205.5, 801.5, 326.6, 477.0, 433.6, 966.4, 1112.0],
        [517.0, 541.5, 338.4, 219.0, 249.5, 335.0, 937.3, 701.8],
        [542.0, 321.0, 1104.0, 576.0, 896.8, 878.4, 728.3, 1243.0],
        [665.0, 827.0, 427.0, 523.2, 725.2, 813.8, 692.2, 284.0],
        [799.0, 855.1, 916.5, 709.3, 1057.0, 1115.5, 300.0, 617.0],
        [852.2, 798.0, 1083.0, 714.6, 1177.4, 1216.8, 40.8, 898.2],
        [602.0, 614.0, 820.0, 517.7, 899.6, 952.7, 272.4, 727.0],
        [903.0, 1092.5, 612.5, 790.0, 932.4, 1034.9, 777.0, 152.3],
        [600.7, 710.0, 522.0, 448.0, 726.6, 811.8, 563.0, 426.8],
    ]

    demands = [
        3000, 3100, 2900, 3100, 3100, 3400, 3500, 3200,
        3000, 3100, 3300, 3200, 3300, 2900, 3100,
    ]
    supplies = [18600, 19600, 17100, 18900, 17000, 19100, 20500, 17200]

    user_count = len(costs)
    center_count = len(costs[0])
    users = range(user_count)
    centers = range(center_count)

    if len(demands) != user_count:
        raise ValueError("需求量长度与用户数不一致。")
    if len(supplies) != center_count:
        raise ValueError("储备量长度与中心数不一致。")

    return users, centers, costs, demands, supplies


def extract_plan(x, users, centers):
    plan = []
    for i in users:
        for j in centers:
            v = x[i, j].X
            if v > 1e-6:
                plan.append((i, j, v))
    return plan


def print_plan(plan, title):
    print(title)
    by_user = {}
    for i, j, v in plan:
        by_user.setdefault(i, []).append((j, v))

    for i in sorted(by_user):
        items = ", ".join(f"中心{j + 1}={v:.1f}" for j, v in sorted(by_user[i]))
        print(f"  用户{i + 1}: {items}")


def solve_part1(users, centers, costs, demands, supplies):
    model = Model("hw02_09_part1")
    model.Params.OutputFlag = 0

    x = model.addVars(users, centers, lb=0.0, name="x")

    for i in users:
        model.addConstr(
            quicksum(x[i, j] for j in centers) == demands[i],
            name=f"demand_{i + 1}",
        )

    for j in centers:
        model.addConstr(
            quicksum(x[i, j] for i in users) <= supplies[j],
            name=f"supply_{j + 1}",
        )

    model.setObjective(
        quicksum(costs[i][j] * x[i, j] for i in users for j in centers),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    plan = extract_plan(x, users, centers)
    return {
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "plan": plan,
    }


def solve_part2(users, centers, costs, demands, supplies, low=1000.0, high=2000.0):
    model = Model("hw02_09_part2")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-8
    model.Params.MIPGapAbs = 1e-8

    x = model.addVars(users, centers, lb=0.0, name="x")
    y = model.addVars(users, centers, vtype=GRB.BINARY, name="y")

    for i in users:
        model.addConstr(
            quicksum(x[i, j] for j in centers) == demands[i],
            name=f"demand_{i + 1}",
        )

    for j in centers:
        model.addConstr(
            quicksum(x[i, j] for i in users) <= supplies[j],
            name=f"supply_{j + 1}",
        )

    for i in users:
        for j in centers:
            model.addConstr(x[i, j] <= high * y[i, j], name=f"upper_{i+1}_{j+1}")
            model.addConstr(x[i, j] >= low * y[i, j], name=f"lower_{i+1}_{j+1}")

    model.setObjective(
        quicksum(costs[i][j] * x[i, j] for i in users for j in centers),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    plan = extract_plan(x, users, centers)
    active_routes = sum(1 for i in users for j in centers if y[i, j].X > 0.5)
    return {
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
        "plan": plan,
        "active_routes": active_routes,
    }


def solve():
    users, centers, costs, demands, supplies = build_data()

    print("=== 习题 2.9 (1) 最小运费调配计划 ===")
    part1 = solve_part1(users, centers, costs, demands, supplies)
    if part1 is None:
        print("第(1)问未得到最优解。")
        return

    print(f"ObjVal: {part1['obj']:.10f}")
    print(f"ObjBound: {part1['obj_bound']:.10f}")
    print_plan(part1["plan"], "最优调配方案（非零配送）：")

    print("\n=== 习题 2.9 (2) 单条配送量在[1000,2000]时 ===")
    part2 = solve_part2(users, centers, costs, demands, supplies, low=1000.0, high=2000.0)
    if part2 is None:
        print("第(2)问未得到最优解。")
        return

    print(f"ObjVal: {part2['obj']:.10f}")
    print(f"ObjBound: {part2['obj_bound']:.10f}")
    print(f"MIPGap: {part2['mip_gap']:.3e}")
    print(f"启用配送关系数: {part2['active_routes']}")
    print_plan(part2["plan"], "最优调配方案（非零配送）：")


if __name__ == "__main__":
    solve()
