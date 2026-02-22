"""
习题 2.3：设备分配（边际利润口径，0-1 整数规划）。

建模口径：
1) c_{i,j} 表示“第 i 台设备分给企业 j 的利润”（边际利润）；
2) 每台设备恰好分配给一个企业；
3) 每个企业至少获得 1 台设备。

运行示例：
  python ch02/hw02_03/solution.py
  python ch02/hw02_03/solution.py --pool
  python ch02/hw02_03/solution.py --pool --pool-size 20
"""

import argparse

from gurobipy import GRB, Model, quicksum


def build_data():
    devices = [1, 2, 3, 4, 5, 6]
    enterprises = ["甲", "乙", "丙", "丁"]

    # 利润矩阵 c[i][j]：第 i 台设备分给企业 j 的利润（千万元）
    c = {
        1: {"甲": 4, "乙": 2, "丙": 3, "丁": 4},
        2: {"甲": 6, "乙": 4, "丙": 5, "丁": 5},
        3: {"甲": 7, "乙": 6, "丙": 7, "丁": 6},
        4: {"甲": 7, "乙": 8, "丙": 8, "丁": 6},
        5: {"甲": 7, "乙": 9, "丙": 8, "丁": 6},
        6: {"甲": 7, "乙": 10, "丙": 8, "丁": 6},
    }
    return devices, enterprises, c


def extract_solution(model, devices, enterprises, x, use_pool=False, solution_number=0):
    if use_pool:
        model.Params.SolutionNumber = solution_number
        get_value = lambda var: var.Xn
        objective = model.PoolObjVal
    else:
        get_value = lambda var: var.X
        objective = model.ObjVal

    assignment = {}
    for i in devices:
        for j in enterprises:
            if get_value(x[i, j]) > 0.5:
                assignment[i] = j
                break

    return {"objective": objective, "assignment": assignment}


def print_solution(title, solution, devices, enterprises, c):
    assignment = solution["assignment"]
    count_by_enterprise = {j: 0 for j in enterprises}
    profit_by_enterprise = {j: 0 for j in enterprises}
    total_profit = 0.0

    print(title)
    print("  设备分配：")
    for i in devices:
        j = assignment[i]
        value = c[i][j]
        count_by_enterprise[j] += 1
        profit_by_enterprise[j] += value
        total_profit += value
        print(f"    设备{i} -> {j}（利润 {value}）")

    print("  企业汇总：")
    for j in enterprises:
        print(
            f"    {j}: 设备数={count_by_enterprise[j]}, "
            f"利润小计={profit_by_enterprise[j]}"
        )

    print(f"  利润合计（校验）: {total_profit}")


def solve(show_pool=False, pool_size=10):
    devices, enterprises, c = build_data()

    model = Model("hw02_03_marginal_profit")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    if show_pool:
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = pool_size
        model.Params.PoolGap = 0

    x = model.addVars(devices, enterprises, vtype=GRB.BINARY, name="x")

    # 每台设备恰好分配给一个企业
    for i in devices:
        model.addConstr(quicksum(x[i, j] for j in enterprises) == 1, name=f"assign_{i}")

    # 每个企业至少获得一台设备
    for j in enterprises:
        model.addConstr(
            quicksum(x[i, j] for i in devices) >= 1,
            name=f"min_one_{j}",
        )

    model.setObjective(
        quicksum(c[i][j] * x[i, j] for i in devices for j in enterprises),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    print("=== 习题 2.3 求解结果（边际利润模型） ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最大总利润（千万元）: {model.ObjVal:.6f}")

    best = extract_solution(
        model,
        devices=devices,
        enterprises=enterprises,
        x=x,
        use_pool=False,
    )
    print_solution("一个最优分配方案：", best, devices, enterprises, c)

    if show_pool:
        print(f"\n已启用解池，最优解数量（受 pool-size 上限影响）: {model.SolCount}")
        for idx in range(model.SolCount):
            sol = extract_solution(
                model,
                devices=devices,
                enterprises=enterprises,
                x=x,
                use_pool=True,
                solution_number=idx,
            )
            print_solution(f"解池方案 #{idx + 1}：", sol, devices, enterprises, c)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 2.3（边际利润设备分配）。")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="输出多个最优解（解池模式）。",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="解池最多保留的解个数（默认 10）。",
    )
    args = parser.parse_args()

    if args.pool_size <= 0:
        raise ValueError("pool-size 必须为正整数。")

    solve(show_pool=args.pool, pool_size=args.pool_size)
