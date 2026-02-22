"""
例题 2.7：容量约束下的供应站选址覆盖问题（整数规划）。

参数说明：
- --pool：开启多解模式，尝试输出多个最优解。
- --pool-size：多解模式下最多保留的解数量（默认 10）。

运行：
  python ch02/ex02_07/solution.py
  python ch02/ex02_07/solution.py --pool
  python ch02/ex02_07/solution.py --pool --pool-size 20
"""

import argparse
import math

from gurobipy import GRB, Model, quicksum


def build_problem_data():
    # 题目给定的 10 个商业网点坐标
    xs = [9.4888, 8.7928, 11.5960, 11.5643, 5.6756, 9.8497, 9.1756, 13.1385, 15.4663, 15.5464]
    ys = [5.6817, 10.3868, 3.9294, 4.4325, 9.9658, 17.6632, 6.1517, 11.8569, 8.8721, 15.5868]
    radius_limit = 10.0
    capacity_limit = 5

    node_count = len(xs)
    nodes = range(node_count)

    distance = {}
    for i in nodes:
        for j in nodes:
            distance[i, j] = math.hypot(xs[i] - xs[j], ys[i] - ys[j])

    feasible_pairs = [(i, j) for i in nodes for j in nodes if distance[i, j] <= radius_limit]

    data = {
        "xs": xs,
        "ys": ys,
        "nodes": nodes,
        "distance": distance,
        "feasible_pairs": feasible_pairs,
        "radius_limit": radius_limit,
        "capacity_limit": capacity_limit,
    }
    return data


def extract_solution(model, data, x, y, use_pool=False, solution_number=0):
    nodes = data["nodes"]
    distance = data["distance"]

    if use_pool:
        model.Params.SolutionNumber = solution_number
        read_value = lambda var: var.Xn
        objective = model.PoolObjVal
    else:
        read_value = lambda var: var.X
        objective = model.ObjVal

    open_stations = [j for j in nodes if read_value(y[j]) > 0.5]

    assignments = {}
    for i in nodes:
        for j in nodes:
            if (i, j) in x and read_value(x[i, j]) > 0.5:
                assignments[i] = j
                break

    max_distance = max(distance[i, assignments[i]] for i in nodes)
    station_load = {
        j: sum(1 for i in nodes if assignments[i] == j) for j in open_stations
    }

    solution = {
        "objective": objective,
        "open_stations": open_stations,
        "assignments": assignments,
        "station_load": station_load,
        "max_distance": max_distance,
    }
    return solution


def print_solution(solution, data, title):
    xs = data["xs"]
    ys = data["ys"]
    nodes = data["nodes"]
    distance = data["distance"]

    print(title)
    print(f"  供应站数量: {len(solution['open_stations'])}")
    print(f"  目标值(站点数): {solution['objective']:.10f}")
    print(f"  最大服务距离: {solution['max_distance']:.4f} km")
    print("  设站网点（编号从 1 开始）：")
    for j in solution["open_stations"]:
        print(f"    网点{j + 1} (x={xs[j]:.4f}, y={ys[j]:.4f})")

    print("  网点分配方案：")
    for i in nodes:
        j = solution["assignments"][i]
        print(
            f"    网点{i + 1} -> 供应站{j + 1}，"
            f"距离 = {distance[i, j]:.4f} km"
        )

    print("  供应站负载统计：")
    for j in solution["open_stations"]:
        print(f"    供应站{j + 1}: {solution['station_load'][j]} 个网点")


def solve(use_pool=False, pool_size=10):
    data = build_problem_data()
    nodes = data["nodes"]
    feasible_pairs = data["feasible_pairs"]
    capacity_limit = data["capacity_limit"]

    model = Model("ex02_07_location_cover")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    if use_pool:
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = pool_size
        model.Params.PoolGap = 0

    y = model.addVars(nodes, vtype=GRB.BINARY, name="y")
    x = model.addVars(feasible_pairs, vtype=GRB.BINARY, name="x")

    for i in nodes:
        candidate_stations = [j for j in nodes if (i, j) in x]
        model.addConstr(
            quicksum(x[i, j] for j in candidate_stations) == 1,
            name=f"assign_{i + 1}",
        )

    for j in nodes:
        serviced_nodes = [i for i in nodes if (i, j) in x]
        model.addConstr(
            quicksum(x[i, j] for i in serviced_nodes) <= capacity_limit * y[j],
            name=f"cap_{j + 1}",
        )

    model.setObjective(quicksum(y[j] for j in nodes), GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    print("=== 例题 2.7 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")

    best_solution = extract_solution(model, data, x, y, use_pool=False)
    print_solution(best_solution, data, title="最优方案（默认输出）：")

    if use_pool:
        print(f"\n多解模式已开启，解池中解数量: {model.SolCount}")
        print(f"解池上限 PoolSolutions: {pool_size}")

        if model.SolCount > 1:
            for sol_idx in range(model.SolCount):
                solution = extract_solution(
                    model, data, x, y, use_pool=True, solution_number=sol_idx
                )
                print()
                print_solution(solution, data, title=f"解池方案 #{sol_idx + 1}:")
        else:
            print("当前实例只找到 1 个最优方案。")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解例题 2.7（容量约束选址覆盖）。")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="开启多解模式，尝试输出多个最优解。",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="多解模式下最多保留的解数量（默认 10）。",
    )
    args = parser.parse_args()

    if args.pool_size <= 0:
        raise ValueError("pool-size 必须为正整数。")

    solve(use_pool=args.pool, pool_size=args.pool_size)

