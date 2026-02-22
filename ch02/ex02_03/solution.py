"""
例题 2.3：旅行商问题（TSP）。

参数说明：
- --n：城市数量 n（默认 6，建议 n>=3）。
- --seed：随机种子（默认 42），用于生成参数化示例旅费矩阵。

运行：
  python ch02/ex02_03/solution.py
  python ch02/ex02_03/solution.py --n 10 --seed 2026
"""

import argparse
import random

from gurobipy import GRB, Model, quicksum


def build_example_data(city_count, seed):
    # 生成一组对称旅费矩阵（主对角线为 0）
    random_generator = random.Random(seed)
    cities = [f"v{i + 1}" for i in range(city_count)]
    cost = [[0 for _ in range(city_count)] for _ in range(city_count)]

    for i in range(city_count):
        for j in range(i + 1, city_count):
            value = random_generator.randint(8, 45)
            cost[i][j] = value
            cost[j][i] = value
    return cities, cost


def print_cost_matrix(cities, cost):
    print("旅费矩阵 c_ij：")
    header = "      " + " ".join(f"{name:>6}" for name in cities)
    print(header)
    for i, city in enumerate(cities):
        row = " ".join(f"{cost[i][j]:>6}" for j in range(len(cities)))
        print(f"{city:>6} {row}")


def solve(city_count=6, seed=42):
    # 原题未给具体数值，这里按 n 和 seed 生成一组参数化示例数据
    cities, cost = build_example_data(city_count=city_count, seed=seed)
    node_ids = range(city_count)

    print("=== 例题 2.3 输入数据 ===")
    print(f"城市数量 n: {city_count}")
    print(f"随机种子 seed: {seed}")
    print_cost_matrix(cities, cost)

    model = Model("ex02_03_tsp")
    model.Params.OutputFlag = 0

    x = model.addVars(
        [(i, j) for i in node_ids for j in node_ids if i != j],
        vtype=GRB.BINARY,
        name="x",
    )

    # MTZ 顺序变量：仅对起点 v1 之外的城市定义
    u = model.addVars(range(1, city_count), lb=1.0, ub=city_count - 1, name="u")

    for i in node_ids:
        model.addConstr(
            quicksum(x[i, j] for j in node_ids if j != i) == 1, name=f"out_{i}"
        )
        model.addConstr(
            quicksum(x[j, i] for j in node_ids if j != i) == 1, name=f"in_{i}"
        )

    for i in range(1, city_count):
        for j in range(1, city_count):
            if i != j:
                model.addConstr(
                    u[i] - u[j] + (city_count - 1) * x[i, j] <= city_count - 2,
                    name=f"mtz_{i}_{j}",
                )

    model.setObjective(
        quicksum(cost[i][j] * x[i, j] for i in node_ids for j in node_ids if i != j),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        selected_arcs = [
            (i, j) for i in node_ids for j in node_ids if i != j and x[i, j].X > 0.5
        ]
        next_city = {i: j for i, j in selected_arcs}

        route = [0]
        current = 0
        for _ in range(city_count - 1):
            current = next_city[current]
            route.append(current)
        route.append(0)

        route_name = " -> ".join(cities[idx] for idx in route)
        route_cost = sum(cost[route[k]][route[k + 1]] for k in range(len(route) - 1))

        print("\n=== 例题 2.3 求解结果 ===")
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
        print(f"最优路线: {route_name}")
        print(f"总旅费校验: {route_cost:.10f}")
        print("选中弧：")
        for i, j in selected_arcs:
            print(f"  {cities[i]} -> {cities[j]}，费用 = {cost[i][j]}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解例题 2.3（旅行商问题）。")
    parser.add_argument(
        "--n",
        type=int,
        default=6,
        help="城市数量 n（默认 6，建议 n>=3）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）。",
    )
    args = parser.parse_args()

    if args.n < 3:
        raise ValueError("n 必须不小于 3。")

    solve(city_count=args.n, seed=args.seed)

