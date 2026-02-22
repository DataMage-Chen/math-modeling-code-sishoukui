"""
例题 2.1：0-1 背包问题。

运行：
  python ch02/ex02_01/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    # 原题未给具体数据，这里先定义一组示例数据
    items = [
        {"name": "地图", "weight": 1.0, "value": 8.0},
        {"name": "水壶", "weight": 2.5, "value": 15.0},
        {"name": "急救包", "weight": 1.5, "value": 14.0},
        {"name": "外套", "weight": 3.0, "value": 12.0},
        {"name": "手电筒", "weight": 1.2, "value": 7.0},
        {"name": "食物包A", "weight": 2.8, "value": 16.0},
        {"name": "食物包B", "weight": 3.6, "value": 20.0},
        {"name": "相机", "weight": 2.0, "value": 11.0},
        {"name": "备用电池", "weight": 0.8, "value": 6.0},
        {"name": "睡袋", "weight": 4.0, "value": 18.0},
    ]
    capacity = 12.0

    item_count = len(items)

    model = Model("ex02_01_knapsack")
    model.Params.OutputFlag = 0

    x = model.addVars(item_count, vtype=GRB.BINARY, name="x")

    model.addConstr(
        quicksum(items[i]["weight"] * x[i] for i in range(item_count)) <= capacity,
        name="capacity",
    )
    model.setObjective(
        quicksum(items[i]["value"] * x[i] for i in range(item_count)),
        GRB.MAXIMIZE,
    )

    model.optimize()

    if model.status == GRB.OPTIMAL:
        total_weight = sum(items[i]["weight"] * x[i].X for i in range(item_count))
        total_value = sum(items[i]["value"] * x[i].X for i in range(item_count))
        selected = [items[i]["name"] for i in range(item_count) if x[i].X > 0.5]

        print("=== 例题 2.1 求解结果 ===")
        print(f"背包容量 b: {capacity:.6g} kg")
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
        print(f"总质量: {total_weight:.6g} kg")
        print(f"总价值: {total_value:.6g}")
        print("入选物品：")
        for name in selected:
            print(f"  - {name}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

