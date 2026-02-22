"""
习题 1.3：两道工序、多设备、多产品的生产计划整数规划（MIP）。

运行：
  python ch01/hw01_03/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    model = Model("hw01_03")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    machine_hours = {
        "A1": {("A", "I"): 5, ("A", "II"): 10},
        "A2": {("A", "I"): 7, ("A", "II"): 9, ("A", "III"): 12},
        "B1": {("B", "I"): 6, ("B", "II"): 8},
        "B2": {("B", "I"): 4, ("B", "III"): 11},
        "B3": {("B", "I"): 7},
    }
    capacity = {"A1": 6000, "A2": 10000, "B1": 4000, "B2": 7000, "B3": 4000}
    full_load_cost = {"A1": 300, "A2": 321, "B1": 250, "B2": 783, "B3": 200}
    raw_cost = {"I": 0.25, "II": 0.35, "III": 0.50}
    sale_price = {"I": 1.25, "II": 2.00, "III": 2.80}

    time_cost_rate = {m: full_load_cost[m] / capacity[m] for m in capacity}

    x = {}
    for machine, entries in machine_hours.items():
        for process, product in entries:
            x[(process, machine, product)] = model.addVar(
                lb=0.0,
                vtype=GRB.INTEGER,
                name=f"x_{process}_{machine}_{product}",
            )

    q = {
        product: model.addVar(lb=0.0, vtype=GRB.INTEGER, name=f"q_{product}")
        for product in raw_cost
    }

    model.addConstr(
        q["I"] == x[("A", "A1", "I")] + x[("A", "A2", "I")], name="flow_I_A"
    )
    model.addConstr(
        q["I"] == x[("B", "B1", "I")] + x[("B", "B2", "I")] + x[("B", "B3", "I")],
        name="flow_I_B",
    )

    model.addConstr(
        q["II"] == x[("A", "A1", "II")] + x[("A", "A2", "II")], name="flow_II_A"
    )
    model.addConstr(q["II"] == x[("B", "B1", "II")], name="flow_II_B")

    model.addConstr(q["III"] == x[("A", "A2", "III")], name="flow_III_A")
    model.addConstr(q["III"] == x[("B", "B2", "III")], name="flow_III_B")

    for machine, entries in machine_hours.items():
        model.addConstr(
            quicksum(
                entries[(process, product)] * x[(process, machine, product)]
                for process, product in entries
            )
            <= capacity[machine],
            name=f"cap_{machine}",
        )

    revenue_minus_raw = quicksum(
        (sale_price[product] - raw_cost[product]) * q[product] for product in q
    )
    equipment_cost = quicksum(
        time_cost_rate[machine]
        * quicksum(
            machine_hours[machine][(process, product)]
            * x[(process, machine, product)]
            for process, product in machine_hours[machine]
        )
        for machine in machine_hours
    )

    model.setObjective(revenue_minus_raw - equipment_cost, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.3 求解结果（整数规划） ===")
        print(f"最优利润: {model.ObjVal:.10f}")
        print(f"最优界: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
        print("产品产量：")
        for product in ["I", "II", "III"]:
            print(f"  q_{product}: {q[product].X:.6g}")

        print("工序分配：")
        for key in sorted(x):
            process, machine, product = key
            print(f"  {process}-{machine}-{product}: {x[key].X:.6g}")

        print("设备利用：")
        for machine, entries in machine_hours.items():
            used = sum(
                entries[(process, product)] * x[(process, machine, product)].X
                for process, product in entries
            )
            utilization = used / capacity[machine] if capacity[machine] > 0 else 0.0
            print(
                f"  {machine}: 使用 {used:.6g} / {capacity[machine]:.6g}, 利用率 {utilization:.2%}"
            )
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()
