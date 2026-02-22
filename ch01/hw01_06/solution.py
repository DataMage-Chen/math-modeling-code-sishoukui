"""
习题 1.6：糖果配方与原料分配优化（线性规划）。

运行：
  python ch01/hw01_06/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("hw01_06")
    model.Params.OutputFlag = 0

    raw_types = ["A", "B", "C"]
    candy_types = ["M", "F"]  # M: 高级奶糖, F: 水果糖

    raw_supply = {"A": 500.0, "B": 750.0, "C": 625.0}
    raw_cost = {"A": 20.0, "B": 12.0, "C": 8.0}
    candy_price = {"M": 24.0, "F": 15.0}

    x = model.addVars(raw_types, candy_types, lb=0.0, name="x")
    y = model.addVars(candy_types, lb=0.0, name="y")

    model.addConstr(y["M"] == x["A", "M"] + x["B", "M"] + x["C", "M"], name="def_M")
    model.addConstr(y["F"] == x["A", "F"] + x["B", "F"] + x["C", "F"], name="def_F")

    model.addConstr(x["A", "M"] >= 0.50 * y["M"], name="M_A")
    model.addConstr(x["B", "M"] >= 0.25 * y["M"], name="M_B")
    model.addConstr(x["C", "M"] <= 0.10 * y["M"], name="M_C")

    model.addConstr(x["A", "F"] <= 0.40 * y["F"], name="F_A")
    model.addConstr(x["B", "F"] <= 0.40 * y["F"], name="F_B")
    model.addConstr(x["C", "F"] >= 0.15 * y["F"], name="F_C")

    for raw in raw_types:
        model.addConstr(x[raw, "M"] + x[raw, "F"] <= raw_supply[raw], name=f"supply_{raw}")

    model.addConstr(y["M"] >= 600.0, name="order_M")
    model.addConstr(y["F"] >= 800.0, name="order_F")

    revenue = candy_price["M"] * y["M"] + candy_price["F"] * y["F"]
    cost = sum(raw_cost[raw] * (x[raw, "M"] + x[raw, "F"]) for raw in raw_types)
    model.setObjective(revenue - cost, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.6 求解结果 ===")
        print(f"最优利润（元）: {model.ObjVal:.10f}")
        print("产量（kg）：")
        print(f"  高级奶糖 y_M = {y['M'].X:.6g}")
        print(f"  水果糖   y_F = {y['F'].X:.6g}")

        print("原料分配（kg）：")
        for raw in raw_types:
            print(
                f"  原料{raw}: "
                f"用于奶糖={x[raw, 'M'].X:.6g}, "
                f"用于水果糖={x[raw, 'F'].X:.6g}, "
                f"合计={x[raw, 'M'].X + x[raw, 'F'].X:.6g}"
            )

        print("配比校验：")
        ratio_am = x["A", "M"].X / y["M"].X if y["M"].X > 0 else 0.0
        ratio_bm = x["B", "M"].X / y["M"].X if y["M"].X > 0 else 0.0
        ratio_cm = x["C", "M"].X / y["M"].X if y["M"].X > 0 else 0.0
        ratio_af = x["A", "F"].X / y["F"].X if y["F"].X > 0 else 0.0
        ratio_bf = x["B", "F"].X / y["F"].X if y["F"].X > 0 else 0.0
        ratio_cf = x["C", "F"].X / y["F"].X if y["F"].X > 0 else 0.0
        print(
            f"  奶糖: A={ratio_am:.4%}, B={ratio_bm:.4%}, C={ratio_cm:.4%}"
            "（要求 A>=50%, B>=25%, C<=10%）"
        )
        print(
            f"  水果糖: A={ratio_af:.4%}, B={ratio_bf:.4%}, C={ratio_cf:.4%}"
            "（要求 A<=40%, B<=40%, C>=15%）"
        )

        revenue_val = revenue.getValue()
        cost_val = cost.getValue()
        print(f"收入（元）: {revenue_val:.10f}")
        print(f"成本（元）: {cost_val:.10f}")
        print(f"利润（元）: {revenue_val - cost_val:.10f}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

