"""
习题 1.4：货机货舱装载优化（线性规划）。

运行：
  python ch01/hw01_04/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    model = Model("hw01_04")
    model.Params.OutputFlag = 0

    compartments = ["front", "middle", "rear"]
    cargo_types = [1, 2, 3, 4]

    weight_cap = {"front": 10.0, "middle": 16.0, "rear": 8.0}
    volume_cap = {"front": 6800.0, "middle": 8700.0, "rear": 5300.0}

    cargo_supply = {1: 18.0, 2: 15.0, 3: 23.0, 4: 12.0}
    unit_volume = {1: 480.0, 2: 650.0, 3: 580.0, 4: 390.0}
    unit_profit = {1: 3100.0, 2: 3800.0, 3: 3500.0, 4: 2850.0}

    x = model.addVars(compartments, cargo_types, lb=0.0, name="x")

    compartment_weight = {
        comp: quicksum(x[comp, cargo] for cargo in cargo_types)
        for comp in compartments
    }

    for comp in compartments:
        model.addConstr(compartment_weight[comp] <= weight_cap[comp], name=f"wcap_{comp}")
        model.addConstr(
            quicksum(unit_volume[cargo] * x[comp, cargo] for cargo in cargo_types)
            <= volume_cap[comp],
            name=f"vcap_{comp}",
        )

    for cargo in cargo_types:
        model.addConstr(
            quicksum(x[comp, cargo] for comp in compartments) <= cargo_supply[cargo],
            name=f"supply_{cargo}",
        )

    model.addConstr(
        16 * compartment_weight["front"] - 10 * compartment_weight["middle"] == 0,
        name="balance_f_m",
    )
    model.addConstr(
        8 * compartment_weight["middle"] - 16 * compartment_weight["rear"] == 0,
        name="balance_m_r",
    )

    model.setObjective(
        quicksum(
            unit_profit[cargo] * x[comp, cargo]
            for comp in compartments
            for cargo in cargo_types
        ),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.4 求解结果 ===")
        print(f"最优利润: {model.ObjVal:.6g}")

        comp_cn = {"front": "前舱", "middle": "中舱", "rear": "后舱"}
        print("装载方案（吨）：")
        for comp in compartments:
            values = [x[comp, cargo].X for cargo in cargo_types]
            values_text = ", ".join(
                f"货物{cargo}={values[idx]:.6g}"
                for idx, cargo in enumerate(cargo_types)
            )
            print(f"  {comp_cn[comp]}: {values_text}")

        print("货舱利用情况：")
        for comp in compartments:
            used_weight = sum(x[comp, cargo].X for cargo in cargo_types)
            used_volume = sum(
                unit_volume[cargo] * x[comp, cargo].X for cargo in cargo_types
            )
            print(
                f"  {comp_cn[comp]}: "
                f"质量 {used_weight:.6g}/{weight_cap[comp]:.6g}, "
                f"体积 {used_volume:.6g}/{volume_cap[comp]:.6g}"
            )

        print("平衡比例校验（应接近相等）：")
        ratio_front = compartment_weight["front"].getValue() / weight_cap["front"]
        ratio_middle = compartment_weight["middle"].getValue() / weight_cap["middle"]
        ratio_rear = compartment_weight["rear"].getValue() / weight_cap["rear"]
        print(
            f"  前舱={ratio_front:.8f}, 中舱={ratio_middle:.8f}, 后舱={ratio_rear:.8f}"
        )
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

