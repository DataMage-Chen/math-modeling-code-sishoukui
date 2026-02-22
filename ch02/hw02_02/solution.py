"""
习题 2.2：校址选择（集合覆盖 0-1 规划）。

运行：
  python ch02/hw02_02/solution.py
"""

from gurobipy import GRB, Model, quicksum


def build_data():
    zones = [f"A{i}" for i in range(1, 9)]
    sites = [f"B{j}" for j in range(1, 7)]
    site_cover = {
        "B1": {"A1", "A5", "A7"},
        "B2": {"A1", "A2", "A5", "A8"},
        "B3": {"A1", "A3", "A5"},
        "B4": {"A2", "A4", "A8"},
        "B5": {"A3", "A6"},
        "B6": {"A4", "A6", "A8"},
    }

    zone_to_sites = {
        zone: [site for site in sites if zone in site_cover[site]] for zone in zones
    }
    return zones, sites, site_cover, zone_to_sites


def solve():
    zones, sites, site_cover, zone_to_sites = build_data()

    model = Model("hw02_02_set_cover")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    y = model.addVars(sites, vtype=GRB.BINARY, name="y")

    for zone in zones:
        cover_sites = zone_to_sites[zone]
        model.addConstr(
            quicksum(y[site] for site in cover_sites) >= 1,
            name=f"cover_{zone}",
        )

    model.setObjective(quicksum(y[site] for site in sites), GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    selected_sites = [site for site in sites if y[site].X > 0.5]

    print("=== 习题 2.2 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最少建校数量: {len(selected_sites)}")
    print("选址方案:")
    for site in selected_sites:
        covered = ", ".join(sorted(site_cover[site]))
        print(f"  {site}（覆盖: {covered}）")

    print("覆盖校验：")
    for zone in zones:
        served_by = [site for site in selected_sites if zone in site_cover[site]]
        print(f"  {zone}: {', '.join(served_by)}")


if __name__ == "__main__":
    solve()
