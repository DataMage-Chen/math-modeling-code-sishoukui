"""
习题 2.5：钢条下料优化（Cutting Stock 整数规划）。

问题：
  - 需求：2.9m、2.1m、1.0m 零件各 100 根
  - 原料：6.9m 钢条

求解：
  1) 下料方式不受限时，最少用多少根原料；
  2) 下料方式最多 3 种时，最少用多少根原料。

运行：
  python ch02/hw02_05/solution.py
"""

from gurobipy import GRB, Model, quicksum


def generate_maximal_patterns(stock_len=6.9, lengths=(2.9, 2.1, 1.0)):
    """
    自动枚举“极大下料模式”：
    该模式在不超长前提下，不能再加入任一类型零件。
    """
    stock_u = int(round(stock_len * 10))
    lens_u = [int(round(v * 10)) for v in lengths]

    patterns = []
    max_a = stock_u // lens_u[0]
    max_b = stock_u // lens_u[1]
    max_c = stock_u // lens_u[2]

    for a in range(max_a + 1):
        for b in range(max_b + 1):
            for c in range(max_c + 1):
                if a == 0 and b == 0 and c == 0:
                    continue

                used = a * lens_u[0] + b * lens_u[1] + c * lens_u[2]
                if used > stock_u:
                    continue

                # 若还能再加入任一类型零件，则不是极大模式
                if used + lens_u[0] <= stock_u:
                    continue
                if used + lens_u[1] <= stock_u:
                    continue
                if used + lens_u[2] <= stock_u:
                    continue

                pattern = {
                    "a": a,
                    "b": b,
                    "c": c,
                    "used": used / 10.0,
                    "waste": (stock_u - used) / 10.0,
                }
                patterns.append(pattern)

    patterns.sort(key=lambda p: (p["waste"], -p["a"], -p["b"], -p["c"]))
    return patterns


def solve_cutting(max_modes=None):
    stock_len = 6.9
    lengths = {"L29": 2.9, "L21": 2.1, "L10": 1.0}
    demand = {"L29": 100, "L21": 100, "L10": 100}
    patterns = generate_maximal_patterns(stock_len=stock_len, lengths=(2.9, 2.1, 1.0))
    pattern_ids = list(range(len(patterns)))

    model_name = "hw02_05_part1" if max_modes is None else "hw02_05_part2"
    model = Model(model_name)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x = model.addVars(pattern_ids, vtype=GRB.INTEGER, lb=0, name="x")

    model.addConstr(
        quicksum(patterns[p]["a"] * x[p] for p in pattern_ids) >= demand["L29"],
        name="demand_29",
    )
    model.addConstr(
        quicksum(patterns[p]["b"] * x[p] for p in pattern_ids) >= demand["L21"],
        name="demand_21",
    )
    model.addConstr(
        quicksum(patterns[p]["c"] * x[p] for p in pattern_ids) >= demand["L10"],
        name="demand_10",
    )

    y = None
    if max_modes is not None:
        y = model.addVars(pattern_ids, vtype=GRB.BINARY, name="y")
        big_m = 300
        for p in pattern_ids:
            model.addConstr(x[p] <= big_m * y[p], name=f"link_{p}")
        model.addConstr(quicksum(y[p] for p in pattern_ids) <= max_modes, name="mode_limit")

    model.setObjective(quicksum(x[p] for p in pattern_ids), GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    selected = []
    total_bars = int(round(model.ObjVal))
    produced = {"L29": 0, "L21": 0, "L10": 0}
    total_waste = 0.0

    for p in pattern_ids:
        cnt = int(round(x[p].X))
        if cnt <= 0:
            continue

        pat = patterns[p]
        selected.append((p, cnt, pat))
        produced["L29"] += pat["a"] * cnt
        produced["L21"] += pat["b"] * cnt
        produced["L10"] += pat["c"] * cnt
        total_waste += pat["waste"] * cnt

    total_input_len = total_bars * stock_len
    total_output_len = (
        produced["L29"] * lengths["L29"]
        + produced["L21"] * lengths["L21"]
        + produced["L10"] * lengths["L10"]
    )
    utilization = total_output_len / total_input_len if total_input_len > 0 else 0.0

    result = {
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
        "patterns": patterns,
        "selected": selected,
        "produced": produced,
        "demand": demand,
        "total_bars": total_bars,
        "total_waste": total_waste,
        "total_input_len": total_input_len,
        "total_output_len": total_output_len,
        "utilization": utilization,
        "mode_count": len(selected),
        "max_modes": max_modes,
        "status": model.status,
    }
    return result


def print_result(title, result):
    print(title)
    print(f"ObjVal: {result['obj']:.10f}")
    print(f"ObjBound: {result['obj_bound']:.10f}")
    print(f"MIPGap: {result['mip_gap']:.3e}")
    print(f"最少原料根数: {result['total_bars']}")
    if result["max_modes"] is not None:
        print(f"使用下料方式数量: {result['mode_count']} / 上限 {result['max_modes']}")

    print("采用的下料方式（模式编号从 1 开始）：")
    for pid, cnt, pat in result["selected"]:
        print(
            f"  模式{pid + 1}: "
            f"(2.9m×{pat['a']}, 2.1m×{pat['b']}, 1.0m×{pat['c']}), "
            f"单根余料={pat['waste']:.1f}m, 使用 {cnt} 根"
        )

    print("产出与需求：")
    print(
        f"  2.9m: 产出 {result['produced']['L29']} / 需求 {result['demand']['L29']}"
    )
    print(
        f"  2.1m: 产出 {result['produced']['L21']} / 需求 {result['demand']['L21']}"
    )
    print(
        f"  1.0m: 产出 {result['produced']['L10']} / 需求 {result['demand']['L10']}"
    )

    print("长度统计：")
    print(f"  原料总长度: {result['total_input_len']:.1f} m")
    print(f"  成品总长度: {result['total_output_len']:.1f} m")
    print(f"  总余料长度: {result['total_waste']:.1f} m")
    print(f"  利用率: {result['utilization']:.4%}")


def solve():
    part1 = solve_cutting(max_modes=None)
    if part1 is None:
        print("第(1)问未得到最优解。")
        return

    part2 = solve_cutting(max_modes=3)
    if part2 is None:
        print("第(2)问未得到最优解。")
        return

    print("=== 习题 2.5 (1) 下料方式不限 ===")
    print_result("求解结果：", part1)

    print("\n=== 习题 2.5 (2) 下料方式不超过 3 种 ===")
    print_result("求解结果：", part2)


if __name__ == "__main__":
    solve()
