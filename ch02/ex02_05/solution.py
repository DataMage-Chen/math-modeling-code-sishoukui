"""
例题 2.5：轮班配工优化（整数规划）。

运行：
  python ch02/ex02_05/solution.py
"""

from gurobipy import GRB, Model, quicksum


def solve():
    # 6 个时段（每段 4 小时）的最低用工需求
    demands = [35, 40, 50, 45, 55, 30]
    periods = [
        "0:00-4:00",
        "4:00-8:00",
        "8:00-12:00",
        "12:00-16:00",
        "16:00-20:00",
        "20:00-24:00",
    ]
    period_count = len(demands)

    model = Model("ex02_05_shift_staffing")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    # x[i]：在第 i 个时段开始上岗的工人数（工作 8 小时，覆盖 i 与 i+1）
    x = model.addVars(period_count, vtype=GRB.INTEGER, lb=0.0, name="x")

    for i in range(period_count):
        previous = (i - 1) % period_count
        model.addConstr(x[previous] + x[i] >= demands[i], name=f"cover_{i + 1}")

    model.setObjective(quicksum(x[i] for i in range(period_count)), GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        total_workers = sum(x[i].X for i in range(period_count))
        print("=== 例题 2.5 求解结果 ===")
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
        print(f"最少配备工人数: {total_workers:.6g}")

        print("各班次起始上岗人数：")
        for i in range(period_count):
            next_idx = (i + 1) % period_count
            print(
                f"  班次{i + 1}（{periods[i]}起）: {x[i].X:.6g} 人，"
                f"覆盖 {periods[i]} 与 {periods[next_idx]}"
            )

        print("各时段覆盖校验：")
        for i in range(period_count):
            previous = (i - 1) % period_count
            cover = x[previous].X + x[i].X
            print(
                f"  时段{i + 1}（{periods[i]}）: 需求 {demands[i]}, 覆盖 {cover:.6g}"
            )
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

