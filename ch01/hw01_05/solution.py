"""
习题 1.5：五年投资组合规划（线性规划）。

运行：
  python ch01/hw01_05/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("hw01_05")
    model.Params.OutputFlag = 0

    # 变量单位均为“万元”
    a = {t: model.addVar(lb=0.0, name=f"a{t}") for t in [1, 2, 3, 4]}
    d = {t: model.addVar(lb=0.0, name=f"d{t}") for t in [1, 2, 3, 4, 5]}
    b = model.addVar(lb=0.0, name="b")
    c = model.addVar(lb=0.0, name="c")

    model.addConstr(a[1] + d[1] == 10, name="cash_y1")
    model.addConstr(a[2] + c + d[2] == 1.06 * d[1], name="cash_y2")
    model.addConstr(a[3] + b + d[3] == 1.06 * d[2] + 1.15 * a[1], name="cash_y3")
    model.addConstr(a[4] + d[4] == 1.06 * d[3] + 1.15 * a[2], name="cash_y4")
    model.addConstr(d[5] == 1.06 * d[4] + 1.15 * a[3], name="cash_y5")

    model.addConstr(b <= 4, name="cap_b")
    model.addConstr(c <= 3, name="cap_c")

    final_wealth = 1.06 * d[5] + 1.15 * a[4] + 1.25 * b + 1.40 * c
    model.setObjective(final_wealth, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("=== 习题 1.5 求解结果 ===")
        print(f"第5年末最大资金本利总额（万元）: {model.ObjVal:.10f}")
        print("投资决策（万元）：")
        print(f"  A1={a[1].X:.6g}, A2={a[2].X:.6g}, A3={a[3].X:.6g}, A4={a[4].X:.6g}")
        print(f"  B={b.X:.6g}, C={c.X:.6g}")
        print(
            f"  D1={d[1].X:.6g}, D2={d[2].X:.6g}, D3={d[3].X:.6g}, "
            f"D4={d[4].X:.6g}, D5={d[5].X:.6g}"
        )

        print("逐年资金平衡校验（左侧应约等于右侧）：")
        y1_lhs, y1_rhs = a[1].X + d[1].X, 10.0
        y2_lhs, y2_rhs = a[2].X + c.X + d[2].X, 1.06 * d[1].X
        y3_lhs, y3_rhs = a[3].X + b.X + d[3].X, 1.06 * d[2].X + 1.15 * a[1].X
        y4_lhs, y4_rhs = a[4].X + d[4].X, 1.06 * d[3].X + 1.15 * a[2].X
        y5_lhs, y5_rhs = d[5].X, 1.06 * d[4].X + 1.15 * a[3].X
        print(f"  年1: {y1_lhs:.10f} = {y1_rhs:.10f}")
        print(f"  年2: {y2_lhs:.10f} = {y2_rhs:.10f}")
        print(f"  年3: {y3_lhs:.10f} = {y3_rhs:.10f}")
        print(f"  年4: {y4_lhs:.10f} = {y4_rhs:.10f}")
        print(f"  年5: {y5_lhs:.10f} = {y5_rhs:.10f}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    solve()

