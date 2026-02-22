"""
例题 3.3：求解二次规划模型（QP）。

运行：
  python ch03/ex03_03/solution.py
"""

from gurobipy import GRB, Model


def analytic_solution_on_active_boundary():
    """
    解析校验：
    先假设最优点在活跃边界 x1 + x2 = 100 上，
    代入 x1 = 100 - x2 后做一维二次函数求极值。
    """
    x2 = 349.0 / 5.4
    x1 = 100.0 - x2
    return x1, x2


def solve():
    model = Model("ex03_03_qp")
    model.Params.OutputFlag = 0

    x1 = model.addVar(lb=0.0, name="x1")
    x2 = model.addVar(lb=0.0, name="x2")

    c_total = model.addConstr(x1 + x2 <= 100, name="total_limit")
    c_ratio = model.addConstr(x1 - 2 * x2 <= 0, name="ratio_limit")

    obj = -x1 * x1 - 0.3 * x1 * x2 - 2 * x2 * x2 + 98 * x1 + 277 * x2
    model.setObjective(obj, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    z_star = model.ObjVal

    # 约束校验
    total_used = x1_star + x2_star
    ratio_lhs = x1_star - 2 * x2_star

    x1_check, x2_check = analytic_solution_on_active_boundary()
    z_check = (
        -x1_check * x1_check
        - 0.3 * x1_check * x2_check
        - 2 * x2_check * x2_check
        + 98 * x1_check
        + 277 * x2_check
    )

    print("=== 例题 3.3 求解结果 ===")
    print(f"最优目标值 z*: {z_star:.10f}")
    print(f"x1*: {x1_star:.10f}")
    print(f"x2*: {x2_star:.10f}")

    print("约束校验：")
    print(f"  x1 + x2 = {total_used:.10f} <= 100")
    print(f"  x1 - 2*x2 = {ratio_lhs:.10f} <= 0")
    print(f"  x1 >= 0: {x1_star >= -1e-9}")
    print(f"  x2 >= 0: {x2_star >= -1e-9}")
    print(f"  total_limit 松弛量: {c_total.Slack:.10e}")
    print(f"  ratio_limit 松弛量: {c_ratio.Slack:.10e}")

    print("解析校验（基于 x1 + x2 = 100 的一维化）：")
    print(f"  x1_check: {x1_check:.10f}")
    print(f"  x2_check: {x2_check:.10f}")
    print(f"  z_check : {z_check:.10f}")
    print(f"  |x1*-x1_check| = {abs(x1_star - x1_check):.3e}")
    print(f"  |x2*-x2_check| = {abs(x2_star - x2_check):.3e}")
    print(f"  |z*-z_check|   = {abs(z_star - z_check):.3e}")


if __name__ == "__main__":
    solve()
