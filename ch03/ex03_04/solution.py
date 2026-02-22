"""
例题 3.4（续例 3.3）：求解二次规划模型（非凸 QP）。

运行：
  python ch03/ex03_04/solution.py
"""

from gurobipy import GRB, Model


def objective_value(x1, x2):
    """计算目标函数值 z(x1, x2)。"""
    return -x1 * x1 - 0.3 * x1 * x2 - 2 * x2 * x2 + 98 * x1 + 277 * x2


def enumerate_vertices():
    """
    枚举可行域顶点（用于校验）：
      A = (0, 0)
      B = (0, 100)
      C = (200/3, 100/3)
    """
    vertices = [
        ("A", 0.0, 0.0),
        ("B", 0.0, 100.0),
        ("C", 200.0 / 3.0, 100.0 / 3.0),
    ]
    values = [(name, x1, x2, objective_value(x1, x2)) for name, x1, x2 in vertices]
    best = min(values, key=lambda t: t[3])
    return values, best


def solve():
    model = Model("ex03_04_nonconvex_qp")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x1 = model.addVar(lb=0.0, name="x1")
    x2 = model.addVar(lb=0.0, name="x2")

    c_total = model.addConstr(x1 + x2 <= 100, name="total_limit")
    c_ratio = model.addConstr(x1 - 2 * x2 <= 0, name="ratio_limit")

    obj = -x1 * x1 - 0.3 * x1 * x2 - 2 * x2 * x2 + 98 * x1 + 277 * x2
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    z_star = model.ObjVal

    vertex_values, vertex_best = enumerate_vertices()

    print("=== 例题 3.4 求解结果 ===")
    print(f"最优目标值 z*: {z_star:.10f}")
    print(f"x1*: {x1_star:.10f}")
    print(f"x2*: {x2_star:.10f}")

    print("约束校验：")
    print(f"  x1 + x2 = {x1_star + x2_star:.10f} <= 100")
    print(f"  x1 - 2*x2 = {x1_star - 2 * x2_star:.10f} <= 0")
    print(f"  total_limit 松弛量: {c_total.Slack:.10e}")
    print(f"  ratio_limit 松弛量: {c_ratio.Slack:.10e}")

    print("顶点枚举校验：")
    for name, xv1, xv2, zv in vertex_values:
        print(f"  顶点{name}: (x1, x2)=({xv1:.10f}, {xv2:.10f}), z={zv:.10f}")

    name_b, x1_b, x2_b, z_b = vertex_best
    print("枚举最优顶点：")
    print(f"  {name_b}: (x1, x2)=({x1_b:.10f}, {x2_b:.10f}), z={z_b:.10f}")
    print("与 Gurobi 结果差值：")
    print(f"  |x1*-x1_b| = {abs(x1_star - x1_b):.3e}")
    print(f"  |x2*-x2_b| = {abs(x2_star - x2_b):.3e}")
    print(f"  |z*-z_b|   = {abs(z_star - z_b):.3e}")


if __name__ == "__main__":
    solve()
