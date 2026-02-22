"""
例题 3.6：求 f(x)=100(x2-x1^2)^2+(1-x1)^2 的极小点。

运行：
  python ch03/ex03_06/solution.py
"""

from gurobipy import GRB, Model


def solve():
    model = Model("ex03_06_rosenbrock")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2

    x1 = model.addVar(lb=-GRB.INFINITY, name="x1")
    x2 = model.addVar(lb=-GRB.INFINITY, name="x2")
    t = model.addVar(lb=-GRB.INFINITY, name="t")

    model.addQConstr(t == x2 - x1 * x1, name="def_t")
    model.setObjective(100 * t * t + (1 - x1) * (1 - x1), GRB.MINIMIZE)

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    t_star = t.X
    f_star = model.ObjVal

    # 解析最优解用于校验
    x1_true, x2_true, f_true = 1.0, 1.0, 0.0
    residual_t = t_star - (x2_star - x1_star * x1_star)
    f_raw = 100 * (x2_star - x1_star * x1_star) ** 2 + (1 - x1_star) ** 2

    print("=== 例题 3.6 求解结果 ===")
    print(f"最优目标值 f*: {f_star:.12f}")
    print(f"x1*: {x1_star:.12f}")
    print(f"x2*: {x2_star:.12f}")
    print(f"t* : {t_star:.12f}")

    print("约束与函数值校验：")
    print(f"  def_t 残差 = {residual_t:.3e}")
    print(f"  按原函数回代 f(x1*,x2*) = {f_raw:.12f}")

    print("解析解校验：")
    print(f"  (x1_true, x2_true) = ({x1_true:.1f}, {x2_true:.1f})")
    print(f"  f_true = {f_true:.1f}")
    print(f"  |x1*-1| = {abs(x1_star - x1_true):.3e}")
    print(f"  |x2*-1| = {abs(x2_star - x2_true):.3e}")
    print(f"  |f*-0|  = {abs(f_star - f_true):.3e}")


if __name__ == "__main__":
    solve()
