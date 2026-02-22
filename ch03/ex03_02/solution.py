"""
例题 3.2：两类彩电产量优化（二次规划/混合整数二次规划）。

默认把产量视为“台数”，因此用整数变量（MIQP）；
并将 MIPGap 设为较小阈值（默认 1e-8）以更严格逼近最优解；
若想看教材常见的连续近似，可加 --relax。

运行：
  python ch03/ex03_02/solution.py
  python ch03/ex03_02/solution.py --relax
  python ch03/ex03_02/solution.py --mipgap 1e-9
"""

import argparse

from gurobipy import GRB, Model


def analytic_stationary_point():
    """
    解析计算一阶条件对应的驻点：
      0.02*x1 + 0.007*x2 = 144
      0.007*x1 + 0.02*x2 = 174
    """
    a11, a12 = 0.02, 0.007
    a21, a22 = 0.007, 0.02
    b1, b2 = 144.0, 174.0

    det = a11 * a22 - a12 * a21
    x1 = (b1 * a22 - a12 * b2) / det
    x2 = (a11 * b2 - b1 * a21) / det
    return x1, x2


def solve(integer=True, mip_gap=1e-11):
    model = Model("ex03_02_qp")
    model.Params.OutputFlag = 0
    if integer:
        # 设更严格的 MIPGap 阈值，尽量收敛到可证明最优
        model.Params.MIPGap = mip_gap

    vtype = GRB.INTEGER if integer else GRB.CONTINUOUS
    x1 = model.addVar(lb=0.0, vtype=vtype, name="x1")
    x2 = model.addVar(lb=0.0, vtype=vtype, name="x2")

    p1 = 339 - 0.01 * x1 - 0.003 * x2
    p2 = 399 - 0.004 * x1 - 0.01 * x2

    revenue = p1 * x1 + p2 * x2
    cost = 195 * x1 + 225 * x2 + 400000
    profit = revenue - cost

    # 等价为凸二次规划：min -profit
    model.setObjective(-profit, GRB.MINIMIZE)

    # 可选合理性约束：平均售价不为负
    model.addConstr(p1 >= 0, name="price1_nonnegative")
    model.addConstr(p2 >= 0, name="price2_nonnegative")

    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    x1_star = x1.X
    x2_star = x2.X
    p1_star = 339 - 0.01 * x1_star - 0.003 * x2_star
    p2_star = 399 - 0.004 * x1_star - 0.01 * x2_star
    revenue_star = p1_star * x1_star + p2_star * x2_star
    cost_star = 195 * x1_star + 225 * x2_star + 400000
    profit_star = revenue_star - cost_star

    # 一阶条件（连续内部点）梯度校验：整数情形下仅作参考
    grad_x1 = 144 - 0.02 * x1_star - 0.007 * x2_star
    grad_x2 = 174 - 0.007 * x1_star - 0.02 * x2_star

    x1_check, x2_check = analytic_stationary_point()

    print("=== 例题 3.2 求解结果 ===")
    print(f"模型类型: {'MIQP（整数台数）' if integer else 'QP（连续近似）'}")
    if integer:
        print(f"ObjVal: {model.ObjVal:.10f}")
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"x1*（19英寸产量）: {x1_star:.6f}")
    print(f"x2*（21英寸产量）: {x2_star:.6f}")
    print(f"对应平均售价 p1: {p1_star:.6f}")
    print(f"对应平均售价 p2: {p2_star:.6f}")
    print(f"总收入: {revenue_star:.6f}")
    print(f"总成本: {cost_star:.6f}")
    print(f"最大利润: {profit_star:.6f}")

    print("梯度校验（应接近 0）：")
    print(f"  dPi/dx1 = {grad_x1:.3e}")
    print(f"  dPi/dx2 = {grad_x2:.3e}")

    print("解析驻点校验：")
    print(f"  x1_check = {x1_check:.6f}")
    print(f"  x2_check = {x2_check:.6f}")
    print(f"  |x1*-x1_check| = {abs(x1_star - x1_check):.3e}")
    print(f"  |x2*-x2_check| = {abs(x2_star - x2_check):.3e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "例题3.2：两类彩电产量优化（默认整数，--relax 为连续放松）"
        )
    )
    parser.add_argument(
        "--relax",
        action="store_true",
        help="使用连续变量放松（QP），不强制产量为整数",
    )
    parser.add_argument(
        "--mipgap",
        type=float,
        default=1e-8,
        help="MIPGap 阈值（仅整数模型生效），默认 1e-8",
    )
    args = parser.parse_args()

    solve(integer=not args.relax, mip_gap=args.mipgap)
