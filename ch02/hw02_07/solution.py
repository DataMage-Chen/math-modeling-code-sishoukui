"""
习题 2.7：家电产品生产计划（整数线性规划）。

运行示例：
  python ch02/hw02_07/solution.py
  python ch02/hw02_07/solution.py --pool
"""

import argparse

from gurobipy import GRB, Model


def extract_solution(model, x1, x2, use_pool=False, idx=0):
    if use_pool:
        model.Params.SolutionNumber = idx
        read = lambda var: var.Xn
        obj = model.PoolObjVal
    else:
        read = lambda var: var.X
        obj = model.ObjVal

    x1_val = int(round(read(x1)))
    x2_val = int(round(read(x2)))
    return {"x1": x1_val, "x2": x2_val, "obj": obj}


def print_solution(title, sol):
    print(title)
    print(f"  产品 I : {sol['x1']}")
    print(f"  产品 II: {sol['x2']}")
    print(f"  利润: {sol['obj']:.6f}")


def solve(show_pool=False, pool_size=10):
    model = Model("hw02_07_plan")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    if show_pool:
        model.Params.PoolSearchMode = 2
        model.Params.PoolSolutions = pool_size
        model.Params.PoolGap = 0

    x1 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x1")
    x2 = model.addVar(vtype=GRB.INTEGER, lb=0, name="x2")

    model.addConstr(5 * x2 <= 15, name="cap_A")
    model.addConstr(6 * x1 + 2 * x2 <= 24, name="cap_B")
    model.addConstr(x1 + x2 <= 5, name="cap_test")

    model.setObjective(2 * x1 + x2, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    print("=== 习题 2.7 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")

    best = extract_solution(model, x1, x2, use_pool=False)
    print_solution("一个最优生产方案：", best)

    print("约束校验：")
    print(f"  设备A: 5*x2 = {5 * best['x2']} <= 15")
    print(f"  设备B: 6*x1+2*x2 = {6 * best['x1'] + 2 * best['x2']} <= 24")
    print(f"  调试 : x1+x2 = {best['x1'] + best['x2']} <= 5")

    if show_pool:
        print(f"\n最优解池数量（受 pool-size 限制）: {model.SolCount}")
        for idx in range(model.SolCount):
            sol = extract_solution(model, x1, x2, use_pool=True, idx=idx)
            print_solution(f"解池方案 #{idx + 1}：", sol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 2.7（家电产品生产计划）。")
    parser.add_argument(
        "--pool",
        action="store_true",
        help="输出多个最优解（解池模式）。",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=10,
        help="解池最多保留的解数（默认 10）。",
    )
    args = parser.parse_args()

    if args.pool_size <= 0:
        raise ValueError("pool-size 必须为正整数。")

    solve(show_pool=args.pool, pool_size=args.pool_size)
