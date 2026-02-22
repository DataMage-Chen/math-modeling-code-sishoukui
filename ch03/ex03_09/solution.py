"""
例题 3.9：供点与选址（连续选址 + 分配）。

运行：
  python ch03/ex03_09/solution.py
  python ch03/ex03_09/solution.py --mipgap 1e-8
  python ch03/ex03_09/solution.py --time-limit 60
  python ch03/ex03_09/solution.py --verbose
"""

import argparse
import math

from gurobipy import GRB, Model, quicksum


SITES = {
    1: {"a": 1.25, "b": 1.25, "c": 3.0},
    2: {"a": 8.75, "b": 0.75, "c": 5.0},
    3: {"a": 0.50, "b": 4.75, "c": 4.0},
    4: {"a": 3.75, "b": 5.00, "c": 7.0},
    5: {"a": 3.00, "b": 6.50, "c": 6.0},
    6: {"a": 7.25, "b": 7.75, "c": 11.0},
}
DEPOTS = [1, 2]
CAPACITY = 20.0


def solve(mip_gap=1e-8, time_limit=0.0, verbose=False):
    model = Model("ex03_09_location_allocation")
    model.Params.OutputFlag = 1 if verbose else 0
    model.Params.NonConvex = 2
    model.Params.MIPGap = mip_gap
    if time_limit > 0:
        model.Params.TimeLimit = time_limit

    min_a = min(SITES[i]["a"] for i in SITES)
    max_a = max(SITES[i]["a"] for i in SITES)
    min_b = min(SITES[i]["b"] for i in SITES)
    max_b = max(SITES[i]["b"] for i in SITES)

    # 料场坐标变量
    x = {k: model.addVar(lb=min_a, ub=max_a, name=f"x_{k}") for k in DEPOTS}
    y = {k: model.addVar(lb=min_b, ub=max_b, name=f"y_{k}") for k in DEPOTS}

    # 运输量与距离变量
    q = {
        (i, k): model.addVar(lb=0.0, name=f"q_{i}_{k}")
        for i in SITES
        for k in DEPOTS
    }
    d = {
        (i, k): model.addVar(lb=0.0, name=f"d_{i}_{k}")
        for i in SITES
        for k in DEPOTS
    }

    # 每个工地需求必须满足
    for i in SITES:
        model.addConstr(
            quicksum(q[i, k] for k in DEPOTS) == SITES[i]["c"],
            name=f"demand_{i}",
        )

    # 每个料场容量上限
    for k in DEPOTS:
        model.addConstr(
            quicksum(q[i, k] for i in SITES) <= CAPACITY,
            name=f"capacity_{k}",
        )

    # 欧氏距离定义：(xk-ai)^2 + (yk-bi)^2 <= dik^2
    for i in SITES:
        ai, bi = SITES[i]["a"], SITES[i]["b"]
        for k in DEPOTS:
            model.addQConstr(
                (x[k] - ai) * (x[k] - ai) + (y[k] - bi) * (y[k] - bi)
                <= d[i, k] * d[i, k],
                name=f"dist_{i}_{k}",
            )

    # 对称性破除：避免料场 1/2 交换产生等价解
    model.addConstr(x[1] <= x[2], name="symmetry_break")

    # 目标：最小化吨千米
    model.setObjective(
        quicksum(q[i, k] * d[i, k] for i in SITES for k in DEPOTS),
        GRB.MINIMIZE,
    )

    model.optimize()

    if model.SolCount == 0:
        print(f"优化结束，状态码：{model.status}，未找到可行解。")
        return

    x_star = {k: x[k].X for k in DEPOTS}
    y_star = {k: y[k].X for k in DEPOTS}
    q_star = {(i, k): q[i, k].X for i in SITES for k in DEPOTS}

    # 用真实欧氏距离回代，核验总吨千米
    dist_real = {}
    total_ton_km_real = 0.0
    for i in SITES:
        ai, bi = SITES[i]["a"], SITES[i]["b"]
        for k in DEPOTS:
            dist = math.hypot(x_star[k] - ai, y_star[k] - bi)
            dist_real[i, k] = dist
            total_ton_km_real += q_star[i, k] * dist

    print("=== 例题 3.9 求解结果 ===")
    print(f"状态码: {model.status}")
    print(f"ObjVal（模型目标）: {model.ObjVal:.10f}")
    try:
        print(f"ObjBound: {model.ObjBound:.10f}")
        print(f"MIPGap: {model.MIPGap:.3e}")
    except Exception:
        pass
    print(f"按真实距离回代吨千米: {total_ton_km_real:.10f}")
    print(f"|ObjVal-回代值|: {abs(model.ObjVal - total_ton_km_real):.3e}")

    print("\n料场位置与容量使用：")
    for k in DEPOTS:
        used = sum(q_star[i, k] for i in SITES)
        print(
            f"  料场{k}: 坐标=({x_star[k]:.6f}, {y_star[k]:.6f}), "
            f"供货={used:.6f}/{CAPACITY:.0f}"
        )

    print("\n工地供货分配（t）：")
    for i in SITES:
        q1, q2 = q_star[i, 1], q_star[i, 2]
        print(
            f"  工地{i}: q(i,1)={q1:.6f}, q(i,2)={q2:.6f}, "
            f"需求={SITES[i]['c']:.1f}"
        )

    print("\n分项吨千米贡献：")
    for i in SITES:
        for k in DEPOTS:
            qik = q_star[i, k]
            if qik <= 1e-8:
                continue
            contrib = qik * dist_real[i, k]
            print(
                f"  工地{i} <- 料场{k}: 距离={dist_real[i, k]:.6f}, "
                f"运量={qik:.6f}, 贡献={contrib:.6f}"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题3.9 供点与选址")
    parser.add_argument(
        "--mipgap",
        type=float,
        default=1e-8,
        help="全局优化相对间隙阈值（默认 1e-8）",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=30.0,
        help="时间上限（秒），0 表示不限时",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="显示 Gurobi 实时求解日志（节点、bound、gap 等）",
    )
    args = parser.parse_args()
    solve(mip_gap=args.mipgap, time_limit=args.time_limit, verbose=args.verbose)
