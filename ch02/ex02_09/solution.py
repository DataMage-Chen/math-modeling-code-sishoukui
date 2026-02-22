"""
例题 2.9：非线性整数规划。

功能：
1) 蒙特卡洛随机可行采样求近似最优解；
2) Gurobi 精确求解（MIQP）；
3) 对比两者目标值误差。

运行示例：
  python ch02/ex02_09/solution.py
  python ch02/ex02_09/solution.py --samples 200000 --seed 2026
"""

import argparse
import random

from gurobipy import GRB, Model


def objective_value(x):
    x1, x2, x3, x4, x5 = x
    return (
        x1 * x1
        + x2 * x2
        + 3 * x3 * x3
        + 4 * x4 * x4
        + 2 * x5 * x5
        - 8 * x1
        - 2 * x2
        - 3 * x3
        - x4
        - 2 * x5
    )


def is_feasible(x):
    x1, x2, x3, x4, x5 = x
    if not all(0 <= value <= 99 for value in x):
        return False
    if x1 + x2 + x3 + x4 + x5 > 400:
        return False
    if x1 + 2 * x2 + 2 * x3 + x4 + 6 * x5 > 800:
        return False
    if 2 * x1 + x2 + 6 * x3 > 200:
        return False
    if x3 + x4 + 5 * x5 > 200:
        return False
    return True


def random_feasible_point(rng, max_trials=200):
    # 通过约束引导采样，显著提高可行命中率
    for _ in range(max_trials):
        x3 = rng.randint(0, 33)

        x5_max = min(99, (200 - x3) // 5)
        if x5_max < 0:
            continue
        x5 = rng.randint(0, x5_max)

        x4_max = min(99, 200 - x3 - 5 * x5)
        if x4_max < 0:
            continue
        x4 = rng.randint(0, x4_max)

        x2_max = min(99, 200 - 6 * x3)
        if x2_max < 0:
            continue
        x2 = rng.randint(0, x2_max)

        x1_max = min(
            99,
            (200 - x2 - 6 * x3) // 2,
            400 - (x2 + x3 + x4 + x5),
            800 - (2 * x2 + 2 * x3 + x4 + 6 * x5),
        )
        if x1_max < 0:
            continue
        x1 = rng.randint(0, x1_max)

        point = [x1, x2, x3, x4, x5]
        if is_feasible(point):
            return point
    return None


def solve_monte_carlo(samples=100000, seed=42):
    if samples <= 0:
        raise ValueError("samples 必须为正整数。")

    rng = random.Random(seed)
    best_x = None
    best_obj = None
    feasible_count = 0

    for _ in range(samples):
        point = random_feasible_point(rng)
        if point is None:
            continue

        feasible_count += 1
        obj = objective_value(point)
        if best_obj is None or obj > best_obj:
            best_obj = obj
            best_x = point

    result = {
        "best_x": best_x,
        "best_obj": best_obj,
        "samples": samples,
        "feasible_count": feasible_count,
        "seed": seed,
    }
    return result


def solve_exact():
    model = Model("ex02_09_exact")
    model.Params.OutputFlag = 0
    model.Params.NonConvex = 2
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x = model.addVars(5, vtype=GRB.INTEGER, lb=0.0, ub=99.0, name="x")

    model.addConstr(x[0] + x[1] + x[2] + x[3] + x[4] <= 400, name="c1")
    model.addConstr(x[0] + 2 * x[1] + 2 * x[2] + x[3] + 6 * x[4] <= 800, name="c2")
    model.addConstr(2 * x[0] + x[1] + 6 * x[2] <= 200, name="c3")
    model.addConstr(x[2] + x[3] + 5 * x[4] <= 200, name="c4")

    model.setObjective(
        x[0] * x[0]
        + x[1] * x[1]
        + 3 * x[2] * x[2]
        + 4 * x[3] * x[3]
        + 2 * x[4] * x[4]
        - 8 * x[0]
        - 2 * x[1]
        - 3 * x[2]
        - x[3]
        - 2 * x[4],
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    exact_x = [int(round(x[i].X)) for i in range(5)]
    result = {
        "x": exact_x,
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
    }
    return result


def print_vector(name, values):
    text = ", ".join(f"{name}{idx + 1}={int(values[idx])}" for idx in range(len(values)))
    print(text)


def solve(samples=100000, seed=42):
    mc = solve_monte_carlo(samples=samples, seed=seed)
    exact = solve_exact()

    if mc["best_x"] is None:
        print("蒙特卡洛未采到可行点，请增大样本数。")
        return
    if exact is None:
        print("精确求解未得到最优解。")
        return

    mc_obj = mc["best_obj"]
    exact_obj = exact["obj"]
    abs_error = abs(exact_obj - mc_obj)
    rel_error = abs_error / max(1.0, abs(exact_obj))

    print("=== 例题 2.9 蒙特卡洛求解 ===")
    print(f"样本数: {mc['samples']}")
    print(f"随机种子: {mc['seed']}")
    print(f"可行样本数: {mc['feasible_count']}")
    print(f"蒙特卡洛最好目标值: {mc_obj:.10f}")
    print_vector("x", mc["best_x"])

    print("\n=== 例题 2.9 精确求解（MIQP） ===")
    print(f"ObjVal: {exact_obj:.10f}")
    print(f"ObjBound: {exact['obj_bound']:.10f}")
    print(f"MIPGap: {exact['mip_gap']:.3e}")
    print_vector("x", exact["x"])

    print("\n=== 误差对比（蒙特卡洛 vs 精确解） ===")
    print(f"目标值绝对误差: {abs_error:.10f}")
    print(f"目标值相对误差: {rel_error:.6%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解例题 2.9（蒙特卡洛 + 精确求解对比）。")
    parser.add_argument(
        "--samples",
        type=int,
        default=100000,
        help="蒙特卡洛样本数（默认 100000）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）。",
    )
    args = parser.parse_args()

    solve(samples=args.samples, seed=args.seed)

