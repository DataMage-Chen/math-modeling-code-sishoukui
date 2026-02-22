"""
习题 1.8（例 1.9 的模型二）：固定收益水平，最小化风险。

参数说明：
- --capital：总资金 M（默认 10000）。
- --k：目标净收益率下限，约束为 sum((r_i-p_i)*x_i) >= k*M（默认 0.15）。
- --k-step：绘制收益阈值-最小风险曲线时的步长（默认 0.005）。

运行示例：
- python ch01/hw01_08/solution.py
- python ch01/hw01_08/solution.py --capital 10000 --k 0.18
- python ch01/hw01_08/solution.py --capital 1 --k 0.21 --k-step 0.005
"""

import argparse

import matplotlib.pyplot as plt
from gurobipy import GRB, Model, quicksum


plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def get_problem_data():
    all_ids = [0, 1, 2, 3, 4]
    risky_ids = [1, 2, 3, 4]
    data = {
        "all_ids": all_ids,
        "risky_ids": risky_ids,
        "r": {0: 0.05, 1: 0.28, 2: 0.21, 3: 0.23, 4: 0.25},
        "q": {0: 0.0, 1: 0.025, 2: 0.015, 3: 0.055, 4: 0.026},
        "p": {0: 0.0, 1: 0.01, 2: 0.02, 3: 0.045, 4: 0.065},
    }
    return data


def calc_k_range(capital, data):
    all_ids = data["all_ids"]
    r = data["r"]
    p = data["p"]

    model = Model("hw01_08_range")
    model.Params.OutputFlag = 0

    x = model.addVars(all_ids, lb=0.0, name="x")
    model.addConstr(
        quicksum((1.0 + p[i]) * x[i] for i in all_ids) == capital,
        name="budget",
    )

    return_expr = quicksum((r[i] - p[i]) * x[i] for i in all_ids)

    model.setObjective(return_expr, GRB.MINIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        return None, None
    k_min = model.ObjVal / capital

    model.setObjective(return_expr, GRB.MAXIMIZE)
    model.optimize()
    if model.status != GRB.OPTIMAL:
        return None, None
    k_max = model.ObjVal / capital

    return k_min, k_max


def solve_once(capital, k, data):
    all_ids = data["all_ids"]
    risky_ids = data["risky_ids"]
    r = data["r"]
    q = data["q"]
    p = data["p"]

    model = Model("hw01_08")
    model.Params.OutputFlag = 0

    x = model.addVars(all_ids, lb=0.0, name="x")
    z = model.addVar(lb=0.0, name="z")

    model.addConstr(
        quicksum((r[i] - p[i]) * x[i] for i in all_ids) >= k * capital,
        name="return_floor",
    )
    model.addConstr(
        quicksum((1.0 + p[i]) * x[i] for i in all_ids) == capital,
        name="budget",
    )
    for i in risky_ids:
        model.addConstr(q[i] * x[i] <= z, name=f"risk_{i}")

    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    return_value = sum((r[i] - p[i]) * x[i].X for i in all_ids)
    budget_use = sum((1.0 + p[i]) * x[i].X for i in all_ids)
    risk_values = {i: q[i] * x[i].X for i in risky_ids}

    result = {
        "z": z.X,
        "return_value": return_value,
        "return_ratio": return_value / capital,
        "budget_use": budget_use,
        "x": {i: x[i].X for i in all_ids},
        "risk_values": risk_values,
    }
    return result


def build_k_grid(k_min, k_max, k_step):
    grid = []
    index = 0
    while True:
        current = k_min + index * k_step
        if current > k_max + 1e-12:
            break
        grid.append(current)
        index += 1

    if not grid or grid[-1] < k_max - 1e-12:
        grid.append(k_max)
    return grid


def plot_k_risk_curve(capital, data, k_min, k_max, k_step, selected_k=None):
    if k_step <= 0:
        raise ValueError("k-step 必须为正数。")

    k_values = build_k_grid(k_min, k_max, k_step)
    feasible_points = []
    for k in k_values:
        result = solve_once(capital=capital, k=k, data=data)
        if result is not None:
            feasible_points.append((k, result["z"]))

    if not feasible_points:
        print("未得到可行点，无法绘制收益与风险关系图。")
        return

    xs = [item[0] for item in feasible_points]
    ys = [item[1] for item in feasible_points]

    plt.figure(figsize=(8, 5))
    plt.plot(xs, ys, marker="o", linewidth=1.8, markersize=3.5, label="最小风险")

    if selected_k is not None and k_min - 1e-12 <= selected_k <= k_max + 1e-12:
        selected_result = solve_once(capital=capital, k=selected_k, data=data)
        if selected_result is not None:
            plt.scatter(
                [selected_k],
                [selected_result["z"]],
                color="red",
                s=45,
                label=f"当前 k={selected_k:.4g}",
                zorder=3,
            )

    plt.xlabel("收益阈值 k")
    plt.ylabel("最小风险 z")
    plt.title("收益阈值与最小风险关系图（模型二）")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(capital=10000.0, k=0.15, k_step=0.005):
    data = get_problem_data()
    k_min, k_max = calc_k_range(capital, data)

    if k_min is None or k_max is None:
        print("无法计算 k 的可行区间。")
        return

    print("=== 习题 1.8 求解结果（例 1.9 模型二） ===")
    print(f"总资金 M = {capital:.6g}")
    print(f"目标净收益率下限 k = {k:.6g}")
    print(f"k 可行区间约为 [{k_min:.6g}, {k_max:.6g}]")

    if k < k_min - 1e-10 or k > k_max + 1e-10:
        print("给定 k 超出可行区间，模型无可行解。")
        return

    result = solve_once(capital=capital, k=k, data=data)
    if result is None:
        print("给定 k 下无可行解。")
        return

    print(f"最小总体风险 z = {result['z']:.10f}")
    print(f"净收益总额 = {result['return_value']:.10f}")
    print(f"净收益率 = {result['return_ratio']:.10f}")
    print(f"资金平衡校验 = {result['budget_use']:.10f}")
    print("投资决策（元）：")
    print(f"  银行 x0 = {result['x'][0]:.10f}")
    for i in data["risky_ids"]:
        print(
            f"  S{i}: x{i} = {result['x'][i]:.10f}, "
            f"q{i}*x{i} = {result['risk_values'][i]:.10f}"
        )

    print(f"\n按步长 {k_step:.6g} 扫描 k 并绘制收益与风险关系图...")
    plot_k_risk_curve(
        capital=capital,
        data=data,
        k_min=k_min,
        k_max=k_max,
        k_step=k_step,
        selected_k=k,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 1.8（例 1.9 模型二）。")
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="总资金 M（默认 10000）。",
    )
    parser.add_argument(
        "--k",
        type=float,
        default=0.15,
        help="目标净收益率下限 k（默认 0.15）。",
    )
    parser.add_argument(
        "--k-step",
        type=float,
        default=0.005,
        help="绘制收益与风险关系图时 k 的步长（默认 0.005）。",
    )
    args = parser.parse_args()

    if args.capital <= 0:
        raise ValueError("capital 必须为正数。")
    if args.k_step <= 0:
        raise ValueError("k-step 必须为正数。")

    solve(capital=args.capital, k=args.k, k_step=args.k_step)

