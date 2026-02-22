"""
例题 1.9：含最低交易费的组合投资模型（MILP）。

参数说明（可看作超参数）：
- --capital：总资金 M，默认 1000。
- --risk-cap：单次求解时的风险上限 alpha（设置后不扫描 Pareto 点）。
- --points：未设置 --risk-cap 时，Pareto 曲线采样点数，默认 10。
- --linear：强制使用线性化约束；不加时优先使用 Gurobi 一般约束。

调用示例：
- python ch01/ex01_09/solution.py
  默认 M=1000，扫描 Pareto 表并绘制“alpha-最优收益/实际风险”折线图。
- python ch01/ex01_09/solution.py --capital 2000 --points 15
  设定总资金和采样点数，输出并绘制更细的 Pareto 曲线。
- python ch01/ex01_09/solution.py --capital 1000 --risk-cap 8
  固定风险上限做单次优化，输出该约束下的最优组合。
- python ch01/ex01_09/solution.py --linear
  强制线性化建模（便于与一般约束版本做对照）。
"""

import argparse

import gurobipy as gp
import matplotlib.pyplot as plt
# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
from gurobipy import GRB, Model


def get_problem_data():
    asset_ids = [1, 2, 3, 4]
    data = {
        "asset_ids": asset_ids,
        "r0": 0.05,
        "r": {1: 0.28, 2: 0.21, 3: 0.23, 4: 0.25},
        "q": {1: 0.025, 2: 0.015, 3: 0.055, 4: 0.026},
        "p": {1: 0.01, 2: 0.02, 3: 0.045, 4: 0.065},
        "u": {1: 103.0, 2: 198.0, 3: 52.0, 4: 40.0},
    }
    return data


def build_model(capital, risk_cap=None, use_genconstr=True):
    data = get_problem_data()
    asset_ids = data["asset_ids"]
    r0 = data["r0"]
    r = data["r"]
    q = data["q"]
    p = data["p"]
    u = data["u"]

    model = Model("ex01_09")
    model.Params.OutputFlag = 0

    x0 = model.addVar(lb=0.0, name="x0")
    x = model.addVars(asset_ids, lb=0.0, name="x")
    y = model.addVars(asset_ids, vtype=GRB.BINARY, name="y")
    t = model.addVars(asset_ids, lb=0.0, name="t")
    c = model.addVars(asset_ids, lb=0.0, name="c")
    risk_amount = model.addVars(asset_ids, lb=0.0, name="risk")
    z = model.addVar(lb=0.0, name="z")

    model.addConstr(
        x0 + gp.quicksum(x[i] + c[i] for i in asset_ids) == capital, name="budget"
    )

    has_indicator = hasattr(model, "addGenConstrIndicator")
    if has_indicator:
        for i in asset_ids:
            model.addGenConstrIndicator(y[i], 0, x[i] == 0, name=f"link_{i}")
    else:
        for i in asset_ids:
            model.addConstr(x[i] <= capital * y[i], name=f"link_{i}")

    has_max = hasattr(model, "addGenConstrMax")
    use_gen = use_genconstr and has_max

    if use_gen:
        uy = model.addVars(asset_ids, lb=0.0, name="uy")
        for i in asset_ids:
            model.addConstr(uy[i] == u[i] * y[i], name=f"uy_{i}")
            model.addGenConstrMax(t[i], [x[i], uy[i]], name=f"tmax_{i}")
            model.addConstr(c[i] == p[i] * t[i], name=f"fee_{i}")
            model.addConstr(risk_amount[i] == q[i] * x[i], name=f"risk_{i}")
        model.addGenConstrMax(z, [risk_amount[i] for i in asset_ids], name="risk_max")
    else:
        for i in asset_ids:
            model.addConstr(t[i] >= x[i], name=f"t_from_x_{i}")
            model.addConstr(t[i] >= u[i] * y[i], name=f"t_from_u_{i}")
            model.addConstr(c[i] == p[i] * t[i], name=f"fee_{i}")
            model.addConstr(risk_amount[i] == q[i] * x[i], name=f"risk_{i}")
            model.addConstr(z >= risk_amount[i], name=f"risk_max_{i}")

    if risk_cap is not None:
        model.addConstr(z <= risk_cap, name="risk_cap")

    return_expression = r0 * x0 + gp.quicksum(r[i] * x[i] for i in asset_ids)
    model.setObjective(return_expression, GRB.MAXIMIZE)

    variables = {
        "x0": x0,
        "x": x,
        "y": y,
        "t": t,
        "c": c,
        "risk_amount": risk_amount,
        "z": z,
    }
    return model, data, variables


def solve_once(capital, risk_cap=None, use_genconstr=True):
    model, data, var = build_model(
        capital=capital, risk_cap=risk_cap, use_genconstr=use_genconstr
    )
    model.optimize()
    if model.status != GRB.OPTIMAL:
        return None

    asset_ids = data["asset_ids"]
    q = data["q"]
    x_value = {i: var["x"][i].X for i in asset_ids}
    risk_real = max(q[i] * x_value[i] for i in asset_ids)

    result = {
        "return": model.ObjVal,
        "risk": risk_real,
        "z": var["z"].X,
        "x0": var["x0"].X,
        "x": x_value,
        "y": {i: round(var["y"][i].X) for i in asset_ids},
        "t": {i: var["t"][i].X for i in asset_ids},
        "c": {i: var["c"][i].X for i in asset_ids},
    }
    return result


def print_single_result(capital, risk_cap, result):
    data = get_problem_data()
    asset_ids = data["asset_ids"]
    q = data["q"]

    print("=== 例题 1.9 单次求解结果 ===")
    print(f"总资金 M = {capital:.6g}")
    if risk_cap is None:
        print("风险上限: 未设置")
    else:
        print(f"风险上限: z <= {risk_cap:.6g}")
    print(f"最优收益: {result['return']:.6g}")
    print(f"组合风险: {result['risk']:.6g}")
    print(f"银行存款 x0: {result['x0']:.6g}")
    print("各资产决策：")
    for i in asset_ids:
        print(
            "  "
            f"S{i}: y={result['y'][i]}, "
            f"x={result['x'][i]:.6g}, "
            f"fee={result['c'][i]:.6g}, "
            f"q*x={q[i] * result['x'][i]:.6g}"
        )

    total_fee = sum(result["c"][i] for i in asset_ids)
    total_use = result["x0"] + sum(result["x"][i] for i in asset_ids) + total_fee
    print(f"手续费合计: {total_fee:.6g}")
    print(f"资金平衡校验: {total_use:.6g}")


def plot_pareto_rows(rows):
    feasible_rows = [row for row in rows if row[1] is not None]
    if not feasible_rows:
        print("没有可行点，无法绘制折线图。")
        return

    alpha_values = [row[0] for row in feasible_rows]
    returns = [row[1] for row in feasible_rows]
    risks = [row[2] for row in feasible_rows]

    plt.figure(figsize=(8, 5))
    plt.plot(alpha_values, returns, marker="o", linewidth=1.8, label="最优收益")
    plt.plot(alpha_values, risks, marker="s", linewidth=1.8, label="实际风险")
    plt.xlabel("风险上限 alpha")
    plt.ylabel("数值")
    plt.title("风险上限(alpha)与最优收益、实际风险")
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend()
    plt.tight_layout()
    plt.show()


def print_pareto_table(capital, points, use_genconstr=True):
    if points < 2:
        raise ValueError("points 必须不小于 2。")

    best_return_solution = solve_once(
        capital=capital, risk_cap=None, use_genconstr=use_genconstr
    )
    if best_return_solution is None:
        print("模型无最优解。")
        return

    risk_upper = best_return_solution["risk"]
    caps = [risk_upper * idx / (points - 1) for idx in range(points)]

    rows = []
    for cap in caps:
        solution = solve_once(
            capital=capital, risk_cap=cap, use_genconstr=use_genconstr
        )
        if solution is None:
            rows.append((cap, None, None))
        else:
            rows.append((cap, solution["return"], solution["risk"]))

    print("=== 例题 1.9 Pareto 点（ε-约束法） ===")
    print(f"总资金 M = {capital:.6g}")
    print(f"采样点数 = {points}")
    print("风险上限(alpha)\t最优收益\t实际风险")
    for cap, return_value, risk_value in rows:
        if return_value is None:
            print(f"{cap:.6g}\t无可行解\t-")
        else:
            print(f"{cap:.6g}\t{return_value:.6g}\t{risk_value:.6g}")

    print("\n收益最大化对应解：")
    print(f"收益 = {best_return_solution['return']:.6g}")
    print(f"风险 = {best_return_solution['risk']:.6g}")

    plot_pareto_rows(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="求解例题 1.9（交易费分段 + 双目标投资组合）。"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1000.0,
        help="总资金 M（默认 1000）。",
    )
    parser.add_argument(
        "--risk-cap",
        type=float,
        default=None,
        help="单次求解时使用的风险上限 alpha（不填则输出 Pareto 表）。",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=10,
        help="Pareto 采样点个数（默认 10）。",
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="强制使用线性化约束，不使用 Gurobi 的 max 一般约束。",
    )
    args = parser.parse_args()

    if args.capital <= 0:
        raise ValueError("capital 必须为正数。")

    use_gen = not args.linear
    if args.risk_cap is None:
        print_pareto_table(
            capital=args.capital, points=args.points, use_genconstr=use_gen
        )
    else:
        if args.risk_cap < 0:
            raise ValueError("risk-cap 不能为负数。")
        one_result = solve_once(
            capital=args.capital, risk_cap=args.risk_cap, use_genconstr=use_gen
        )
        if one_result is None:
            print("给定风险上限下无可行解。")
        else:
            print_single_result(
                capital=args.capital, risk_cap=args.risk_cap, result=one_result
            )


