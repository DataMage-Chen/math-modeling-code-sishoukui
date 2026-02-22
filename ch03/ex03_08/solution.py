"""
例题 3.8：投资组合问题（Markowitz）。

运行：
  python ch03/ex03_08/solution.py
  python ch03/ex03_08/solution.py --target-return 0.15 --var-cap 0.09
  python ch03/ex03_08/solution.py --population-cov
"""

import argparse

from gurobipy import GRB, Model, quicksum


ASSETS = ["A", "B", "C"]

# 表 3.2：三种股票过去 12 年年收益率
RETURNS = {
    "A": [0.3, 0.103, 0.216, -0.056, -0.071, 0.056, 0.038, 0.089, 0.09, 0.083, 0.035, 0.176],
    "B": [0.225, 0.29, 0.216, -0.272, 0.144, 0.107, 0.321, 0.305, 0.195, 0.39, -0.072, 0.715],
    "C": [0.149, 0.26, 0.419, -0.078, 0.169, -0.035, 0.133, 0.732, 0.021, 0.131, 0.006, 0.908],
}


def compute_mean_and_cov(returns, sample_cov=True):
    """由历史收益率计算均值向量与协方差矩阵。"""
    n = len(next(iter(returns.values())))
    means = {a: sum(returns[a]) / n for a in ASSETS}

    denom = n - 1 if sample_cov else n
    cov = {i: {} for i in ASSETS}
    for i in ASSETS:
        for j in ASSETS:
            s = 0.0
            for t in range(n):
                s += (returns[i][t] - means[i]) * (returns[j][t] - means[j])
            cov[i][j] = s / denom
    return means, cov


def qp_variance_expr(x, cov):
    """组合方差二次项：x^T Σ x。"""
    return quicksum(cov[i][j] * x[i] * x[j] for i in ASSETS for j in ASSETS)


def solve_case_1(means, cov, target_return):
    """
    (1) 在期望收益率不低于 target_return 的条件下，最小化方差。
    """
    model = Model("ex03_08_case1")
    model.Params.OutputFlag = 0

    x = model.addVars(ASSETS, lb=0.0, ub=1.0, name="x")

    model.addConstr(quicksum(x[a] for a in ASSETS) == 1.0, name="budget")
    model.addConstr(
        quicksum(means[a] * x[a] for a in ASSETS) >= target_return, name="ret_lb"
    )

    var_expr = qp_variance_expr(x, cov)
    model.setObjective(var_expr, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"[模型一] 优化结束，状态码：{model.status}")
        return None

    x_star = {a: x[a].X for a in ASSETS}
    ret_star = sum(means[a] * x_star[a] for a in ASSETS)
    var_star = model.ObjVal
    return x_star, ret_star, var_star


def solve_case_2(means, cov, var_cap):
    """
    (2) 在方差不超过 var_cap 的条件下，最大化期望收益率。
    """
    model = Model("ex03_08_case2")
    model.Params.OutputFlag = 0

    x = model.addVars(ASSETS, lb=0.0, ub=1.0, name="x")

    model.addConstr(quicksum(x[a] for a in ASSETS) == 1.0, name="budget")
    model.addQConstr(qp_variance_expr(x, cov) <= var_cap, name="var_ub")

    ret_expr = quicksum(means[a] * x[a] for a in ASSETS)
    model.setObjective(ret_expr, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"[模型二] 优化结束，状态码：{model.status}")
        return None

    x_star = {a: x[a].X for a in ASSETS}
    ret_star = model.ObjVal
    var_star = sum(
        cov[i][j] * x_star[i] * x_star[j] for i in ASSETS for j in ASSETS
    )
    return x_star, ret_star, var_star


def print_stats(means, cov, sample_cov):
    """打印收益均值与协方差矩阵。"""
    print("=== 数据统计 ===")
    print(f"协方差口径: {'样本协方差(n-1)' if sample_cov else '总体协方差(n)'}")
    print("各股票平均收益率：")
    for a in ASSETS:
        print(f"  {a}: {means[a]:.6f}")

    print("协方差矩阵 Σ：")
    header = "        " + "".join([f"{a:>12s}" for a in ASSETS])
    print(header)
    for i in ASSETS:
        row = f"{i:>4s}  " + "".join([f"{cov[i][j]:12.6f}" for j in ASSETS])
        print(row)


def print_solution_case_1(result, target_return):
    """打印模型一结果。"""
    if result is None:
        return
    x_star, ret_star, var_star = result
    print("\n=== 例题 3.8（1）最小风险配置 ===")
    print(f"收益下限: {target_return:.6f}")
    print(f"最小方差: {var_star:.10f}")
    print(f"对应标准差: {var_star ** 0.5:.10f}")
    print(f"组合期望收益率: {ret_star:.10f}")
    print("最优投资比例：")
    for a in ASSETS:
        print(f"  x_{a} = {x_star[a]:.10f}")
    print(f"比例和校验: {sum(x_star.values()):.10f}")


def print_solution_case_2(result, var_cap):
    """打印模型二结果。"""
    if result is None:
        return
    x_star, ret_star, var_star = result
    print("\n=== 例题 3.8（2）最大收益配置 ===")
    print(f"方差上限: {var_cap:.6f}")
    print(f"最大期望收益率: {ret_star:.10f}")
    print(f"对应组合方差: {var_star:.10f}")
    print(f"对应标准差: {var_star ** 0.5:.10f}")
    print("最优投资比例：")
    for a in ASSETS:
        print(f"  x_{a} = {x_star[a]:.10f}")
    print(f"比例和校验: {sum(x_star.values()):.10f}")


def main():
    parser = argparse.ArgumentParser(description="例题3.8 投资组合优化")
    parser.add_argument(
        "--target-return",
        type=float,
        default=0.15,
        help="模型一的收益率下限，默认 0.15",
    )
    parser.add_argument(
        "--var-cap",
        type=float,
        default=0.09,
        help="模型二的方差上限，默认 0.09",
    )
    parser.add_argument(
        "--population-cov",
        action="store_true",
        help="使用总体协方差（分母 n）；默认使用样本协方差（分母 n-1）",
    )
    args = parser.parse_args()

    sample_cov = not args.population_cov
    means, cov = compute_mean_and_cov(RETURNS, sample_cov=sample_cov)
    print_stats(means, cov, sample_cov)

    result1 = solve_case_1(means, cov, args.target_return)
    print_solution_case_1(result1, args.target_return)

    result2 = solve_case_2(means, cov, args.var_cap)
    print_solution_case_2(result2, args.var_cap)


if __name__ == "__main__":
    main()
