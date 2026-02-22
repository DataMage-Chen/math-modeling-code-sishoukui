"""
习题 1.7：max-min 线性规划。

参数说明：
- --seed：随机种子，用于生成矩阵 A（默认 42）。

运行示例：
- python ch01/hw01_07/solution.py
- python ch01/hw01_07/solution.py --seed 2026
"""

import argparse
import random

from gurobipy import GRB, Model, quicksum


def generate_matrix(rows, cols, seed):
    random.seed(seed)
    return [[random.randint(0, 10) for _ in range(cols)] for _ in range(rows)]


def solve(seed=42):
    row_count = 100
    col_count = 150
    matrix_a = generate_matrix(row_count, col_count, seed)

    model = Model("hw01_07")
    model.Params.OutputFlag = 0

    x = model.addVars(row_count, lb=0.0, name="x")
    v = model.addVar(lb=-GRB.INFINITY, name="v")

    for col in range(col_count):
        model.addConstr(
            quicksum(matrix_a[row][col] * x[row] for row in range(row_count)) >= v,
            name=f"c_{col + 1}",
        )

    model.addConstr(quicksum(x[row] for row in range(row_count)) == 1, name="sum_x")
    model.setObjective(v, GRB.MAXIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        x_value = [x[row].X for row in range(row_count)]
        col_values = [
            sum(matrix_a[row][col] * x_value[row] for row in range(row_count))
            for col in range(col_count)
        ]
        min_col_value = min(col_values)
        active_count = sum(1 for value in x_value if value > 1e-8)

        top_items = sorted(
            [(index + 1, value) for index, value in enumerate(x_value)],
            key=lambda item: item[1],
            reverse=True,
        )[:10]

        print("=== 习题 1.7 求解结果 ===")
        print(f"随机种子: {seed}")
        print(f"最优目标值 v: {model.ObjVal:.10f}")
        print(f"校验 min_j(sum_i a_ij*x_i): {min_col_value:.10f}")
        print(f"二者差值: {abs(model.ObjVal - min_col_value):.3e}")
        print(f"x 中非零变量个数: {active_count}")
        print("x 的前 10 个较大分量（索引从 1 开始）：")
        for idx, value in top_items:
            print(f"  x_{idx} = {value:.10f}")
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 1.7。")
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）。",
    )
    args = parser.parse_args()
    solve(seed=args.seed)

