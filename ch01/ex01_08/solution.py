"""
例题 1.8：最小化最大绝对误差。

运行：
  python ch01/ex01_08/solution.py
  python ch01/ex01_08/solution.py --linear
"""

import argparse

from gurobipy import GRB, Model


def solve(use_genconstr=True):
    # 自选数据点 (t_i, y_i)
    t = [1, 2, 3, 4, 5]
    y = [1.2, 2.0, 2.7, 3.9, 5.1]
    n = len(t)

    model = Model("ex01_08")
    model.Params.OutputFlag = 0

    a = model.addVar(lb=-GRB.INFINITY, name="a")
    b = model.addVar(lb=-GRB.INFINITY, name="b")
    x = {i: model.addVar(lb=-GRB.INFINITY, name=f"x_{i}") for i in range(n)}
    z = model.addVar(lb=0.0, name="z")

    for i, ti in enumerate(t):
        model.addConstr(x[i] == a + b * ti, name=f"fit_{i}")

    can_use_gen = hasattr(model, "addGenConstrAbs") and hasattr(
        model, "addGenConstrMax"
    )
    use_genconstr = use_genconstr and can_use_gen

    if use_genconstr:
        # 若支持一般约束，直接用 abs/max（需要先用变量表示残差）
        err = {}
        abs_err = {}
        for i in range(n):
            err[i] = model.addVar(lb=-GRB.INFINITY, name=f"err_{i}")
            abs_err[i] = model.addVar(lb=0.0, name=f"abs_{i}")
            model.addConstr(err[i] == x[i] - y[i], name=f"err_def_{i}")
            model.addGenConstrAbs(abs_err[i], err[i], name=f"abs_{i}")
        model.addGenConstrMax(z, [abs_err[i] for i in range(n)], name="max_abs")
    else:
        # 线性化：|x_i - y_i| <= z
        for i in range(n):
            model.addConstr(x[i] - y[i] <= z, name=f"pos_{i}")
            model.addConstr(y[i] - x[i] <= z, name=f"neg_{i}")

    model.setObjective(z, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.OPTIMAL:
        print("数据点 (t_i, y_i)：")
        for i in range(n):
            print(f"  t={t[i]:>2}  y={y[i]:.6g}")
        print(f"最优直线：x = {a.X:.6g} + {b.X:.6g} * t")
        print(f"最大绝对误差：{z.X:.6g}")
        print("拟合值与误差：")
        for i in range(n):
            xi = x[i].X
            err = xi - y[i]
            print(
                f"  t={t[i]:>2}  x={xi:.6g}  y={y[i]:.6g}  误差={err:.6g}"
            )
    else:
        print(f"优化结束，状态码：{model.status}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="求解例题 1.8（最小化最大绝对误差）。"
    )
    parser.add_argument(
        "--linear",
        action="store_true",
        help="强制使用线性化方式，而不使用一般约束。",
    )
    args = parser.parse_args()
    solve(use_genconstr=not args.linear)
