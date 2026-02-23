"""
例题 5.1：插值函数构造与估值（基于 SciPy）。

运行：
  python ch05/ex05_01_02/solution.py
  python ch05/ex05_01_02/solution.py --queries 1.5 2.6
"""

import argparse

try:
    import numpy as np
    from scipy.interpolate import BarycentricInterpolator
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装 numpy 和 scipy：\n"
        "  pip install numpy scipy"
    ) from exc


def format_polynomial(coeffs):
    """把多项式系数格式化为可读表达式（从高次到低次）。"""
    degree = len(coeffs) - 1
    terms = []
    for idx, c in enumerate(coeffs):
        power = degree - idx
        if abs(c) < 1e-12:
            continue
        sign = "+" if c >= 0 else "-"
        val = abs(c)
        if power == 0:
            term = f"{val:.10f}"
        elif power == 1:
            term = f"{val:.10f}*x"
        else:
            term = f"{val:.10f}*x^{power}"
        terms.append((sign, term))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    expr = first_term if first_sign == "+" else f"-{first_term}"
    for sign, term in terms[1:]:
        expr += f" {sign} {term}"
    return expr


def solve(queries):
    x_obs = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    y_obs = np.array([16, 18, 21, 17, 15, 12], dtype=float)

    # 用重心拉格朗日法构造插值函数（数值稳定）
    interpolator = BarycentricInterpolator(x_obs, y_obs)
    y_query = interpolator(np.array(queries, dtype=float))

    # 输出显式 5 次多项式系数（便于报告书写）
    coeffs = np.polyfit(x_obs, y_obs, deg=len(x_obs) - 1)
    poly = np.poly1d(coeffs)

    # 校验是否通过全部观测点
    y_fit = poly(x_obs)
    max_err = float(np.max(np.abs(y_fit - y_obs)))

    print("=== 例题 5.1 求解结果（插值） ===")
    print("观测点：")
    for xi, yi in zip(x_obs, y_obs):
        print(f"  ({xi:.1f}, {yi:.1f})")

    print("\n插值多项式（5次）系数（高次到低次）：")
    print("  [" + ", ".join(f"{c:.12f}" for c in coeffs) + "]")
    print("显式表达式（近似）：")
    print(f"  y = {format_polynomial(coeffs)}")

    print("\n插值条件校验：")
    print(f"  max|hat_f(x_i)-y_i| = {max_err:.3e}")

    print("\n指定点估值：")
    for xq, yq in zip(queries, y_query):
        print(f"  hat_f({xq}) = {float(yq):.10f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.1 插值求解（SciPy）")
    parser.add_argument(
        "--queries",
        type=float,
        nargs="+",
        default=[1.5, 2.6],
        help="需要估值的 x 点，默认 1.5 2.6",
    )
    args = parser.parse_args()
    solve(args.queries)
