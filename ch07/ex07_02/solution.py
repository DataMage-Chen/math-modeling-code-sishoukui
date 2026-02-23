"""
例题 7.2：正态总体均值的单侧置信区间

样本数据（单位：h）：
1050, 1100, 1120, 1250, 1280

运行示例：
  python ch07/ex07_02/solution.py
  python ch07/ex07_02/solution.py --side lower
  python ch07/ex07_02/solution.py --side upper --confidence 0.99
  python ch07/ex07_02/solution.py --data 1000 1020 980 1100 1040
"""

import argparse
import math
import statistics

try:
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install scipy"
    ) from exc


DEFAULT_DATA = [1050, 1100, 1120, 1250, 1280]


def one_sided_ci_mean(data, confidence=0.95):
    """
    计算正态总体均值（方差未知）的一侧置信区间基础量。

    对于置信水平 gamma：
      下侧区间: [xbar - t_{gamma,df} * s/sqrt(n), +inf)
      上侧区间: (-inf, xbar + t_{gamma,df} * s/sqrt(n)]
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence 需在 (0,1) 内。")
    if len(data) < 2:
        raise ValueError("样本量至少为 2。")

    n = len(data)
    df = n - 1
    x_bar = statistics.mean(data)
    s = statistics.stdev(data)
    t_crit = stats.t.ppf(confidence, df)
    se = s / math.sqrt(n)
    margin = t_crit * se

    return {
        "n": n,
        "df": df,
        "x_bar": x_bar,
        "s": s,
        "confidence": confidence,
        "t_crit": t_crit,
        "se": se,
        "margin": margin,
        "lower_bound": x_bar - margin,
        "upper_bound": x_bar + margin,
    }


def print_report(result, side):
    """打印结果。"""
    print("=== 例题 7.2 求解结果 ===")
    print("方法: t 分布单侧区间（总体方差未知，正态总体）")
    print(f"样本量 n = {result['n']}, 自由度 df = {result['df']}")
    print(f"样本均值 x̄ = {result['x_bar']:.10f}")
    print(f"样本标准差 s = {result['s']:.10f}")
    print(f"置信水平 = {result['confidence']:.4f}")
    print(f"单侧临界值 t_(gamma,df) = {result['t_crit']:.10f}")
    print(f"标准误 se = {result['se']:.10f}")
    print(f"误差限 = {result['margin']:.10f}")

    lower = result["lower_bound"]
    upper = result["upper_bound"]

    if side == "lower":
        print(f"\n下侧单侧置信区间: [ {lower:.10f}, +inf )")
    elif side == "upper":
        print(f"\n上侧单侧置信区间: ( -inf, {upper:.10f} ]")
    else:
        print(f"\n下侧单侧置信区间: [ {lower:.10f}, +inf )")
        print(f"上侧单侧置信区间: ( -inf, {upper:.10f} ]")


def solve(data, confidence=0.95, side="both"):
    """主流程。"""
    result = one_sided_ci_mean(data=data, confidence=confidence)
    print_report(result, side=side)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.2：总体均值单侧置信区间")
    parser.add_argument(
        "--data",
        nargs="*",
        type=float,
        default=None,
        help="样本数据列表；不传则使用题目给定的 5 个样本",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="置信水平（0~1），默认 0.95",
    )
    parser.add_argument(
        "--side",
        type=str,
        choices=["lower", "upper", "both"],
        default="both",
        help="输出下侧/上侧/两侧单侧区间，默认 both",
    )
    args = parser.parse_args()

    sample_data = DEFAULT_DATA if not args.data else args.data
    solve(data=sample_data, confidence=args.confidence, side=args.side)
