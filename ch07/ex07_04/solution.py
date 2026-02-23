"""
例题 7.4（续例 7.3）：方差相等时两总体均值差的置信区间

数据（同例 7.3）：
- 金球：6.683, 6.681, 6.676, 6.678, 6.679, 6.672
- 铅球：6.661, 6.661, 6.667, 6.667, 6.664

要求：
在方差相等假设下，求 μ1-μ2 在置信水平 0.90 下的区间。

运行示例：
  python ch07/ex07_04/solution.py
  python ch07/ex07_04/solution.py --confidence 0.95
  python ch07/ex07_04/solution.py --data1 1 2 3 --data2 0.5 1.5 2.5
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


DEFAULT_DATA_1 = [6.683, 6.681, 6.676, 6.678, 6.679, 6.672]  # 金球
DEFAULT_DATA_2 = [6.661, 6.661, 6.667, 6.667, 6.664]  # 铅球


def ci_two_means_equal_var(data1, data2, confidence=0.90):
    """
    两正态总体均值差区间（方差未知但相等）：
      (x1-x2) ± t_(1-a/2, n1+n2-2) * Sp * sqrt(1/n1 + 1/n2)
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence 需在 (0,1) 内。")
    if len(data1) < 2 or len(data2) < 2:
        raise ValueError("两组样本量都至少为 2。")

    n1, n2 = len(data1), len(data2)
    x1, x2 = statistics.mean(data1), statistics.mean(data2)
    s1_2 = statistics.variance(data1)
    s2_2 = statistics.variance(data2)

    df = n1 + n2 - 2
    alpha = 1.0 - confidence
    sp2 = ((n1 - 1) * s1_2 + (n2 - 1) * s2_2) / df
    sp = math.sqrt(sp2)
    se = sp * math.sqrt(1.0 / n1 + 1.0 / n2)
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)

    diff = x1 - x2
    margin = t_crit * se
    low = diff - margin
    high = diff + margin

    return {
        "n1": n1,
        "n2": n2,
        "x1": x1,
        "x2": x2,
        "s1_2": s1_2,
        "s2_2": s2_2,
        "df": df,
        "confidence": confidence,
        "sp2": sp2,
        "sp": sp,
        "se": se,
        "t_crit": t_crit,
        "diff": diff,
        "margin": margin,
        "low": low,
        "high": high,
    }


def print_report(res):
    """打印结果。"""
    print("=== 例题 7.4 求解结果 ===")
    print("方法：两独立正态总体，方差未知但相等（合并方差 t 区间）")
    print(f"n1={res['n1']}, n2={res['n2']}, df={res['df']}")
    print(f"x̄1={res['x1']:.10f}, x̄2={res['x2']:.10f}")
    print(f"s1^2={res['s1_2']:.10f}, s2^2={res['s2_2']:.10f}")
    print(f"合并方差 Sp^2={res['sp2']:.10f}, Sp={res['sp']:.10f}")
    print(f"标准误 SE={res['se']:.10f}")
    print(f"置信水平={res['confidence']:.4f}, t临界值={res['t_crit']:.10f}")
    print(f"均值差估计 x̄1-x̄2={res['diff']:.10f}")
    print(f"误差限={res['margin']:.10f}")
    print(f"(μ1-μ2) 的置信区间: [{res['low']:.10f}, {res['high']:.10f}]")


def solve(data1, data2, confidence=0.90):
    """主流程。"""
    res = ci_two_means_equal_var(data1, data2, confidence=confidence)
    print_report(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.4：方差相等时两均值差区间估计")
    parser.add_argument(
        "--data1",
        nargs="*",
        type=float,
        default=None,
        help="总体1样本（默认金球数据）",
    )
    parser.add_argument(
        "--data2",
        nargs="*",
        type=float,
        default=None,
        help="总体2样本（默认铅球数据）",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
        help="置信水平（0~1），默认0.90",
    )
    args = parser.parse_args()

    data1 = DEFAULT_DATA_1 if not args.data1 else args.data1
    data2 = DEFAULT_DATA_2 if not args.data2 else args.data2
    solve(data1=data1, data2=data2, confidence=args.confidence)
