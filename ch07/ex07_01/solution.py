"""
例题 7.1：正态总体均值的置信区间估计

题目数据（单位：g）：
506, 508, 499, 503, 504, 510, 497, 512,
514, 505, 493, 496, 506, 502, 509, 496

运行示例：
  python ch07/ex07_01/solution.py
  python ch07/ex07_01/solution.py --confidence 0.99
  python ch07/ex07_01/solution.py --sigma-known 6.2
  python ch07/ex07_01/solution.py --data 500 501 499 503 504 498
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


DEFAULT_DATA = [
    506, 508, 499, 503, 504, 510, 497, 512,
    514, 505, 493, 496, 506, 502, 509, 496,
]


def confidence_interval_mean(data, confidence=0.95, sigma_known=None):
    """
    计算总体均值的置信区间。

    - 若 sigma_known 为 None：使用 t 区间（总体方差未知）；
    - 若 sigma_known 给定：使用 z 区间（总体方差已知）。
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence 需在 (0,1) 内。")
    if sigma_known is not None and sigma_known <= 0:
        raise ValueError("sigma_known 必须为正。")

    n = len(data)
    if n < 2:
        raise ValueError("样本量至少为 2。")

    x_bar = statistics.mean(data)
    alpha = 1.0 - confidence

    if sigma_known is None:
        s = statistics.stdev(data)
        df = n - 1
        crit = stats.t.ppf(1.0 - alpha / 2.0, df)
        se = s / math.sqrt(n)
        method = "t区间（总体方差未知）"
        extra = {"s": s, "df": df}
    else:
        crit = stats.norm.ppf(1.0 - alpha / 2.0)
        se = sigma_known / math.sqrt(n)
        method = "z区间（总体方差已知）"
        extra = {"sigma_known": sigma_known}

    margin = crit * se
    low = x_bar - margin
    high = x_bar + margin

    return {
        "n": n,
        "x_bar": x_bar,
        "alpha": alpha,
        "confidence": confidence,
        "method": method,
        "critical_value": crit,
        "standard_error": se,
        "margin": margin,
        "ci_low": low,
        "ci_high": high,
        "extra": extra,
    }


def print_report(result):
    """打印结果报告。"""
    print("=== 例题 7.1 求解结果 ===")
    print(f"方法: {result['method']}")
    print(f"样本量 n = {result['n']}")
    print(f"样本均值 x̄ = {result['x_bar']:.10f}")
    if "s" in result["extra"]:
        print(f"样本标准差 s = {result['extra']['s']:.10f}")
        print(f"自由度 df = {result['extra']['df']}")
    if "sigma_known" in result["extra"]:
        print(f"已知总体标准差 sigma = {result['extra']['sigma_known']:.10f}")

    print(f"置信水平 = {result['confidence']:.4f}")
    print(f"临界值 = {result['critical_value']:.10f}")
    print(f"标准误 = {result['standard_error']:.10f}")
    print(f"误差限 = {result['margin']:.10f}")
    print(
        f"均值置信区间: [{result['ci_low']:.10f}, {result['ci_high']:.10f}]"
    )


def solve(data, confidence=0.95, sigma_known=None):
    """主流程。"""
    result = confidence_interval_mean(
        data=data,
        confidence=confidence,
        sigma_known=sigma_known,
    )
    print_report(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.1：正态总体均值置信区间")
    parser.add_argument(
        "--data",
        nargs="*",
        type=float,
        default=None,
        help="样本数据列表；不传则使用题目给定 16 个样本",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.95,
        help="置信水平（0~1），默认 0.95",
    )
    parser.add_argument(
        "--sigma-known",
        type=float,
        default=None,
        help="可选：已知总体标准差 sigma；若不提供则使用 t 区间",
    )
    args = parser.parse_args()

    sample_data = DEFAULT_DATA if not args.data else args.data
    solve(
        data=sample_data,
        confidence=args.confidence,
        sigma_known=args.sigma_known,
    )
