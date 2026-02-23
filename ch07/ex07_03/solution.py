"""
例题 7.3：正态总体均值与方差的区间估计（两组实验数据）

数据（单位：10^-11 m^3·kg^-1·s^-2）：
1) 金球：6.683, 6.681, 6.676, 6.678, 6.679, 6.672
2) 铅球：6.661, 6.661, 6.667, 6.667, 6.664

要求：
- 分别求两组数据的 μ 在置信水平 0.9 下的区间；
- 分别求两组数据的 σ^2 在置信水平 0.9 下的区间。

运行示例：
  python ch07/ex07_03/solution.py
  python ch07/ex07_03/solution.py --confidence 0.95
  python ch07/ex07_03/solution.py --data-gold 6.68 6.67 6.69 --data-lead 6.66 6.67 6.65
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


DEFAULT_GOLD = [6.683, 6.681, 6.676, 6.678, 6.679, 6.672]
DEFAULT_LEAD = [6.661, 6.661, 6.667, 6.667, 6.664]


def estimate_intervals_normal_unknown(data, confidence=0.9):
    """
    正态总体 N(μ,σ²)，μ、σ²未知：
    - μ 的区间：t 区间
    - σ² 的区间：卡方区间
    """
    if not (0 < confidence < 1):
        raise ValueError("confidence 需在 (0,1) 内。")
    if len(data) < 2:
        raise ValueError("样本量至少为 2。")

    n = len(data)
    df = n - 1
    alpha = 1.0 - confidence

    x_bar = statistics.mean(data)
    s2 = statistics.variance(data)  # 无偏样本方差
    s = math.sqrt(s2)

    # μ 区间（t）
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    margin_mu = t_crit * s / math.sqrt(n)
    mu_low = x_bar - margin_mu
    mu_high = x_bar + margin_mu

    # σ² 区间（chi-square）
    chi2_low_quant = stats.chi2.ppf(alpha / 2.0, df)
    chi2_high_quant = stats.chi2.ppf(1.0 - alpha / 2.0, df)
    sigma2_low = df * s2 / chi2_high_quant
    sigma2_high = df * s2 / chi2_low_quant

    return {
        "n": n,
        "df": df,
        "alpha": alpha,
        "confidence": confidence,
        "x_bar": x_bar,
        "s": s,
        "s2": s2,
        "t_crit": t_crit,
        "mu_low": mu_low,
        "mu_high": mu_high,
        "chi2_low_quant": chi2_low_quant,
        "chi2_high_quant": chi2_high_quant,
        "sigma2_low": sigma2_low,
        "sigma2_high": sigma2_high,
    }


def print_one_group(name, data, result):
    """打印单组结果。"""
    print(f"\n--- {name} ---")
    print(f"样本数据: {data}")
    print(f"n={result['n']}, df={result['df']}, 置信水平={result['confidence']:.4f}")
    print(f"样本均值 x̄={result['x_bar']:.10f}")
    print(f"样本标准差 s={result['s']:.10f}")
    print(f"样本方差 s^2={result['s2']:.10f}")
    print(f"t 临界值 t_(1-alpha/2,df)={result['t_crit']:.10f}")
    print(
        f"μ 的 {result['confidence']:.2f} 置信区间: "
        f"[{result['mu_low']:.10f}, {result['mu_high']:.10f}]"
    )
    print(
        f"σ^2 的 {result['confidence']:.2f} 置信区间: "
        f"[{result['sigma2_low']:.10f}, {result['sigma2_high']:.10f}]"
    )


def solve(data_gold, data_lead, confidence=0.9):
    """主流程。"""
    res_gold = estimate_intervals_normal_unknown(data_gold, confidence=confidence)
    res_lead = estimate_intervals_normal_unknown(data_lead, confidence=confidence)

    print("=== 例题 7.3 求解结果 ===")
    print("模型假设：总体服从 N(μ,σ²)，且 μ、σ²均未知。")
    print_one_group("金球实验", data_gold, res_gold)
    print_one_group("铅球实验", data_lead, res_lead)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.3：μ 与 σ² 的区间估计")
    parser.add_argument(
        "--data-gold",
        nargs="*",
        type=float,
        default=None,
        help="金球实验样本数据；不传则使用题目默认数据",
    )
    parser.add_argument(
        "--data-lead",
        nargs="*",
        type=float,
        default=None,
        help="铅球实验样本数据；不传则使用题目默认数据",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="置信水平（0~1），默认 0.9",
    )
    args = parser.parse_args()

    data_gold = DEFAULT_GOLD if not args.data_gold else args.data_gold
    data_lead = DEFAULT_LEAD if not args.data_lead else args.data_lead

    solve(data_gold=data_gold, data_lead=data_lead, confidence=args.confidence)
