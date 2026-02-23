"""
例题 7.7：检验印刷错误个数是否服从泊松分布（卡方拟合优度检验）

已知 100 页书中每页印刷错误个数频数：
  错误数:      0   1   2   3   4   5   6  >=7
  页数:       36  40  19   2   0   2   1   0

问题：在显著性水平 alpha=0.05 下，是否可认为每页错误数服从泊松分布？

运行示例：
  python ch07/ex07_07/solution.py
  python ch07/ex07_07/solution.py --alpha 0.01
  python ch07/ex07_07/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy scipy matplotlib"
    ) from exc


# 让 matplotlib 尽量正确显示中文和负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


# 题目给定频数（总页数 100）
COUNTS = {
    0: 36,
    1: 40,
    2: 19,
    3: 2,
    4: 0,
    5: 2,
    6: 1,
    ">=7": 0,
}


def estimate_lambda(counts):
    """
    估计泊松参数 lambda。
    本题 >=7 频数为 0，因此样本均值可直接由 0~6 类精确算出。
    """
    total_pages = sum(counts.values())
    total_errors = sum(k * counts[k] for k in range(7))
    if counts[">=7"] != 0:
        raise ValueError("本实现默认 >=7 频数为 0；若非 0 需额外给定尾部代表值。")
    return total_errors / total_pages, total_pages


def grouped_observed_expected(counts, lam, n):
    """
    为满足卡方检验“期望频数不宜过小”要求，将类别合并为：
      0, 1, 2, >=3
    """
    obs = np.array(
        [
            counts[0],
            counts[1],
            counts[2],
            counts[3] + counts[4] + counts[5] + counts[6] + counts[">=7"],
        ],
        dtype=float,
    )

    exp = np.array(
        [
            n * stats.poisson.pmf(0, lam),
            n * stats.poisson.pmf(1, lam),
            n * stats.poisson.pmf(2, lam),
            n * (1.0 - stats.poisson.cdf(2, lam)),
        ],
        dtype=float,
    )
    labels = ["0", "1", "2", ">=3"]
    return labels, obs, exp


def chi_square_gof(obs, exp, alpha, estimated_params=1):
    """卡方拟合优度检验（手工计算统计量与自由度）。"""
    stat = float(np.sum((obs - exp) ** 2 / exp))
    m = obs.size
    df = m - 1 - estimated_params
    p_value = float(1.0 - stats.chi2.cdf(stat, df))
    crit = float(stats.chi2.ppf(1.0 - alpha, df))
    reject = stat > crit
    return {
        "stat": stat,
        "df": df,
        "p_value": p_value,
        "crit": crit,
        "reject": reject,
    }


def print_report(lam, n, labels, obs, exp, test_res, alpha):
    """打印检验报告。"""
    print("=== 例题 7.7 求解结果 ===")
    print("原假设 H0: 每页印刷错误数 X ~ Poisson(lambda)")
    print("备择假设 H1: X 不服从该泊松分布")
    print(f"样本量 n = {n}")
    print(f"lambda 的极大似然估计（样本均值）: lambda_hat = {lam:.10f}")
    print(f"显著性水平 alpha = {alpha:.4f}")

    print("\n合并分组（用于卡方检验）观察频数 vs 期望频数：")
    for lb, o, e in zip(labels, obs, exp):
        print(f"  组 {lb:>3}: 观察={o:8.4f}, 期望={e:8.4f}")

    print("\n卡方检验统计量：")
    print(f"  chi2 = {test_res['stat']:.10f}")
    print(f"  df   = {test_res['df']}")
    print(f"  临界值 chi2_(1-alpha,df) = {test_res['crit']:.10f}")
    print(f"  p-value = {test_res['p_value']:.10f}")

    if test_res["reject"]:
        print("\n结论：拒绝 H0（在该显著性水平下，不支持泊松分布假设）。")
    else:
        print("\n结论：不拒绝 H0（在该显著性水平下，可认为与泊松分布相容）。")


def plot_result(labels, obs, exp, lam):
    """绘制分组频数比较图。"""
    x = np.arange(len(labels))
    w = 0.36

    plt.figure(figsize=(8.6, 5.4))
    plt.bar(x - w / 2, obs, width=w, color="#1f77b4", alpha=0.85, label="观察频数")
    plt.bar(x + w / 2, exp, width=w, color="#ff7f0e", alpha=0.85, label="泊松期望频数")
    plt.xticks(x, labels)
    plt.ylabel("频数")
    plt.xlabel("错误个数分组")
    plt.title(f"例题 7.7：泊松拟合优度检验（lambda_hat={lam:.3f}）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(alpha=0.05, show_plot=True):
    """主流程。"""
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")

    lam, n = estimate_lambda(COUNTS)
    labels, obs, exp = grouped_observed_expected(COUNTS, lam=lam, n=n)
    test_res = chi_square_gof(obs, exp, alpha=alpha, estimated_params=1)
    print_report(lam, n, labels, obs, exp, test_res, alpha)

    if show_plot:
        plot_result(labels, obs, exp, lam=lam)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.7：泊松分布拟合优度检验")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平，默认 0.05")
    parser.add_argument("--no-plot", action="store_true", help="仅输出数值，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, show_plot=not args.no_plot)
