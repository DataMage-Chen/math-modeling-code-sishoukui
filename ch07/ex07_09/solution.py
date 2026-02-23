"""
例题 7.9：正态分布拟合优度检验（参数已知）

题意：
  某大学 200 名一年级学生数学成绩分组频数如下：
    20<x<=30: 5
    30<x<=40: 15
    40<x<=50: 30
    50<x<=60: 51
    60<x<=70: 60
    70<x<=80: 23
    80<x<=90: 10
    90<x<=100: 6
  检验样本是否来自 N(60, 15^2)，显著性水平 alpha=0.1。

运行示例：
  python ch07/ex07_09/solution.py
  python ch07/ex07_09/solution.py --alpha 0.05
  python ch07/ex07_09/solution.py --mu0 62 --sigma0 14
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


N = 200
INTERVALS = [
    (20, 30),
    (30, 40),
    (40, 50),
    (50, 60),
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100),
]
OBS = np.array([5, 15, 30, 51, 60, 23, 10, 6], dtype=float)


def interval_probs_normal(intervals, mu, sigma):
    """计算每个分组区间的理论概率 P(a<X<=b)。"""
    probs = []
    for a, b in intervals:
        p = stats.norm.cdf((b - mu) / sigma) - stats.norm.cdf((a - mu) / sigma)
        probs.append(p)
    return np.array(probs, dtype=float)


def expected_counts(n, probs):
    """计算期望频数。"""
    return n * probs


def chi_square_gof(obs, exp, alpha, estimated_params=0):
    """卡方拟合优度检验。"""
    stat = float(np.sum((obs - exp) ** 2 / exp))
    m = obs.size
    df = m - 1 - estimated_params
    crit = float(stats.chi2.ppf(1.0 - alpha, df))
    p_value = float(1.0 - stats.chi2.cdf(stat, df))
    reject = stat > crit
    return {
        "stat": stat,
        "df": df,
        "crit": crit,
        "p_value": p_value,
        "reject": reject,
    }


def print_report(alpha, mu0, sigma0, probs, exp, test_res):
    """打印检验报告。"""
    print("=== 例题 7.9 求解结果 ===")
    print("原假设 H0: 成绩 X ~ N(mu0, sigma0^2)")
    print(f"给定参数: mu0={mu0:.6f}, sigma0={sigma0:.6f}")
    print(f"显著性水平 alpha={alpha:.4f}")

    print("\n各组观察频数与期望频数：")
    for (a, b), o, p, e in zip(INTERVALS, OBS, probs, exp):
        print(f"  {a:>3}<x<={b:<3}: 观察={o:8.4f}, 概率={p:.8f}, 期望={e:8.4f}")

    print("\n卡方检验统计量：")
    print(f"  chi2 = {test_res['stat']:.10f}")
    print(f"  df   = {test_res['df']}")
    print(f"  临界值 chi2_(1-alpha,df) = {test_res['crit']:.10f}")
    print(f"  p-value = {test_res['p_value']:.10f}")

    if test_res["reject"]:
        print("\n结论：拒绝 H0（样本与该正态分布不相容）。")
    else:
        print("\n结论：不拒绝 H0（样本与该正态分布相容）。")


def plot_result(obs, exp, mu0, sigma0):
    """绘制观察频数与期望频数对比。"""
    labels = [f"{a}-{b}" for a, b in INTERVALS]
    x = np.arange(len(labels))
    w = 0.36

    plt.figure(figsize=(10, 5.6))
    plt.bar(x - w / 2, obs, width=w, color="#1f77b4", alpha=0.85, label="观察频数")
    plt.bar(x + w / 2, exp, width=w, color="#ff7f0e", alpha=0.85, label="正态期望频数")
    plt.xticks(x, labels)
    plt.ylabel("频数")
    plt.xlabel("分数组间")
    plt.title(f"例题 7.9：N({mu0:.1f},{sigma0:.1f}^2) 拟合优度检验")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(alpha=0.1, mu0=60.0, sigma0=15.0, show_plot=True):
    """主流程。"""
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")
    if sigma0 <= 0:
        raise ValueError("sigma0 必须为正。")

    probs = interval_probs_normal(INTERVALS, mu=mu0, sigma=sigma0)
    exp = expected_counts(N, probs)
    test_res = chi_square_gof(OBS, exp, alpha=alpha, estimated_params=0)
    print_report(alpha, mu0, sigma0, probs, exp, test_res)

    if show_plot:
        plot_result(OBS, exp, mu0, sigma0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.9：正态分布拟合优度检验（参数已知）")
    parser.add_argument("--alpha", type=float, default=0.1, help="显著性水平，默认 0.1")
    parser.add_argument("--mu0", type=float, default=60.0, help="原假设均值 mu0，默认 60")
    parser.add_argument("--sigma0", type=float, default=15.0, help="原假设标准差 sigma0，默认 15")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, mu0=args.mu0, sigma0=args.sigma0, show_plot=not args.no_plot)
