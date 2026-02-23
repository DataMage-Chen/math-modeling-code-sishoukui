"""
例题 7.10（续例 7.5）：检验样本是否来自正态总体

数据：续用例 7.5 的 84 个头颅最大宽度样本（mm）。
任务：在 alpha=0.1 下检验“样本来自某正态总体”。

实现方法：
  使用卡方拟合优度检验，并先由样本估计正态参数 mu、sigma。
  为保证期望频数充分，采用“等理论概率分组”（默认 8 组）。

运行示例：
  python ch07/ex07_10/solution.py
  python ch07/ex07_10/solution.py --alpha 0.05
  python ch07/ex07_10/solution.py --num-groups 7
  python ch07/ex07_10/solution.py --no-plot
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


DATA = np.array(
    [
        141, 148, 132, 138, 154, 142, 150, 146, 155, 158,
        150, 140, 147, 148, 144, 150, 149, 145, 149, 158,
        143, 141, 144, 144, 126, 140, 144, 142, 141, 140,
        145, 135, 147, 146, 141, 136, 140, 146, 142, 137,
        148, 154, 137, 139, 143, 140, 131, 143, 141, 149,
        148, 135, 148, 152, 143, 144, 141, 143, 147, 146,
        150, 132, 142, 142, 143, 153, 149, 146, 149, 138,
        142, 149, 142, 137, 134, 144, 146, 147, 140, 142,
        140, 137, 152, 145,
    ],
    dtype=float,
)


def fit_normal_params(data):
    """估计正态参数（采用样本均值 + MLE 标准差）。"""
    mu_hat = float(np.mean(data))
    sigma_hat = float(np.std(data, ddof=0))
    return mu_hat, sigma_hat


def build_equal_prob_bins(mu, sigma, m):
    """
    按拟合正态分布的等概率分组构造区间边界：
      每组理论概率均为 1/m。
    """
    probs = np.linspace(0.0, 1.0, m + 1)
    edges = stats.norm.ppf(probs, loc=mu, scale=sigma)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def chi_square_gof_normal(data, alpha=0.1, num_groups=8):
    """
    卡方拟合优度检验：
      H0: 数据来自某正态总体 N(mu, sigma^2)（mu、sigma 由样本估计）
    """
    if data.size < 20:
        raise ValueError("样本量过小，卡方拟合优度检验不稳定。")
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")
    if num_groups < 5:
        raise ValueError("分组数建议不少于 5。")

    n = data.size
    mu_hat, sigma_hat = fit_normal_params(data)
    edges = build_equal_prob_bins(mu_hat, sigma_hat, num_groups)

    obs, _ = np.histogram(data, bins=edges)
    obs = obs.astype(float)
    exp = np.full(num_groups, n / num_groups, dtype=float)

    # 估计参数个数 p=2（mu, sigma）
    p = 2
    df = num_groups - 1 - p
    if df <= 0:
        raise ValueError("自由度<=0，请增加分组数。")

    stat = float(np.sum((obs - exp) ** 2 / exp))
    crit = float(stats.chi2.ppf(1.0 - alpha, df))
    p_value = float(1.0 - stats.chi2.cdf(stat, df))
    reject = stat > crit

    labels = []
    for i in range(num_groups):
        left = edges[i]
        right = edges[i + 1]
        if np.isneginf(left):
            labels.append(f"(-inf,{right:.2f}]")
        elif np.isposinf(right):
            labels.append(f"({left:.2f},+inf)")
        else:
            labels.append(f"({left:.2f},{right:.2f}]")

    return {
        "n": n,
        "mu_hat": mu_hat,
        "sigma_hat": sigma_hat,
        "edges": edges,
        "labels": labels,
        "obs": obs,
        "exp": exp,
        "num_groups": num_groups,
        "df": df,
        "stat": stat,
        "crit": crit,
        "p_value": p_value,
        "reject": reject,
        "alpha": alpha,
    }


def print_report(res):
    """打印报告。"""
    print("=== 例题 7.10 求解结果 ===")
    print("原假设 H0: 数据来自某正态总体 N(mu, sigma^2)")
    print(f"样本量 n = {res['n']}")
    print(f"拟合参数: mu_hat = {res['mu_hat']:.10f}, sigma_hat = {res['sigma_hat']:.10f}")
    print(f"分组数 m = {res['num_groups']}（等理论概率分组）")
    print(f"显著性水平 alpha = {res['alpha']:.4f}")

    print("\n分组观察频数与期望频数：")
    for lb, o, e in zip(res["labels"], res["obs"], res["exp"]):
        print(f"  {lb:>18}: 观察={o:7.3f}, 期望={e:7.3f}")

    print("\n卡方检验统计量：")
    print(f"  chi2 = {res['stat']:.10f}")
    print(f"  df   = {res['df']}  (m-1-p, p=2)")
    print(f"  临界值 chi2_(1-alpha,df) = {res['crit']:.10f}")
    print(f"  p-value = {res['p_value']:.10f}")

    if res["reject"]:
        print("\n结论：拒绝 H0（在该显著性水平下，不支持正态总体假设）。")
    else:
        print("\n结论：不拒绝 H0（在该显著性水平下，样本与正态总体相容）。")


def plot_result(res):
    """绘制分组频数对比图。"""
    x = np.arange(res["num_groups"])
    w = 0.36
    plt.figure(figsize=(11.2, 5.4))
    plt.bar(x - w / 2, res["obs"], width=w, color="#1f77b4", alpha=0.85, label="观察频数")
    plt.bar(x + w / 2, res["exp"], width=w, color="#ff7f0e", alpha=0.85, label="正态期望频数")
    plt.xticks(x, res["labels"], rotation=25, ha="right")
    plt.ylabel("频数")
    plt.xlabel("分组区间")
    plt.title("例题 7.10：正态总体拟合优度（卡方检验）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(alpha=0.1, num_groups=8, show_plot=True):
    """主流程。"""
    res = chi_square_gof_normal(DATA, alpha=alpha, num_groups=num_groups)
    print_report(res)
    if show_plot:
        plot_result(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.10：正态总体拟合优度检验")
    parser.add_argument("--alpha", type=float, default=0.1, help="显著性水平，默认 0.1")
    parser.add_argument("--num-groups", type=int, default=8, help="分组数（默认 8）")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, num_groups=args.num_groups, show_plot=not args.no_plot)
