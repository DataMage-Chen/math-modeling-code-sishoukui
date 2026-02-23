"""
例题 7.12：烘干温度对抗弯强度影响检验

数据（表 7.8）：
120℃组（n=9）:
  41.5, 42.0, 40.0, 42.5, 42.0, 42.2, 42.7, 42.1, 41.4
160℃组（n=6）:
  41.2, 41.8, 42.4, 41.6, 41.7, 41.3

默认在 alpha=0.05 下输出：
1) 主检验：Welch 两样本 t 检验；
2) 补充检验：秩和检验（Mann-Whitney U / Wilcoxon rank-sum）。

运行示例：
  python ch07/ex07_12/solution.py
  python ch07/ex07_12/solution.py --equal-var
  python ch07/ex07_12/solution.py --alpha 0.01 --no-plot
"""

import argparse
import math

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


DATA_120 = np.array([41.5, 42.0, 40.0, 42.5, 42.0, 42.2, 42.7, 42.1, 41.4], dtype=float)
DATA_160 = np.array([41.2, 41.8, 42.4, 41.6, 41.7, 41.3], dtype=float)


def summarize(x):
    """返回样本统计量。"""
    return {
        "n": x.size,
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)),
        "var": float(np.var(x, ddof=1)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def two_sample_t_test(x1, x2, alpha=0.05, equal_var=False):
    """
    两独立样本 t 检验，并计算均值差 (mu1-mu2) 的双侧置信区间。
    - equal_var=False: Welch
    - equal_var=True : 合并方差 t
    """
    n1, n2 = x1.size, x2.size
    m1, m2 = float(np.mean(x1)), float(np.mean(x2))
    v1, v2 = float(np.var(x1, ddof=1)), float(np.var(x2, ddof=1))
    diff = m1 - m2

    if equal_var:
        df = n1 + n2 - 2
        sp2 = ((n1 - 1) * v1 + (n2 - 1) * v2) / df
        se = math.sqrt(sp2 * (1.0 / n1 + 1.0 / n2))
        method = "合并方差 t 检验"
    else:
        se = math.sqrt(v1 / n1 + v2 / n2)
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / den
        method = "Welch t 检验"

    t_stat = diff / se
    p_value = 2.0 * (1.0 - stats.t.cdf(abs(t_stat), df))
    t_crit = stats.t.ppf(1.0 - alpha / 2.0, df)
    ci_low = diff - t_crit * se
    ci_high = diff + t_crit * se
    reject = p_value < alpha

    return {
        "method": method,
        "diff": diff,
        "se": se,
        "df": float(df),
        "t_stat": float(t_stat),
        "p_value": float(p_value),
        "alpha": alpha,
        "reject": reject,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
    }


def rank_sum_tests(x1, x2, alpha=0.05):
    """
    秩和检验（双侧，Mann-Whitney U / Wilcoxon rank-sum 等价口径）。
    """
    n1, n2 = x1.size, x2.size
    mw_cc = stats.mannwhitneyu(
        x1, x2, alternative="two-sided", method="asymptotic", use_continuity=True
    )
    # 教材常见“秩和统计量 W”口径（通常取较小样本组）
    combined = np.concatenate([x1, x2])
    ranks = stats.rankdata(combined, method="average")
    r1 = float(np.sum(ranks[:n1]))
    r2 = float(np.sum(ranks[n1:]))
    if n1 <= n2:
        w_small = r1
        small_label = "120℃组"
    else:
        w_small = r2
        small_label = "160℃组"

    # 由 U 统计量决定显著性
    u1 = float(mw_cc.statistic)  # x1 相对 x2 的 U
    u_min = min(u1, n1 * n2 - u1)

    return {
        "u_stat": u1,
        "u_min": float(u_min),
        "mw_p": float(mw_cc.pvalue),
        "mw_reject": float(mw_cc.pvalue) < alpha,
        "w_small": w_small,
        "w_small_label": small_label,
        "r1": r1,
        "r2": r2,
    }


def diagnostics(x1, x2):
    """给出常见前提检验结果（辅助信息）。"""
    sh1 = stats.shapiro(x1)
    sh2 = stats.shapiro(x2)
    lev = stats.levene(x1, x2, center="median")
    return {
        "shapiro_120": (float(sh1.statistic), float(sh1.pvalue)),
        "shapiro_160": (float(sh2.statistic), float(sh2.pvalue)),
        "levene": (float(lev.statistic), float(lev.pvalue)),
    }


def print_report(alpha, s120, s160, t_res, rank_res, diag):
    """打印结果。"""
    print("=== 例题 7.12 求解结果 ===")
    print("原假设 H0: 两温度组总体位置参数相同（均值/中位数差异不显著）")
    print("备择假设 H1: 两温度组总体位置参数不同")
    print(f"显著性水平 alpha = {alpha:.4f}")

    print("\n样本描述统计：")
    print(
        f"  120℃: n={s120['n']}, mean={s120['mean']:.6f}, std={s120['std']:.6f}, "
        f"min={s120['min']:.3f}, max={s120['max']:.3f}"
    )
    print(
        f"  160℃: n={s160['n']}, mean={s160['mean']:.6f}, std={s160['std']:.6f}, "
        f"min={s160['min']:.3f}, max={s160['max']:.3f}"
    )

    print("\n辅助诊断（仅供参考）：")
    print(
        f"  Shapiro 120℃: W={diag['shapiro_120'][0]:.6f}, p={diag['shapiro_120'][1]:.6f}"
    )
    print(
        f"  Shapiro 160℃: W={diag['shapiro_160'][0]:.6f}, p={diag['shapiro_160'][1]:.6f}"
    )
    print(
        f"  Levene 方差齐性: stat={diag['levene'][0]:.6f}, p={diag['levene'][1]:.6f}"
    )

    print(f"\n主检验方法: {t_res['method']}")
    print(f"  均值差估计 (120℃-160℃) = {t_res['diff']:.10f}")
    print(f"  t统计量 = {t_res['t_stat']:.10f}")
    print(f"  自由度 df = {t_res['df']:.10f}")
    print(f"  p-value = {t_res['p_value']:.10f}")
    print(
        f"  均值差 {(1-t_res['alpha']):.2%} 置信区间 = "
        f"[{t_res['ci_low']:.10f}, {t_res['ci_high']:.10f}]"
    )
    if t_res["reject"]:
        print("  t检验结论：拒绝 H0（均值存在显著差异）")
    else:
        print("  t检验结论：不拒绝 H0（均值差异不显著）")

    print("\n补充：秩和检验（非参数）")
    print(f"  秩和 R120 = {rank_res['r1']:.10f}, R160 = {rank_res['r2']:.10f}")
    print(
        f"  教材常用 W（较小样本组 {rank_res['w_small_label']}）= "
        f"{rank_res['w_small']:.10f}"
    )
    print(f"  Mann-Whitney U = {rank_res['u_stat']:.10f}, U_min = {rank_res['u_min']:.10f}")
    print(f"  p-value = {rank_res['mw_p']:.10f}")
    if rank_res["mw_reject"]:
        print("  秩和检验结论：拒绝 H0（位置参数差异显著）")
    else:
        print("  秩和检验结论：不拒绝 H0（位置参数差异不显著）")


def plot_result(x1, x2):
    """绘制箱线图 + 抖动散点。"""
    plt.figure(figsize=(8.2, 5.4))

    data = [x1, x2]
    labels = ["120℃", "160℃"]
    # Matplotlib>=3.9 使用 tick_labels，旧版本仍使用 labels。
    try:
        plt.boxplot(data, tick_labels=labels, widths=0.45, showmeans=True)
    except TypeError:
        plt.boxplot(data, labels=labels, widths=0.45, showmeans=True)

    rng = np.random.default_rng(2026)
    for i, arr in enumerate(data, start=1):
        jitter = rng.uniform(-0.08, 0.08, size=arr.size)
        plt.scatter(np.full(arr.size, i) + jitter, arr, color="#1f77b4", alpha=0.75, s=30)

    plt.ylabel("抗弯强度")
    plt.title("例题 7.12：两温度组抗弯强度对比")
    plt.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.show()


def solve(alpha=0.05, equal_var=False, show_plot=True):
    """主流程。"""
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")

    s120 = summarize(DATA_120)
    s160 = summarize(DATA_160)
    t_res = two_sample_t_test(DATA_120, DATA_160, alpha=alpha, equal_var=equal_var)
    rank_res = rank_sum_tests(DATA_120, DATA_160, alpha=alpha)
    diag = diagnostics(DATA_120, DATA_160)

    print_report(alpha, s120, s160, t_res, rank_res, diag)
    if show_plot:
        plot_result(DATA_120, DATA_160)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.12：均值检验 + 秩和检验")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平，默认 0.05")
    parser.add_argument(
        "--equal-var",
        action="store_true",
        help="若指定，则 t 检验使用合并方差；默认 Welch",
    )
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, equal_var=args.equal_var, show_plot=not args.no_plot)
