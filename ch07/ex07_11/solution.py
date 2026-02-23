"""
例题 7.11（续例 7.5）：Kolmogorov-Smirnov 正态性检验

任务：
  用 K-S 检验法检验例 7.5 的 84 个数据是否服从正态分布（alpha=0.05）。

说明：
  对“参数未知且由样本估计”的正态性检验，严格来说应使用 Lilliefors 校正。
  本实现同时给出：
    1) 经典 K-S（参数当作已知）结果；
    2) Lilliefors 检验结果（statsmodels 可用时）。

运行示例：
  python ch07/ex07_11/solution.py
  python ch07/ex07_11/solution.py --alpha 0.01
  python ch07/ex07_11/solution.py --no-plot
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


def fit_normal(data):
    """估计正态参数（均值+样本标准差口径）。"""
    mu = float(np.mean(data))
    sigma = float(np.std(data, ddof=1))
    return mu, sigma


def ks_test_with_fitted_normal(data):
    """经典 K-S：将拟合后的 mu,sigma 代入正态 CDF。"""
    mu, sigma = fit_normal(data)
    z = (data - mu) / sigma
    ks_stat, p_value = stats.kstest(z, "norm")
    return {
        "mu": mu,
        "sigma": sigma,
        "ks_stat": float(ks_stat),
        "p_value": float(p_value),
    }


def lilliefors_test_if_available(data):
    """
    若可用则执行 Lilliefors（更适用于“参数由样本估计”的正态性检验）。
    """
    try:
        from statsmodels.stats.diagnostic import lilliefors
    except Exception:
        return None

    stat, p_value = lilliefors(data, dist="norm")
    return {
        "stat": float(stat),
        "p_value": float(p_value),
    }


def ecdf(data):
    """经验分布函数坐标。"""
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1, dtype=float) / n
    return x, y


def print_report(alpha, res_ks, res_lf):
    """打印检验报告。"""
    print("=== 例题 7.11 求解结果 ===")
    print(f"样本量 n = {DATA.size}")
    print(f"拟合正态参数: mu_hat={res_ks['mu']:.10f}, sigma_hat={res_ks['sigma']:.10f}")
    print(f"显著性水平 alpha = {alpha:.4f}")

    print("\n[经典 K-S 检验]")
    print(f"  D统计量 = {res_ks['ks_stat']:.10f}")
    print(f"  p-value = {res_ks['p_value']:.10f}")
    if res_ks["p_value"] < alpha:
        print("  结论：拒绝正态性假设（按经典 K-S 判定）")
    else:
        print("  结论：不拒绝正态性假设（按经典 K-S 判定）")

    print("\n[Lilliefors 校正检验]")
    if res_lf is None:
        print("  未检测到 statsmodels，跳过 Lilliefors 检验。")
        print("  如需该检验可安装：pip install statsmodels")
    else:
        print(f"  D统计量 = {res_lf['stat']:.10f}")
        print(f"  p-value = {res_lf['p_value']:.10f}")
        if res_lf["p_value"] < alpha:
            print("  结论：拒绝正态性假设（按 Lilliefors 判定）")
        else:
            print("  结论：不拒绝正态性假设（按 Lilliefors 判定）")


def plot_ecdf_vs_cdf(data, mu, sigma):
    """绘制经验分布函数与拟合正态 CDF 对比图。"""
    x_ecdf, y_ecdf = ecdf(data)
    x_grid = np.linspace(np.min(data) - 3, np.max(data) + 3, 400)
    cdf_fit = stats.norm.cdf(x_grid, loc=mu, scale=sigma)

    plt.figure(figsize=(9.2, 5.8))
    plt.step(
        np.concatenate(([x_ecdf[0] - 1], x_ecdf)),
        np.concatenate(([0.0], y_ecdf)),
        where="post",
        linewidth=2.0,
        color="#1f77b4",
        label="经验分布函数 F_n(x)",
    )
    plt.plot(x_grid, cdf_fit, color="#d62728", linewidth=2.0, label="拟合正态 CDF")
    plt.xlabel("x (mm)")
    plt.ylabel("分布函数值")
    plt.title("例题 7.11：ECDF 与拟合正态分布 CDF 对比")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(alpha=0.05, show_plot=True):
    """主流程。"""
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")

    res_ks = ks_test_with_fitted_normal(DATA)
    res_lf = lilliefors_test_if_available(DATA)
    print_report(alpha, res_ks, res_lf)

    if show_plot:
        plot_ecdf_vs_cdf(DATA, mu=res_ks["mu"], sigma=res_ks["sigma"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.11：K-S 正态性检验")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平，默认 0.05")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, show_plot=not args.no_plot)
