"""
例题 7.14：样本中位数估计量 MSE 的 Bootstrap 估计

数据（单位：kcal/mol）：
136.3, 136.6, 135.8, 135.4, 134.7, 135.0, 134.1, 143.3, 147.8, 148.8,
134.8, 135.2, 134.9, 149.5, 141.2, 135.4, 134.8, 135.8, 135.0, 133.7, 134.4,
134.9, 134.8, 134.5, 134.3, 135.2

任务：
以样本中位数 M 估计总体中位数 theta，计算 MSE=E[(M-theta)^2] 的 Bootstrap 估计。

运行示例：
  python ch07/ex07_14/solution.py
  python ch07/ex07_14/solution.py --n-bootstrap 200000 --seed 2026
  python ch07/ex07_14/solution.py --plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy matplotlib"
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
        136.3, 136.6, 135.8, 135.4, 134.7, 135.0, 134.1, 143.3, 147.8, 148.8,
        134.8, 135.2, 134.9, 149.5, 141.2, 135.4, 134.8, 135.8, 135.0, 133.7, 134.4,
        134.9, 134.8, 134.5, 134.3, 135.2,
    ],
    dtype=float,
)


def bootstrap_median_mse(data, n_bootstrap=100000, seed=2026):
    """
    Bootstrap 估计中位数估计量 M 的 MSE。

    记原样本中位数为 M_hat，作为 theta 的替代。
    生成 B 个有放回重抽样样本，计算其样本中位数 M*_b。

    MSE_boot = (1/B) * sum_b (M*_b - M_hat)^2
    """
    if n_bootstrap < 1000:
        raise ValueError("n_bootstrap 建议至少为 1000。")

    n = data.size
    rng = np.random.default_rng(seed)
    m_hat = float(np.median(data))

    resamples = rng.choice(data, size=(n_bootstrap, n), replace=True)
    med_star = np.median(resamples, axis=1)

    err = med_star - m_hat
    mse_boot = float(np.mean(err**2))
    var_boot = float(np.var(med_star, ddof=1))
    bias_boot = float(np.mean(med_star) - m_hat)

    # MSE 估计本身的 Monte Carlo 标准误（反映模拟误差，不是统计推断误差）
    mse_mc_se = float(np.std(err**2, ddof=1) / np.sqrt(n_bootstrap))

    return {
        "n": n,
        "m_hat": m_hat,
        "med_star": med_star,
        "mse_boot": mse_boot,
        "var_boot": var_boot,
        "bias_boot": bias_boot,
        "mse_from_decomp": var_boot + bias_boot**2,
        "mse_mc_se": mse_mc_se,
    }


def print_report(data, res, n_bootstrap, seed):
    """打印计算结果。"""
    print("=== 例题 7.14 求解结果 ===")
    print(f"样本量 n = {res['n']}")
    print(f"样本均值 = {np.mean(data):.10f}")
    print(f"样本标准差 = {np.std(data, ddof=1):.10f}")
    print(f"样本中位数 M_hat = {res['m_hat']:.10f}")

    print("\nBootstrap 设置：")
    print(f"  B = {n_bootstrap}")
    print(f"  seed = {seed}")

    print("\nBootstrap 对 MSE 的估计：")
    print(f"  MSE_boot = E*[(M* - M_hat)^2] = {res['mse_boot']:.10f}")
    print(f"  Var_boot(M*) = {res['var_boot']:.10f}")
    print(f"  Bias_boot(M*) = {res['bias_boot']:.10f}")
    print(f"  Var + Bias^2 = {res['mse_from_decomp']:.10f}")
    print(f"  MSE 估计的 Monte Carlo 标准误 = {res['mse_mc_se']:.10f}")


def plot_distribution(med_star, m_hat):
    """绘制 Bootstrap 中位数分布。"""
    plt.figure(figsize=(9.0, 5.4))
    plt.hist(med_star, bins="auto", density=True, color="#1f77b4", alpha=0.78, edgecolor="white")
    plt.axvline(m_hat, color="#d62728", linestyle="--", linewidth=1.8, label="原样本中位数 M_hat")
    plt.xlabel("Bootstrap 样本中位数 M*")
    plt.ylabel("密度")
    plt.title("例题 7.14：Bootstrap 中位数分布")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(data, n_bootstrap=100000, seed=2026, show_plot=False):
    """主流程。"""
    res = bootstrap_median_mse(data, n_bootstrap=n_bootstrap, seed=seed)
    print_report(data, res, n_bootstrap=n_bootstrap, seed=seed)
    if show_plot:
        plot_distribution(res["med_star"], res["m_hat"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.14：中位数 MSE 的 Bootstrap 估计")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100000,
        help="Bootstrap 重抽样次数 B，默认 100000",
    )
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument("--plot", action="store_true", help="绘制 Bootstrap 中位数分布图")
    args = parser.parse_args()

    solve(data=DATA, n_bootstrap=args.n_bootstrap, seed=args.seed, show_plot=args.plot)
