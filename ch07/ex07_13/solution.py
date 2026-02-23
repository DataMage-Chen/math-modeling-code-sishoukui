"""
例题 7.13：样本中位数标准误差的 Bootstrap 估计

数据（基金年回报率）：
18.2, 9.5, 12.0, 21.1, 10.2

任务：
以样本中位数作为总体中位数 theta 的估计，
并用 Bootstrap 估计该中位数估计量的标准误差。

运行示例：
  python ch07/ex07_13/solution.py
  python ch07/ex07_13/solution.py --n-bootstrap 200000 --seed 2026
  python ch07/ex07_13/solution.py --plot
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


DATA = np.array([18.2, 9.5, 12.0, 21.1, 10.2], dtype=float)


def bootstrap_median_se(data, n_bootstrap=100000, seed=2026):
    """
    Bootstrap 过程：
    1) 从原样本有放回抽样，生成 B 组重抽样样本；
    2) 每组计算样本中位数；
    3) 这些中位数的样本标准差，作为中位数估计量标准误差的 Bootstrap 估计。
    """
    if n_bootstrap < 1000:
        raise ValueError("n_bootstrap 建议至少 1000。")

    n = data.size
    rng = np.random.default_rng(seed)
    resamples = rng.choice(data, size=(n_bootstrap, n), replace=True)
    medians = np.median(resamples, axis=1)
    se_hat = float(np.std(medians, ddof=1))
    return medians, se_hat


def print_report(data, medians, se_hat, n_bootstrap, seed):
    """打印核心结果。"""
    sample_median = float(np.median(data))
    sample_mean = float(np.mean(data))
    sample_std = float(np.std(data, ddof=1))

    # 顺带给一个 bootstrap 百分位区间（非题目必须，仅参考）
    ci_low, ci_high = np.percentile(medians, [2.5, 97.5])

    print("=== 例题 7.13 求解结果 ===")
    print(f"样本数据: {data.tolist()}")
    print(f"样本量 n = {data.size}")
    print(f"样本均值 = {sample_mean:.10f}")
    print(f"样本标准差 = {sample_std:.10f}")
    print(f"样本中位数（theta 的点估计）= {sample_median:.10f}")

    print("\nBootstrap 设置：")
    print(f"  重抽样次数 B = {n_bootstrap}")
    print(f"  随机种子 seed = {seed}")

    print("\nBootstrap 结果：")
    print(f"  中位数估计量标准误差 se_boot = {se_hat:.10f}")
    print(f"  中位数的 95% 百分位区间（参考）= [{ci_low:.10f}, {ci_high:.10f}]")


def plot_bootstrap_distribution(medians):
    """绘制 Bootstrap 中位数分布。"""
    plt.figure(figsize=(8.8, 5.4))
    plt.hist(medians, bins="auto", density=True, color="#1f77b4", alpha=0.75, edgecolor="white")
    plt.xlabel("Bootstrap 样本中位数")
    plt.ylabel("密度")
    plt.title("例题 7.13：Bootstrap 中位数分布")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.show()


def solve(data, n_bootstrap=100000, seed=2026, show_plot=False):
    """主流程。"""
    medians, se_hat = bootstrap_median_se(
        data=data,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    print_report(
        data=data,
        medians=medians,
        se_hat=se_hat,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    if show_plot:
        plot_bootstrap_distribution(medians)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.13：中位数标准误差的 Bootstrap 估计")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=100000,
        help="Bootstrap 重抽样次数 B，默认 100000",
    )
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument("--plot", action="store_true", help="绘制 Bootstrap 中位数分布图")
    args = parser.parse_args()

    solve(
        data=DATA,
        n_bootstrap=args.n_bootstrap,
        seed=args.seed,
        show_plot=args.plot,
    )
