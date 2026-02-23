"""
例题 7.6（续例 7.5）：正态参数估计与 Q-Q 图检验

题意：
  假设例 7.5 的 84 个头颅宽度数据来自正态总体，
  估计该正态分布参数，并通过 Q-Q 图判断拟合效果。

运行示例：
  python ch07/ex07_06/solution.py
  python ch07/ex07_06/solution.py --alpha 0.05
  python ch07/ex07_06/solution.py --no-plot
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


def estimate_normal_params(data):
    """
    正态参数估计：
    - μ 的估计：样本均值
    - σ 的估计：给出 MLE 与无偏估计两种口径
    """
    n = data.size
    mu_hat = float(np.mean(data))
    sigma_mle = float(np.std(data, ddof=0))
    sigma_unbiased = float(np.std(data, ddof=1))
    var_mle = sigma_mle**2
    var_unbiased = sigma_unbiased**2
    return {
        "n": n,
        "mu_hat": mu_hat,
        "sigma_mle": sigma_mle,
        "sigma_unbiased": sigma_unbiased,
        "var_mle": var_mle,
        "var_unbiased": var_unbiased,
    }


def qq_points(data, mu_hat, sigma_hat):
    """
    构造 Q-Q 图点：
      z_i = Phi^{-1}((i-0.5)/n), x_(i) 为样本顺序统计量
    并对 x_(i)=a+b*z_i 做线性拟合。
    """
    x_sorted = np.sort(data)
    n = x_sorted.size
    p = (np.arange(1, n + 1, dtype=float) - 0.5) / n
    z = stats.norm.ppf(p)

    # 拟合线 x = a + b z（若正态拟合好，应接近线性）
    slope, intercept = np.polyfit(z, x_sorted, deg=1)
    x_fit = intercept + slope * z
    ss_res = float(np.sum((x_sorted - x_fit) ** 2))
    ss_tot = float(np.sum((x_sorted - np.mean(x_sorted)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0

    # 由估计参数直接给出的理论线
    x_theory = mu_hat + sigma_hat * z

    return {
        "z": z,
        "x_sorted": x_sorted,
        "x_fit": x_fit,
        "x_theory": x_theory,
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": r2,
    }


def goodness_of_fit_tests(data):
    """
    给出一个常用数值参考：Shapiro-Wilk 正态性检验。
    注：Q-Q 图才是本题主要求，检验仅作辅助说明。
    """
    stat, p_value = stats.shapiro(data)
    return float(stat), float(p_value)


def print_report(param_res, qq_res, shapiro_res, alpha):
    """打印参数与拟合效果。"""
    n = param_res["n"]
    mu_hat = param_res["mu_hat"]
    sigma_mle = param_res["sigma_mle"]
    sigma_unbiased = param_res["sigma_unbiased"]
    var_mle = param_res["var_mle"]
    var_unbiased = param_res["var_unbiased"]
    slope = qq_res["slope"]
    intercept = qq_res["intercept"]
    r2 = qq_res["r2"]
    sh_stat, sh_p = shapiro_res

    print("=== 例题 7.6 求解结果 ===")
    print(f"样本量 n = {n}")
    print("正态参数估计：")
    print(f"  mu_hat = x̄ = {mu_hat:.10f}")
    print(f"  sigma_hat (MLE, 分母 n) = {sigma_mle:.10f}")
    print(f"  sigma_hat (无偏, 分母 n-1) = {sigma_unbiased:.10f}")
    print(f"  var_hat (MLE) = {var_mle:.10f}")
    print(f"  var_hat (无偏) = {var_unbiased:.10f}")

    print("\nQ-Q 图线性拟合指标（x_(i)=a+b z_i）：")
    print(f"  intercept a = {intercept:.10f}")
    print(f"  slope b     = {slope:.10f}")
    print(f"  R^2         = {r2:.10f}")
    print("  （若拟合好，散点应接近直线，且 a≈mu_hat, b≈sigma_hat）")

    print("\n辅助检验（Shapiro-Wilk）：")
    print(f"  W统计量 = {sh_stat:.10f}")
    print(f"  p值 = {sh_p:.10f}")
    if sh_p > alpha:
        print(f"  在显著性水平 alpha={alpha:.4f} 下：未拒绝正态性假设。")
    else:
        print(f"  在显著性水平 alpha={alpha:.4f} 下：拒绝正态性假设。")


def plot_qq(qq_res):
    """绘制 Q-Q 图。"""
    z = qq_res["z"]
    x_sorted = qq_res["x_sorted"]
    x_fit = qq_res["x_fit"]
    x_theory = qq_res["x_theory"]

    plt.figure(figsize=(8.2, 6.0))
    plt.scatter(z, x_sorted, s=26, color="#1f77b4", alpha=0.85, label="样本分位点")
    plt.plot(z, x_theory, color="#d62728", linewidth=2.0, label="理论线（mu_hat, sigma_hat）")
    plt.plot(z, x_fit, color="#2ca02c", linestyle="--", linewidth=1.8, label="线性拟合线")
    plt.xlabel("标准正态理论分位点 z")
    plt.ylabel("样本分位点 x_(i) (mm)")
    plt.title("例题 7.6：正态 Q-Q 图")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(data, alpha=0.05, show_plot=True):
    """主流程。"""
    param_res = estimate_normal_params(data)
    # Q-Q 理论线通常用无偏标准差口径
    qq_res = qq_points(
        data=data,
        mu_hat=param_res["mu_hat"],
        sigma_hat=param_res["sigma_unbiased"],
    )
    shapiro_res = goodness_of_fit_tests(data)
    print_report(param_res, qq_res, shapiro_res, alpha=alpha)

    if show_plot:
        plot_qq(qq_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.6：正态参数估计与 Q-Q 图")
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Shapiro 辅助检验显著性水平，默认 0.05",
    )
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(data=DATA, alpha=args.alpha, show_plot=not args.no_plot)
