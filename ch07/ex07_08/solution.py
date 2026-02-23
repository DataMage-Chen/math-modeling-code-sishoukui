"""
例题 7.8：分组寿命数据的指数分布拟合优度检验

数据（n=300）：
  0<=t<=100      : 121
  100<t<=200     : 78
  200<t<=300     : 43
  t>300          : 58

检验：
  H0: T ~ Exp(lambda=0.005), t>=0
  alpha = 0.05

运行示例：
  python ch07/ex07_08/solution.py
  python ch07/ex07_08/solution.py --alpha 0.01
  python ch07/ex07_08/solution.py --lambda0 0.0045
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


N = 300
OBS = np.array([121, 78, 43, 58], dtype=float)
LABELS = ["0<=t<=100", "100<t<=200", "200<t<=300", "t>300"]


def expected_counts_exp(n, lam):
    """按给定指数分布参数 lambda 计算四组期望频数。"""
    p1 = 1.0 - np.exp(-lam * 100.0)  # [0,100]
    p2 = np.exp(-lam * 100.0) - np.exp(-lam * 200.0)  # (100,200]
    p3 = np.exp(-lam * 200.0) - np.exp(-lam * 300.0)  # (200,300]
    p4 = np.exp(-lam * 300.0)  # >300
    probs = np.array([p1, p2, p3, p4], dtype=float)
    exp = n * probs
    return probs, exp


def chi_square_gof(obs, exp, alpha, estimated_params=0):
    """卡方拟合优度检验统计量。"""
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


def print_report(alpha, lam, probs, exp, test_res):
    """打印报告。"""
    print("=== 例题 7.8 求解结果 ===")
    print("原假设 H0: 灯泡寿命 T 服从指数分布 f(t)=lambda*exp(-lambda*t), t>=0")
    print(f"给定参数: lambda0 = {lam:.10f}")
    print(f"显著性水平 alpha = {alpha:.4f}")

    print("\n各组观察频数与期望频数：")
    for lb, o, p, e in zip(LABELS, OBS, probs, exp):
        print(f"  {lb:>10}: 观察={o:8.4f}, 概率={p:.8f}, 期望={e:8.4f}")

    print("\n卡方检验统计量：")
    print(f"  chi2 = {test_res['stat']:.10f}")
    print(f"  df   = {test_res['df']}")
    print(f"  临界值 chi2_(1-alpha,df) = {test_res['crit']:.10f}")
    print(f"  p-value = {test_res['p_value']:.10f}")

    if test_res["reject"]:
        print("\n结论：拒绝 H0（该指数分布与样本不相容）。")
    else:
        print("\n结论：不拒绝 H0（样本与该指数分布相容）。")


def plot_result(obs, exp, lam):
    """绘制观察频数与期望频数对比图。"""
    x = np.arange(len(obs))
    w = 0.35
    plt.figure(figsize=(8.8, 5.4))
    plt.bar(x - w / 2, obs, width=w, color="#1f77b4", alpha=0.85, label="观察频数")
    plt.bar(x + w / 2, exp, width=w, color="#ff7f0e", alpha=0.85, label="指数分布期望频数")
    plt.xticks(x, LABELS)
    plt.ylabel("频数")
    plt.xlabel("寿命分组")
    plt.title(f"例题 7.8：指数分布拟合优度检验（lambda={lam:.4f}）")
    plt.grid(axis="y", alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(alpha=0.05, lambda0=0.005, show_plot=True):
    """主流程。"""
    if not (0 < alpha < 1):
        raise ValueError("alpha 需在 (0,1) 内。")
    if lambda0 <= 0:
        raise ValueError("lambda0 必须为正。")

    probs, exp = expected_counts_exp(N, lambda0)
    test_res = chi_square_gof(OBS, exp, alpha=alpha, estimated_params=0)
    print_report(alpha=alpha, lam=lambda0, probs=probs, exp=exp, test_res=test_res)

    if show_plot:
        plot_result(OBS, exp, lambda0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.8：指数分布拟合优度检验")
    parser.add_argument("--alpha", type=float, default=0.05, help="显著性水平，默认 0.05")
    parser.add_argument("--lambda0", type=float, default=0.005, help="原假设指数参数 lambda")
    parser.add_argument("--no-plot", action="store_true", help="仅输出结果，不绘图")
    args = parser.parse_args()

    solve(alpha=args.alpha, lambda0=args.lambda0, show_plot=not args.no_plot)
