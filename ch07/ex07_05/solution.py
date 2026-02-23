"""
例题 7.5：经验分布函数（ECDF）计算与绘图

题目给定 84 个伊特拉斯坎男子头颅最大宽度（mm）样本，
要求计算经验分布函数并画出图形。

运行示例：
  python ch07/ex07_05/solution.py
  python ch07/ex07_05/solution.py --query 135 140 145 150
  python ch07/ex07_05/solution.py --no-plot
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


DATA = [
    141, 148, 132, 138, 154, 142, 150, 146, 155, 158,
    150, 140, 147, 148, 144, 150, 149, 145, 149, 158,
    143, 141, 144, 144, 126, 140, 144, 142, 141, 140,
    145, 135, 147, 146, 141, 136, 140, 146, 142, 137,
    148, 154, 137, 139, 143, 140, 131, 143, 141, 149,
    148, 135, 148, 152, 143, 144, 141, 143, 147, 146,
    150, 132, 142, 142, 143, 153, 149, 146, 149, 138,
    142, 149, 142, 137, 134, 144, 146, 147, 140, 142,
    140, 137, 152, 145,
]


def ecdf(data):
    """返回经验分布函数的基础量：排序样本与对应累计概率。"""
    x_sorted = np.sort(np.asarray(data, dtype=float))
    n = x_sorted.size
    y = np.arange(1, n + 1, dtype=float) / n
    return x_sorted, y


def ecdf_at(x_sorted, query):
    """计算经验分布函数 F_n(x)=P(X<=x) 在若干查询点处的值。"""
    n = x_sorted.size
    idx = np.searchsorted(x_sorted, query, side="right")
    return idx / n


def print_report(data, x_sorted):
    """打印样本与 ECDF 关键结果。"""
    n = len(data)
    mean_val = float(np.mean(x_sorted))
    std_val = float(np.std(x_sorted, ddof=1))
    min_val = float(x_sorted[0])
    max_val = float(x_sorted[-1])

    print("=== 例题 7.5 求解结果 ===")
    print(f"样本量 n = {n}")
    print(f"最小值 = {min_val:.0f}, 最大值 = {max_val:.0f}")
    print(f"样本均值 = {mean_val:.6f}")
    print(f"样本标准差 = {std_val:.6f}")

    uniq, counts = np.unique(x_sorted, return_counts=True)
    cum_probs = np.cumsum(counts) / n
    print("\n经验分布函数关键台阶点（x, F_n(x)=P(X<=x)）：")
    for x_val, fn_val in zip(uniq, cum_probs):
        print(f"  x={int(x_val):3d}, F_n(x)={fn_val:.6f}")


def plot_ecdf(x_sorted, y):
    """绘制经验分布函数阶梯图。"""
    n = x_sorted.size

    # 构造更直观的右连续阶梯函数
    x_step = np.concatenate(([x_sorted[0] - 1], x_sorted))
    y_step = np.concatenate(([0.0], y))

    plt.figure(figsize=(9.2, 5.8))
    plt.step(x_step, y_step, where="post", color="#1f77b4", linewidth=2.1, label="经验分布函数 F_n(x)")
    plt.scatter(x_sorted, y, s=14, color="#d62728", alpha=0.7, label="样本阶梯点")
    plt.axhline(0.25, color="#666666", linestyle=":", linewidth=1.1)
    plt.axhline(0.50, color="#666666", linestyle=":", linewidth=1.1)
    plt.axhline(0.75, color="#666666", linestyle=":", linewidth=1.1)
    plt.ylim(-0.02, 1.02)
    plt.xlabel("头颅最大宽度 x (mm)")
    plt.ylabel("F_n(x)")
    plt.title("例题 7.5：头颅最大宽度的经验分布函数")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.show()


def solve(data, query=None, show_plot=True):
    """主流程。"""
    x_sorted, y = ecdf(data)
    print_report(data, x_sorted)

    if query:
        q = np.asarray(query, dtype=float)
        fq = ecdf_at(x_sorted, q)
        print("\n查询点的经验分布函数值：")
        for qi, fi in zip(q, fq):
            print(f"  F_n({qi:.4f}) = {fi:.6f}")

    if show_plot:
        plot_ecdf(x_sorted, y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题 7.5：经验分布函数计算与绘图")
    parser.add_argument(
        "--query",
        nargs="*",
        type=float,
        default=None,
        help="可选：查询若干 x 处的 F_n(x)",
    )
    parser.add_argument("--no-plot", action="store_true", help="仅输出数值，不绘图")
    args = parser.parse_args()

    solve(data=DATA, query=args.query, show_plot=not args.no_plot)
