"""
例题 5.3：等距节点多项式插值（Runge 函数）。

运行：
  python ch05/ex05_03/solution.py
  python ch05/ex05_03/solution.py --n-list 6 8 10 --num-grid 2001
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import BarycentricInterpolator
except ImportError as exc:
    raise SystemExit(
        "缺少依赖，请先安装：\n"
        "  pip install numpy scipy matplotlib"
    ) from exc

# 让 matplotlib 在不同系统下尽量正确显示中文与负号
plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans CJK SC",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


def runge(x):
    """Runge 函数 f(x)=1/(1+x^2)。"""
    return 1.0 / (1.0 + x * x)


def build_interpolator(n, a, b):
    """构造 n 次等距节点插值器。"""
    x_nodes = np.linspace(a, b, n + 1)
    y_nodes = runge(x_nodes)
    interp = BarycentricInterpolator(x_nodes, y_nodes)
    return x_nodes, y_nodes, interp


def analyze(n_list, a, b, num_grid):
    """对多个 n 计算插值曲线和误差指标。"""
    x_grid = np.linspace(a, b, num_grid)
    y_true = runge(x_grid)

    results = []
    for n in n_list:
        x_nodes, y_nodes, interp = build_interpolator(n, a, b)
        y_pred = interp(x_grid)
        abs_err = np.abs(y_pred - y_true)

        max_err = float(np.max(abs_err))
        rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
        end_left = float(abs(y_pred[0] - y_true[0]))
        end_right = float(abs(y_pred[-1] - y_true[-1]))

        results.append(
            {
                "n": n,
                "x_nodes": x_nodes,
                "y_nodes": y_nodes,
                "y_pred": y_pred,
                "abs_err": abs_err,
                "max_err": max_err,
                "rmse": rmse,
                "end_left": end_left,
                "end_right": end_right,
            }
        )
    return x_grid, y_true, results


def print_report(results):
    """打印误差对比报告。"""
    print("=== 例题 5.3 误差分析结果 ===")
    print("n 次插值多项式在区间 [-5,5] 的误差指标：")
    for r in results:
        print(
            f"  n={r['n']:>2d}: "
            f"max|e(x)|={r['max_err']:.10e}, "
            f"RMSE={r['rmse']:.10e}, "
            f"|e(-5)|={r['end_left']:.10e}, "
            f"|e(5)|={r['end_right']:.10e}"
        )

    print("\n提示：")
    print("  在等距节点下，n 增大后最大误差不一定单调减小；")
    print("  端点附近可能出现振荡（Runge 现象）。")


def plot_results(x_grid, y_true, results):
    """绘制函数对比图。"""
    fig, ax1 = plt.subplots(1, 1, figsize=(10, 5.5))

    # 真函数 + 不同 n 的插值曲线
    ax1.plot(x_grid, y_true, "k-", linewidth=2.2, label="f(x)=1/(1+x^2)")
    colors = ["#d95f02", "#1b9e77", "#7570b3", "#e7298a", "#66a61e"]
    for idx, r in enumerate(results):
        color = colors[idx % len(colors)]
        ax1.plot(
            x_grid,
            r["y_pred"],
            color=color,
            linewidth=1.8,
            label=f"P_{r['n']}(x)",
        )
        ax1.scatter(
            r["x_nodes"],
            r["y_nodes"],
            color=color,
            s=18,
            alpha=0.7,
            zorder=3,
        )

    ax1.set_title("Runge 函数与等距节点插值多项式对比")
    ax1.set_xlabel("x")
    ax1.set_ylabel("函数值")
    ax1.grid(alpha=0.3)
    ax1.legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="例题5.3：等距节点插值误差分析")
    parser.add_argument(
        "--n-list",
        type=int,
        nargs="+",
        default=[6, 8, 10],
        help="插值次数列表，默认 6 8 10",
    )
    parser.add_argument(
        "--a",
        type=float,
        default=-5.0,
        help="区间左端点，默认 -5",
    )
    parser.add_argument(
        "--b",
        type=float,
        default=5.0,
        help="区间右端点，默认 5",
    )
    parser.add_argument(
        "--num-grid",
        type=int,
        default=2001,
        help="评估网格点数，默认 2001",
    )
    args = parser.parse_args()

    x_grid, y_true, results = analyze(
        n_list=args.n_list,
        a=args.a,
        b=args.b,
        num_grid=args.num_grid,
    )
    print_report(results)
    plot_results(x_grid, y_true, results)


if __name__ == "__main__":
    main()
