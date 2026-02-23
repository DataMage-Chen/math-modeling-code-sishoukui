"""
例题 6.18：基于 Logistic 模型的美国人口预测（预测 2010 年）。

运行示例：
  python ch06/ex06_18/solution.py
  python ch06/ex06_18/solution.py --fit-mode free-p0
  python ch06/ex06_18/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit
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


YEARS = np.array(
    [
        1790, 1800, 1810, 1820, 1830, 1840, 1850, 1860,
        1870, 1880, 1890, 1900, 1910, 1920, 1930, 1940,
        1950, 1960, 1970, 1980, 1990, 2000,
    ],
    dtype=float,
)
POP = np.array(
    [
        3.9, 5.3, 7.2, 9.6, 12.9, 17.1, 23.2, 31.4,
        38.6, 50.2, 62.9, 76.0, 92.0, 106.5, 123.2, 131.7,
        150.7, 179.3, 204.0, 226.5, 251.4, 281.4,
    ],
    dtype=float,
)

BASE_YEAR = 1790.0
P0 = float(POP[0])


def logistic_general(t, k, r, b):
    """三参数 Logistic 形式。"""
    t = np.asarray(t, dtype=float)
    return k / (1.0 + b * np.exp(-r * t))


def logistic_fixed_p0(t, k, r):
    """固定 P(0)=P0 的两参数 Logistic 形式。"""
    t = np.asarray(t, dtype=float)
    b = (k - P0) / P0
    return k / (1.0 + b * np.exp(-r * t))


def calc_metrics(y_true, y_hat):
    """计算 SSE、RMSE、R^2。"""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    residual = y_true - y_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def fit_model(fit_mode):
    """
    拟合 Logistic 参数。
    fit_mode:
      - fixed-p0: 固定 P(0)=3.9，只拟合 K,r（默认）
      - free-p0 : 拟合 K,r,b 三参数
    """
    t_data = YEARS - BASE_YEAR
    max_pop = float(np.max(POP))

    if fit_mode == "fixed-p0":
        # K 必须大于当前最大观测人口，r>0
        lower = [max_pop + 1e-6, 1e-8]
        upper = [10000.0, 2.0]
        p0 = [400.0, 0.03]
        popt, _ = curve_fit(
            logistic_fixed_p0,
            t_data,
            POP,
            p0=p0,
            bounds=(lower, upper),
            maxfev=50000,
        )
        k_hat, r_hat = [float(v) for v in popt]
        b_hat = float((k_hat - P0) / P0)
        y_hat = logistic_fixed_p0(t_data, k_hat, r_hat)
        model_func = lambda t: logistic_fixed_p0(t, k_hat, r_hat)
    elif fit_mode == "free-p0":
        # 三参数拟合：K>max_pop, r>0, b>0
        lower = [max_pop + 1e-6, 1e-8, 1e-8]
        upper = [10000.0, 2.0, 1e6]
        p0 = [400.0, 0.03, 50.0]
        popt, _ = curve_fit(
            logistic_general,
            t_data,
            POP,
            p0=p0,
            bounds=(lower, upper),
            maxfev=100000,
        )
        k_hat, r_hat, b_hat = [float(v) for v in popt]
        y_hat = logistic_general(t_data, k_hat, r_hat, b_hat)
        model_func = lambda t: logistic_general(t, k_hat, r_hat, b_hat)
    else:
        raise ValueError("fit_mode 仅支持 fixed-p0 或 free-p0")

    sse, rmse, r2 = calc_metrics(POP, y_hat)
    return {
        "fit_mode": fit_mode,
        "k": k_hat,
        "r": r_hat,
        "b": b_hat,
        "y_hat": y_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "model_func": model_func,
    }


def print_report(result, predict_year):
    """打印拟合与预测结果。"""
    t_predict = predict_year - BASE_YEAR
    p_predict = float(result["model_func"](t_predict))

    print("=== 例题 6.18 求解结果（Logistic 人口模型） ===")
    print(f"拟合模式: {result['fit_mode']}")
    print(f"时间基准: t = year - {int(BASE_YEAR)}")
    print("\n拟合参数：")
    print(f"  K = {result['k']:.12f} (百万)")
    print(f"  r = {result['r']:.12f} (1/年)")
    print(f"  b = {result['b']:.12f}")

    if result["fit_mode"] == "fixed-p0":
        print(f"  固定初值 P(0)=3.9（对应 {int(BASE_YEAR)} 年）")
    else:
        p0_implied = result["k"] / (1.0 + result["b"])
        print(f"  拟合隐含初值 P(0)=K/(1+b)={p0_implied:.12f}")

    print("\n拟合指标：")
    print(f"  SSE  = {result['sse']:.12f}")
    print(f"  RMSE = {result['rmse']:.12f}")
    print(f"  R^2  = {result['r2']:.12f}")

    print("\n预测结果：")
    print(f"  预测 {int(predict_year)} 年人口 = {p_predict:.6f} 百万")

    print("\n样本点拟合误差（前 6 条）：")
    residual = POP - result["y_hat"]
    for i in range(min(6, YEARS.size)):
        print(
            f"  year={int(YEARS[i])}, "
            f"obs={POP[i]:8.3f}, fit={result['y_hat'][i]:8.3f}, resid={residual[i]:+.3f}"
        )

    return p_predict


def plot_result(result, predict_year, p_predict):
    """绘制观测点、拟合曲线和预测点。"""
    years_dense = np.linspace(float(np.min(YEARS)), float(predict_year), 900)
    t_dense = years_dense - BASE_YEAR
    pop_dense = result["model_func"](t_dense)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.5, 7.5), sharex=True)

    ax1.scatter(YEARS, POP, color="#1f77b4", s=45, label="观测数据")
    ax1.plot(years_dense, pop_dense, color="#d62728", linewidth=2.0, label="Logistic 拟合曲线")
    ax1.scatter([predict_year], [p_predict], color="#2ca02c", marker="x", s=80, label=f"{int(predict_year)} 预测点")
    ax1.set_ylabel("人口（百万）")
    ax1.set_title("美国人口 Logistic 拟合与预测")
    ax1.grid(alpha=0.3)
    ax1.legend()

    residual = POP - result["y_hat"]
    ax2.plot(YEARS, residual, color="#9467bd", marker="o", linewidth=1.5, label="残差（obs-fit）")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("年份")
    ax2.set_ylabel("残差（百万）")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(fit_mode="fixed-p0", predict_year=2010.0, show_plot=True):
    result = fit_model(fit_mode=fit_mode)
    p_predict = print_report(result, predict_year=predict_year)
    if show_plot:
        plot_result(result, predict_year=predict_year, p_predict=p_predict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.18 Logistic 人口预测")
    parser.add_argument(
        "--fit-mode",
        type=str,
        choices=["fixed-p0", "free-p0"],
        default="fixed-p0",
        help="拟合模式：fixed-p0（默认）或 free-p0",
    )
    parser.add_argument("--predict-year", type=float, default=2010.0, help="预测年份，默认 2010")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        fit_mode=args.fit_mode,
        predict_year=args.predict_year,
        show_plot=not args.no_plot,
    )

