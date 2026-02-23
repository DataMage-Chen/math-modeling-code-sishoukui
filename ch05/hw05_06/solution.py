"""
习题 5.6：非线性函数参数拟合（lsqcurvefit 思路 + fit 思路）

运行示例：
  python ch05/hw05_06/solution.py
  python ch05/hw05_06/solution.py --init-a 1.0 --init-b 0.05 --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import curve_fit, least_squares
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


A_TRUE = 1.1
B_TRUE = 0.01
X_DATA = np.arange(1, 21, dtype=float)


def g_model(x, a, b):
    """题目给定模型函数。"""
    x = np.asarray(x, dtype=float)
    numerator = 10.0 * a
    denominator = 10.0 * b + (a - 10.0 * b) * np.exp(-a * np.sin(x))
    return numerator / denominator


def calc_metrics(y_true, y_hat):
    """计算 SSE、RMSE、R^2。"""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    residual = y_true - y_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return residual, sse, rmse, r2


def fit_by_lsqcurvefit_style(x_data, y_data, init_a, init_b):
    """对应 MATLAB lsqcurvefit 思路：直接最小化残差向量。"""

    def residual_func(params):
        a, b = params
        return g_model(x_data, a, b) - y_data

    result = least_squares(
        residual_func,
        x0=np.array([init_a, init_b], dtype=float),
        bounds=([1e-8, 1e-8], [np.inf, np.inf]),
        method="trf",
    )

    a_hat, b_hat = float(result.x[0]), float(result.x[1])
    y_hat = g_model(x_data, a_hat, b_hat)
    _, sse, rmse, r2 = calc_metrics(y_data, y_hat)

    return {
        "a": a_hat,
        "b": b_hat,
        "y_hat": y_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nfev": int(result.nfev),
    }


def fit_by_fit_style(x_data, y_data, init_a, init_b):
    """对应 MATLAB fittype + fit 思路：自定义函数后做非线性回归。"""
    popt, _ = curve_fit(
        g_model,
        x_data,
        y_data,
        p0=[init_a, init_b],
        bounds=([1e-8, 1e-8], [np.inf, np.inf]),
        maxfev=20000,
    )

    a_hat, b_hat = float(popt[0]), float(popt[1])
    y_hat = g_model(x_data, a_hat, b_hat)
    _, sse, rmse, r2 = calc_metrics(y_data, y_hat)

    return {
        "a": a_hat,
        "b": b_hat,
        "y_hat": y_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
    }


def print_method_result(title, result):
    """打印单个方法的结果。"""
    da = result["a"] - A_TRUE
    db = result["b"] - B_TRUE

    print(f"\n--- {title} ---")
    print(f"参数估计: a={result['a']:.12f}, b={result['b']:.12f}")
    print(f"参数误差: da={da:+.3e}, db={db:+.3e}")
    print(f"SSE={result['sse']:.12e}, RMSE={result['rmse']:.12e}, R^2={result['r2']:.12f}")

    if "success" in result:
        print(f"优化状态: success={result['success']}, status={result['status']}, nfev={result['nfev']}")
        print(f"优化信息: {result['message']}")


def plot_result(y_data, res_lsq, res_fit):
    """绘制观测点与两种拟合曲线。"""
    x_dense = np.linspace(float(np.min(X_DATA)), float(np.max(X_DATA)), 1200)
    y_true_dense = g_model(x_dense, A_TRUE, B_TRUE)
    y_lsq_dense = g_model(x_dense, res_lsq["a"], res_lsq["b"])
    y_fit_dense = g_model(x_dense, res_fit["a"], res_fit["b"])

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    ax.scatter(X_DATA, y_data, color="#1f77b4", s=45, label="模拟观测点")
    ax.plot(x_dense, y_true_dense, color="#2ca02c", linewidth=2.0, label="真函数")
    ax.plot(x_dense, y_lsq_dense, color="#d62728", linewidth=1.8, linestyle="--", label="least_squares 拟合")
    ax.plot(x_dense, y_fit_dense, color="#9467bd", linewidth=1.8, linestyle="-.", label="curve_fit 拟合")

    ax.set_title("习题 5.6：两种非线性拟合方法对比")
    ax.set_xlabel("x")
    ax.set_ylabel("g(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(init_a=1.0, init_b=0.05, show_plot=True):
    y_data = g_model(X_DATA, A_TRUE, B_TRUE)

    print("=== 习题 5.6 求解结果 ===")
    print(f"真实参数: a={A_TRUE}, b={B_TRUE}")
    print(f"初始猜测: a0={init_a}, b0={init_b}")
    print("观测点: x=1,...,20（由真函数生成）")

    res_lsq = fit_by_lsqcurvefit_style(X_DATA, y_data, init_a=init_a, init_b=init_b)
    res_fit = fit_by_fit_style(X_DATA, y_data, init_a=init_a, init_b=init_b)

    print_method_result("方法(1) lsqcurvefit 思路（least_squares）", res_lsq)
    print_method_result("方法(2) fittype+fit 思路（curve_fit）", res_fit)

    if show_plot:
        plot_result(y_data, res_lsq, res_fit)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.6 非线性参数拟合")
    parser.add_argument("--init-a", type=float, default=1.0, help="参数 a 的初始值，默认 1.0")
    parser.add_argument("--init-b", type=float, default=0.05, help="参数 b 的初始值，默认 0.05")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(init_a=args.init_a, init_b=args.init_b, show_plot=not args.no_plot)

