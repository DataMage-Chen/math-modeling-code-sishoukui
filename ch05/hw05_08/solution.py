"""
习题 5.8：插值、最小二乘多项式拟合、正态分布非线性拟合

运行示例：
  python ch05/hw05_08/solution.py
  python ch05/hw05_08/solution.py --poly-degree auto --max-degree 10 --no-plot
  python ch05/hw05_08/solution.py --poly-degree 6
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator, interp1d
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


# 表 5.17 数据
X_DATA = np.array(
    [
        -2.0, -1.7, -1.4, -1.1, -0.8, -0.5, -0.2, 0.1,
        0.4, 0.7, 1.0, 1.3, 1.6, 1.9, 2.2, 2.5,
        2.8, 3.1, 3.4, 3.7, 4.0, 4.3, 4.6, 4.9,
    ],
    dtype=float,
)

Y_DATA = np.array(
    [
        0.1029, 0.1174, 0.1316, 0.1448, 0.1566, 0.1662, 0.1733, 0.1775,
        0.1785, 0.1764, 0.1711, 0.1630, 0.1526, 0.1402, 0.1266, 0.1122,
        0.0977, 0.0835, 0.0702, 0.0588, 0.0479, 0.0373, 0.0291, 0.0224,
    ],
    dtype=float,
)


def calc_metrics(y_true, y_hat):
    """计算 SSE、RMSE、R^2。"""
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    residual = y_true - y_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2, residual


def build_interpolators():
    """构造三种插值函数。"""
    linear_interp = interp1d(X_DATA, Y_DATA, kind="linear")
    spline_interp = CubicSpline(X_DATA, Y_DATA, bc_type="not-a-knot")
    pchip_interp = PchipInterpolator(X_DATA, Y_DATA)
    return {
        "linear": linear_interp,
        "spline": spline_interp,
        "pchip": pchip_interp,
    }


def interpolation_summary(interp_dict, x_dense):
    """给出插值方法的简要数值比较。"""
    y_min_obs = float(np.min(Y_DATA))
    y_max_obs = float(np.max(Y_DATA))
    summary = {}

    for name, func in interp_dict.items():
        y_dense = np.asarray(func(x_dense), dtype=float)
        y_min = float(np.min(y_dense))
        y_max = float(np.max(y_dense))
        overshoot_low = max(0.0, y_min_obs - y_min)
        overshoot_high = max(0.0, y_max - y_max_obs)
        summary[name] = {
            "y_min": y_min,
            "y_max": y_max,
            "overshoot_low": overshoot_low,
            "overshoot_high": overshoot_high,
        }
    return summary


def kfold_cv_rmse_poly(x, y, degree, k_fold=6):
    """计算多项式给定阶次的 K 折交叉验证 RMSE。"""
    n = x.size
    indices = np.arange(n)
    folds = np.array_split(indices, k_fold)
    all_residual = []

    for test_idx in folds:
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[test_idx], y[test_idx]

        coef = np.polyfit(x_train, y_train, deg=degree)
        y_pred_test = np.polyval(coef, x_test)
        all_residual.append(y_test - y_pred_test)

    residual = np.concatenate(all_residual)
    return float(np.sqrt(np.mean(residual ** 2)))


def choose_degree_by_cv(x, y, max_degree=10, k_fold=6):
    """用 CV-RMSE 选择较合适的多项式阶次。"""
    max_degree = min(max_degree, x.size - 2)
    candidates = list(range(1, max_degree + 1))
    cv_scores = []

    for deg in candidates:
        cv_rmse = kfold_cv_rmse_poly(x, y, degree=deg, k_fold=k_fold)
        cv_scores.append(cv_rmse)

    best_idx = int(np.argmin(cv_scores))
    best_degree = candidates[best_idx]
    return best_degree, candidates, cv_scores


def fit_polynomial(x, y, degree):
    """拟合指定阶次多项式。"""
    coef = np.polyfit(x, y, deg=degree)
    y_hat = np.polyval(coef, x)
    sse, rmse, r2, residual = calc_metrics(y, y_hat)

    dof = x.size - (degree + 1)
    residual_std = float(np.sqrt(sse / dof)) if dof > 0 else np.nan
    return {
        "degree": int(degree),
        "coef": coef,
        "y_hat": y_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "residual_std": residual_std,
    }


def gaussian_pdf(x, mu, sigma):
    """正态分布密度函数。"""
    x = np.asarray(x, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    return (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-((x - mu) ** 2) / (2.0 * sigma ** 2))


def fit_gaussian_nls(x, y):
    """非线性最小二乘拟合正态密度参数 mu、sigma。"""
    mu0 = float(x[np.argmax(y)])
    sigma0 = max(1e-3, 1.0 / (np.sqrt(2.0 * np.pi) * np.max(y)))

    popt, _ = curve_fit(
        gaussian_pdf,
        x,
        y,
        p0=[mu0, sigma0],
        bounds=([np.min(x) - 2.0, 1e-6], [np.max(x) + 2.0, 50.0]),
        maxfev=20000,
    )
    mu_hat, sigma_hat = float(popt[0]), float(popt[1])
    y_hat = gaussian_pdf(x, mu_hat, sigma_hat)
    sse, rmse, r2, _ = calc_metrics(y, y_hat)
    return {
        "mu": mu_hat,
        "sigma": sigma_hat,
        "y_hat": y_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
    }


def print_interpolation_report(summary):
    """打印插值方法比较。"""
    print("\n--- 任务(1) 插值方法比较 ---")
    print("方法对比（区间内最值与过冲）：")
    for name, info in summary.items():
        print(
            f"  {name:7s}: min={info['y_min']:.6f}, max={info['y_max']:.6f}, "
            f"下过冲={info['overshoot_low']:.3e}, 上过冲={info['overshoot_high']:.3e}"
        )
    print("说明：")
    print("  - linear：无过冲、但仅分段线性，拐点处不光滑。")
    print("  - spline：整体更光滑，但可能出现轻微过冲。")
    print("  - pchip：保形性较好，通常在平滑与抑制过冲间更平衡。")


def print_poly_report(poly_result, degree_candidates=None, cv_scores=None):
    """打印多项式拟合结果。"""
    print("\n--- 任务(2) 最小二乘多项式拟合 ---")
    print(f"选定阶次 m = {poly_result['degree']}")
    if degree_candidates is not None and cv_scores is not None:
        print("CV-RMSE（按阶次）：")
        for d, score in zip(degree_candidates, cv_scores):
            print(f"  m={d:2d}: {score:.8e}")
    print("多项式系数（高次到低次）：")
    print("  [" + ", ".join(f"{c:.12e}" for c in poly_result["coef"]) + "]")
    print(
        f"SSE={poly_result['sse']:.12e}, RMSE={poly_result['rmse']:.12e}, "
        f"R^2={poly_result['r2']:.12f}"
    )
    print(f"残差标准差 s = {poly_result['residual_std']:.12e}")


def print_gaussian_report(gauss_result):
    """打印正态分布拟合结果。"""
    print("\n--- 任务(3) 正态分布非线性拟合 ---")
    print(f"参数估计: mu={gauss_result['mu']:.12f}, sigma={gauss_result['sigma']:.12f}")
    print(
        f"SSE={gauss_result['sse']:.12e}, RMSE={gauss_result['rmse']:.12e}, "
        f"R^2={gauss_result['r2']:.12f}"
    )


def plot_all(interp_dict, poly_result, gauss_result, degree_candidates=None, cv_scores=None):
    """绘制三部分结果。"""
    x_dense = np.linspace(float(np.min(X_DATA)), float(np.max(X_DATA)), 1200)
    y_linear = interp_dict["linear"](x_dense)
    y_spline = interp_dict["spline"](x_dense)
    y_pchip = interp_dict["pchip"](x_dense)

    y_poly_dense = np.polyval(poly_result["coef"], x_dense)
    y_gauss_dense = gaussian_pdf(x_dense, gauss_result["mu"], gauss_result["sigma"])

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0))

    ax = axes[0, 0]
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=32, zorder=4, label="观测点")
    ax.plot(x_dense, y_linear, color="#ff7f0e", linewidth=1.8, label="线性插值")
    ax.plot(x_dense, y_spline, color="#2ca02c", linewidth=1.8, linestyle="--", label="三次样条")
    ax.plot(x_dense, y_pchip, color="#d62728", linewidth=1.8, linestyle="-.", label="PCHIP")
    ax.set_title("任务(1) 插值曲线对比")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=32, zorder=4, label="观测点")
    ax.plot(x_dense, y_poly_dense, color="#9467bd", linewidth=2.0, label=f"{poly_result['degree']}阶多项式拟合")
    ax.set_title("任务(2) 最小二乘多项式拟合")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 0]
    ax.scatter(X_DATA, Y_DATA, color="#1f77b4", s=32, zorder=4, label="观测点")
    ax.plot(x_dense, y_gauss_dense, color="#e377c2", linewidth=2.0, label="正态分布拟合")
    ax.set_title("任务(3) 正态分布非线性拟合")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    if degree_candidates is not None and cv_scores is not None:
        ax.plot(degree_candidates, cv_scores, marker="o", color="#8c564b", linewidth=1.8)
        ax.set_title("多项式阶次选择（CV-RMSE）")
        ax.set_xlabel("阶次 m")
        ax.set_ylabel("CV-RMSE")
        ax.grid(alpha=0.3)
    else:
        err_poly = Y_DATA - poly_result["y_hat"]
        err_gauss = Y_DATA - gauss_result["y_hat"]
        ax.plot(X_DATA, err_poly, marker="o", color="#9467bd", label="多项式残差")
        ax.plot(X_DATA, err_gauss, marker="s", color="#e377c2", label="正态拟合残差")
        ax.axhline(0.0, color="black", linewidth=1.0)
        ax.set_title("残差对比")
        ax.set_xlabel("x")
        ax.set_ylabel("残差")
        ax.grid(alpha=0.3)
        ax.legend()

    plt.tight_layout()
    plt.show()


def solve(poly_degree="auto", max_degree=10, k_fold=6, show_plot=True):
    x_dense = np.linspace(float(np.min(X_DATA)), float(np.max(X_DATA)), 1200)

    interp_dict = build_interpolators()
    interp_cmp = interpolation_summary(interp_dict, x_dense=x_dense)

    degree_candidates = None
    cv_scores = None
    if poly_degree == "auto":
        best_degree, degree_candidates, cv_scores = choose_degree_by_cv(
            X_DATA, Y_DATA, max_degree=max_degree, k_fold=k_fold
        )
        poly_result = fit_polynomial(X_DATA, Y_DATA, degree=best_degree)
    else:
        degree = int(poly_degree)
        if degree < 1:
            raise ValueError("多项式阶次至少为 1。")
        if degree >= X_DATA.size:
            raise ValueError(f"多项式阶次需小于样本数 {X_DATA.size}。")
        poly_result = fit_polynomial(X_DATA, Y_DATA, degree=degree)

    gauss_result = fit_gaussian_nls(X_DATA, Y_DATA)

    print("=== 习题 5.8 求解结果 ===")
    print_interpolation_report(interp_cmp)
    print_poly_report(poly_result, degree_candidates=degree_candidates, cv_scores=cv_scores)
    print_gaussian_report(gauss_result)

    if show_plot:
        plot_all(
            interp_dict,
            poly_result,
            gauss_result,
            degree_candidates=degree_candidates,
            cv_scores=cv_scores,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.8 插值、最小二乘多项式与正态分布拟合")
    parser.add_argument(
        "--poly-degree",
        type=str,
        default="auto",
        help="多项式阶次，填整数或 auto（默认 auto）",
    )
    parser.add_argument("--max-degree", type=int, default=10, help="自动选阶时最大阶次，默认 10")
    parser.add_argument("--k-fold", type=int, default=6, help="自动选阶的 K 折数，默认 6")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        poly_degree=args.poly_degree,
        max_degree=args.max_degree,
        k_fold=args.k_fold,
        show_plot=not args.no_plot,
    )

