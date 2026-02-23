"""
习题 5.7：二维非线性函数反拟合

运行示例：
  python ch05/hw05_07/solution.py
  python ch05/hw05_07/solution.py --method nonlinear --no-plot
  python ch05/hw05_07/solution.py --init-a 1.5 --init-b 2.0 --denom-eps 1e-6
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.optimize import least_squares
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


A_TRUE = 2.0
B_TRUE = 3.0


def f_model(x, y, a, b):
    """题设模型 f(x,y)=a*x*y/(1+b*sin(x))。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    denom = 1.0 + b * np.sin(x)
    return (a * x * y) / denom


def build_simulated_data():
    """按题目要求生成模拟数据。"""
    x_line = np.linspace(-6.0, 6.0, 30)
    y_line = np.linspace(-6.0, 6.0, 40)
    x_grid, y_grid = np.meshgrid(x_line, y_line)
    z_grid = f_model(x_grid, y_grid, A_TRUE, B_TRUE)
    return x_line, y_line, x_grid, y_grid, z_grid


def filter_valid_points(x_grid, y_grid, z_grid, denom_eps):
    """过滤分母过小或非有限点，避免拟合时数值不稳定。"""
    denom_true = 1.0 + B_TRUE * np.sin(x_grid)
    mask = np.isfinite(z_grid) & (np.abs(denom_true) > denom_eps)

    x_vec = x_grid[mask]
    y_vec = y_grid[mask]
    z_vec = z_grid[mask]
    return mask, x_vec, y_vec, z_vec


def calc_metrics(z_obs, z_hat):
    """计算 SSE、RMSE、R^2。"""
    z_obs = np.asarray(z_obs, dtype=float)
    z_hat = np.asarray(z_hat, dtype=float)
    residual = z_obs - z_hat
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((z_obs - np.mean(z_obs)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def fit_nonlinear(x_vec, y_vec, z_vec, init_a, init_b):
    """方法一：非线性最小二乘（对应直接反拟合思路）。"""

    def residual(params):
        a, b = params
        z_hat = f_model(x_vec, y_vec, a, b)
        return z_hat - z_vec

    res = least_squares(
        residual,
        x0=np.array([init_a, init_b], dtype=float),
        method="trf",
        max_nfev=30000,
    )

    a_hat, b_hat = float(res.x[0]), float(res.x[1])
    z_hat = f_model(x_vec, y_vec, a_hat, b_hat)
    sse, rmse, r2 = calc_metrics(z_vec, z_hat)
    return {
        "a": a_hat,
        "b": b_hat,
        "z_hat": z_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
        "success": bool(res.success),
        "status": int(res.status),
        "nfev": int(res.nfev),
        "message": str(res.message),
    }


def fit_linearized(x_vec, y_vec, z_vec):
    """方法二：代数重排后做线性最小二乘。"""
    # a*(x*y) - b*(z*sin(x)) = z
    design = np.column_stack([x_vec * y_vec, -z_vec * np.sin(x_vec)])
    coef, *_ = np.linalg.lstsq(design, z_vec, rcond=None)
    a_hat, b_hat = float(coef[0]), float(coef[1])

    z_hat = f_model(x_vec, y_vec, a_hat, b_hat)
    sse, rmse, r2 = calc_metrics(z_vec, z_hat)
    return {
        "a": a_hat,
        "b": b_hat,
        "z_hat": z_hat,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
    }


def print_method_result(title, result):
    """打印单方法结果。"""
    da = result["a"] - A_TRUE
    db = result["b"] - B_TRUE
    print(f"\n--- {title} ---")
    print(f"参数估计: a={result['a']:.12f}, b={result['b']:.12f}")
    print(f"参数误差: da={da:+.3e}, db={db:+.3e}")
    print(f"SSE={result['sse']:.12e}, RMSE={result['rmse']:.12e}, R^2={result['r2']:.12f}")
    if "success" in result:
        print(
            "优化状态: "
            f"success={result['success']}, status={result['status']}, "
            f"nfev={result['nfev']}"
        )
        print(f"优化信息: {result['message']}")


def build_full_surface_prediction(x_grid, y_grid, result):
    """给定拟合参数，生成整个网格上的预测面。"""
    return f_model(x_grid, y_grid, result["a"], result["b"])


def plot_surfaces(x_grid, y_grid, z_grid, fit_results):
    """绘制观测曲面与拟合曲面对比。"""
    methods = list(fit_results.keys())
    fig = plt.figure(figsize=(6.8 * len(methods), 5.4))

    for idx, method in enumerate(methods, start=1):
        ax = fig.add_subplot(1, len(methods), idx, projection="3d")
        ax.plot_surface(
            x_grid,
            y_grid,
            z_grid,
            cmap="Blues",
            alpha=0.55,
            linewidth=0,
            antialiased=True,
        )
        z_hat_grid = fit_results[method]["z_hat_grid"]
        ax.plot_wireframe(
            x_grid,
            y_grid,
            z_hat_grid,
            rstride=2,
            cstride=2,
            color="#d62728",
            linewidth=0.8,
            alpha=0.9,
        )
        ax.set_title(f"{method}：观测面(蓝)+拟合面(红网格)")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    plt.tight_layout()
    plt.show()


def plot_observed_vs_pred(z_vec, fit_results):
    """绘制观测值-预测值散点图。"""
    fig, ax = plt.subplots(figsize=(7.6, 5.2))

    z_min = float(np.min(z_vec))
    z_max = float(np.max(z_vec))
    ref_line = np.linspace(z_min, z_max, 200)
    ax.plot(ref_line, ref_line, color="black", linewidth=1.2, label="y=x 参考线")

    color_map = {
        "nonlinear": "#1f77b4",
        "linearized": "#ff7f0e",
    }
    label_map = {
        "nonlinear": "非线性最小二乘",
        "linearized": "线性化最小二乘",
    }
    for method, result in fit_results.items():
        ax.scatter(
            z_vec,
            result["z_hat"],
            s=18,
            alpha=0.7,
            color=color_map.get(method, None),
            label=f"{label_map.get(method, method)} (R^2={result['r2']:.4f})",
        )

    ax.set_title("观测值与预测值对比")
    ax.set_xlabel("观测值 z")
    ax.set_ylabel("预测值 z_hat")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(method="both", init_a=1.0, init_b=1.0, denom_eps=1e-8, show_plot=True):
    x_line, y_line, x_grid, y_grid, z_grid = build_simulated_data()
    mask, x_vec, y_vec, z_vec = filter_valid_points(x_grid, y_grid, z_grid, denom_eps=denom_eps)

    total_points = x_grid.size
    used_points = x_vec.size

    print("=== 习题 5.7 求解结果 ===")
    print(f"真参数: a={A_TRUE}, b={B_TRUE}")
    print(f"网格规模: x点数={x_line.size}, y点数={y_line.size}, 总点数={total_points}")
    print(f"有效点数: {used_points}（过滤 {total_points - used_points} 个，denom_eps={denom_eps}）")
    print(f"拟合方法: {method}")
    print(f"初始值(非线性法): a0={init_a}, b0={init_b}")

    fit_results = {}
    if method in ("nonlinear", "both"):
        res_nl = fit_nonlinear(x_vec, y_vec, z_vec, init_a=init_a, init_b=init_b)
        fit_results["nonlinear"] = res_nl
        print_method_result("方法(1) 非线性最小二乘", res_nl)

    if method in ("linearized", "both"):
        res_lin = fit_linearized(x_vec, y_vec, z_vec)
        fit_results["linearized"] = res_lin
        print_method_result("方法(2) 线性化最小二乘", res_lin)

    if show_plot and fit_results:
        for key, result in fit_results.items():
            z_hat_grid = build_full_surface_prediction(x_grid, y_grid, result)
            z_hat_grid = np.where(mask, z_hat_grid, np.nan)
            result["z_hat_grid"] = z_hat_grid

        plot_surfaces(x_grid, y_grid, np.where(mask, z_grid, np.nan), fit_results)
        plot_observed_vs_pred(z_vec, fit_results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.7 二维非线性函数反拟合")
    parser.add_argument(
        "--method",
        type=str,
        choices=["nonlinear", "linearized", "both"],
        default="both",
        help="拟合方法：nonlinear / linearized / both（默认 both）",
    )
    parser.add_argument("--init-a", type=float, default=1.0, help="非线性法参数 a 初始值，默认 1.0")
    parser.add_argument("--init-b", type=float, default=1.0, help="非线性法参数 b 初始值，默认 1.0")
    parser.add_argument(
        "--denom-eps",
        type=float,
        default=1e-8,
        help="分母稳定性阈值：|1+b_true*sin(x)|<=eps 的点会被过滤，默认 1e-8",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        method=args.method,
        init_a=args.init_a,
        init_b=args.init_b,
        denom_eps=args.denom_eps,
        show_plot=not args.no_plot,
    )

