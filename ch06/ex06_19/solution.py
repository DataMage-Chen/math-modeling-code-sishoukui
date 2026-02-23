"""
例题 6.19：捕食者-被捕食者模型参数拟合（a,b,c,d）。

运行示例：
  python ch06/ex06_19/solution.py
  python ch06/ex06_19/solution.py --n-starts 30 --seed 2026 --no-plot
  python ch06/ex06_19/solution.py --method LSODA --rtol 1e-8 --atol 1e-10
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
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


T_OBS = np.array([0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 14, 16, 18], dtype=float)
X_OBS = np.array([60, 63, 64, 63, 61, 58, 53, 44, 39, 38, 41, 46, 53], dtype=float)
Y_OBS = np.array([30, 34, 38, 44, 50, 55, 58, 56, 47, 38, 30, 27, 26], dtype=float)

X0 = float(X_OBS[0])
Y0 = float(Y_OBS[0])


def lotka_rhs(_, state, a, b, c, d):
    """Lotka-Volterra 方程右端。"""
    x, y = state
    dx = a * x - b * x * y
    dy = -c * y + d * x * y
    return [dx, dy]


def simulate(theta, t_eval, method="RK45", rtol=1e-7, atol=1e-9):
    """给定参数 theta=[a,b,c,d] 积分到指定时刻。"""
    a, b, c, d = [float(v) for v in theta]
    sol = solve_ivp(
        fun=lambda t, s: lotka_rhs(t, s, a=a, b=b, c=c, d=d),
        t_span=(float(np.min(t_eval)), float(np.max(t_eval))),
        y0=[X0, Y0],
        t_eval=t_eval,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    if not sol.success:
        return None, None, False
    return sol.y[0], sol.y[1], True


def metrics(obs, fit):
    """计算 SSE、RMSE、R^2。"""
    obs = np.asarray(obs, dtype=float)
    fit = np.asarray(fit, dtype=float)
    residual = obs - fit
    sse = float(np.sum(residual ** 2))
    rmse = float(np.sqrt(np.mean(residual ** 2)))
    sst = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0
    return sse, rmse, r2


def fit_parameters(
    n_starts=20,
    seed=2026,
    method="RK45",
    rtol=1e-7,
    atol=1e-9,
):
    """
    多初值最小二乘拟合。
    用对数参数化保证 a,b,c,d > 0：theta = exp(phi)。
    """
    rng = np.random.default_rng(seed)

    # 基准初值（经验）
    theta_base = np.array([0.2, 0.0055, 0.4, 0.009], dtype=float)
    phi_base = np.log(theta_base)

    sx = float(np.std(X_OBS, ddof=1))
    sy = float(np.std(Y_OBS, ddof=1))

    def residual_phi(phi):
        theta = np.exp(phi)
        x_fit, y_fit, ok = simulate(theta, T_OBS, method=method, rtol=rtol, atol=atol)
        if (not ok) or np.any(~np.isfinite(x_fit)) or np.any(~np.isfinite(y_fit)) or np.any(x_fit <= 0) or np.any(y_fit <= 0):
            # 返回大残差惩罚不可行参数
            return np.full(T_OBS.size * 2, 1e6, dtype=float)

        rx = (x_fit - X_OBS) / sx
        ry = (y_fit - Y_OBS) / sy
        return np.concatenate([rx, ry])

    best = None
    for k in range(n_starts):
        if k == 0:
            phi0 = phi_base.copy()
        else:
            # 在 log-space 加扰动
            phi0 = phi_base + rng.normal(loc=0.0, scale=0.6, size=4)

        res = least_squares(
            residual_phi,
            x0=phi0,
            method="trf",
            max_nfev=3000,
        )
        if (best is None) or (res.cost < best.cost):
            best = res

    phi_hat = best.x
    theta_hat = np.exp(phi_hat)
    x_fit, y_fit, ok = simulate(theta_hat, T_OBS, method=method, rtol=rtol, atol=atol)
    if not ok:
        raise RuntimeError("最优参数下积分失败。")

    sse_x, rmse_x, r2_x = metrics(X_OBS, x_fit)
    sse_y, rmse_y, r2_y = metrics(Y_OBS, y_fit)

    return {
        "theta": theta_hat,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "sse_x": sse_x,
        "rmse_x": rmse_x,
        "r2_x": r2_x,
        "sse_y": sse_y,
        "rmse_y": rmse_y,
        "r2_y": r2_y,
        "best_cost": float(best.cost),
        "best_nfev": int(best.nfev),
        "best_success": bool(best.success),
        "best_message": str(best.message),
        "method": method,
        "rtol": rtol,
        "atol": atol,
        "n_starts": n_starts,
        "seed": seed,
    }


def print_report(result):
    """打印拟合结果。"""
    a, b, c, d = result["theta"]
    print("=== 例题 6.19 参数拟合结果（捕食者-被捕食者） ===")
    print(f"多初值次数: {result['n_starts']}, 随机种子: {result['seed']}")
    print(f"积分器: {result['method']} (rtol={result['rtol']}, atol={result['atol']})")
    print("\n参数估计：")
    print(f"  a = {a:.12f}")
    print(f"  b = {b:.12f}")
    print(f"  c = {c:.12f}")
    print(f"  d = {d:.12f}")

    print("\n优化信息：")
    print(f"  success = {result['best_success']}")
    print(f"  nfev    = {result['best_nfev']}")
    print(f"  cost    = {result['best_cost']:.12f}")
    print(f"  message = {result['best_message']}")

    print("\n拟合指标：")
    print(f"  兔子 x(t): SSE={result['sse_x']:.6f}, RMSE={result['rmse_x']:.6f}, R^2={result['r2_x']:.8f}")
    print(f"  狐狸 y(t): SSE={result['sse_y']:.6f}, RMSE={result['rmse_y']:.6f}, R^2={result['r2_y']:.8f}")

    print("\n观测 vs 拟合：")
    print("    t      x_obs    x_fit      y_obs    y_fit")
    for t, xo, xf, yo, yf in zip(T_OBS, X_OBS, result["x_fit"], Y_OBS, result["y_fit"]):
        print(f"  {t:4.0f}   {xo:8.3f} {xf:8.3f}   {yo:8.3f} {yf:8.3f}")


def plot_result(result):
    """绘制时间序列拟合图与相图。"""
    theta = result["theta"]
    t_dense = np.linspace(float(np.min(T_OBS)), float(np.max(T_OBS)), 900)
    x_dense, y_dense, ok = simulate(
        theta,
        t_dense,
        method=result["method"],
        rtol=result["rtol"],
        atol=result["atol"],
    )
    if not ok:
        raise RuntimeError("绘图阶段积分失败。")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2))

    # 时间序列
    axes[0].scatter(T_OBS, X_OBS, color="#1f77b4", s=42, label="兔子观测")
    axes[0].plot(t_dense, x_dense, color="#1f77b4", linewidth=1.8, linestyle="-", label="兔子拟合")
    axes[0].scatter(T_OBS, Y_OBS, color="#d62728", s=42, label="狐狸观测")
    axes[0].plot(t_dense, y_dense, color="#d62728", linewidth=1.8, linestyle="-", label="狐狸拟合")
    axes[0].set_title("时间序列拟合")
    axes[0].set_xlabel("t（月）")
    axes[0].set_ylabel("种群数量")
    axes[0].grid(alpha=0.3)
    axes[0].legend(ncol=2, fontsize=9)

    # 相图
    axes[1].scatter(X_OBS, Y_OBS, color="#7f7f7f", s=35, label="观测点")
    axes[1].plot(x_dense, y_dense, color="#2ca02c", linewidth=2.0, label="模型相轨线")
    axes[1].scatter([X_OBS[0]], [Y_OBS[0]], color="black", s=55, marker="x", label="初始点")
    axes[1].set_title("相图（x-y 平面）")
    axes[1].set_xlabel("兔子 x")
    axes[1].set_ylabel("狐狸 y")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def solve(n_starts=20, seed=2026, method="RK45", rtol=1e-7, atol=1e-9, show_plot=True):
    result = fit_parameters(
        n_starts=n_starts,
        seed=seed,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    print_report(result)
    if show_plot:
        plot_result(result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.19 捕食者-被捕食者参数拟合")
    parser.add_argument("--n-starts", type=int, default=20, help="多初值次数，默认 20")
    parser.add_argument("--seed", type=int, default=2026, help="随机种子，默认 2026")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="积分方法，默认 RK45",
    )
    parser.add_argument("--rtol", type=float, default=1e-7, help="相对容差，默认 1e-7")
    parser.add_argument("--atol", type=float, default=1e-9, help="绝对容差，默认 1e-9")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_starts=max(1, args.n_starts),
        seed=args.seed,
        method=args.method,
        rtol=args.rtol,
        atol=args.atol,
        show_plot=not args.no_plot,
    )

