"""
例题 6.13：求解 y''-2y'+y=e^x, y(0)=1, y'(0)=-1 在 [-1,1] 上的数值解。

运行示例：
  python ch06/ex06_13/solution.py
  python ch06/ex06_13/solution.py --method DOP853 --n-grid 401
  python ch06/ex06_13/solution.py --no-reference --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_ivp
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


def ode_rhs(x, state):
    """一阶系统右端：state=[y, yp]。"""
    y, yp = state
    dy = yp
    dyp = 2.0 * yp - y + np.exp(x)
    return [dy, dyp]


def exact_solution(x):
    """解析解（用于可选校验）。"""
    x = np.asarray(x, dtype=float)
    return np.exp(x) * (1.0 - 2.0 * x + 0.5 * x ** 2)


def solve_both_directions(method="RK45", n_grid=301, rtol=1e-9, atol=1e-12):
    """从 x=0 向左右双向积分，拼接得到 [-1,1] 解。"""
    if n_grid < 21:
        raise ValueError("n_grid 建议不小于 21。")

    n_left = n_grid // 2 + 1
    n_right = n_grid - n_left + 1

    x_left = np.linspace(0.0, -1.0, n_left)
    x_right = np.linspace(0.0, 1.0, n_right)
    y0 = [1.0, -1.0]

    sol_left = solve_ivp(
        fun=ode_rhs,
        t_span=(0.0, -1.0),
        y0=y0,
        t_eval=x_left,
        method=method,
        rtol=rtol,
        atol=atol,
    )
    sol_right = solve_ivp(
        fun=ode_rhs,
        t_span=(0.0, 1.0),
        y0=y0,
        t_eval=x_right,
        method=method,
        rtol=rtol,
        atol=atol,
    )

    if not sol_left.success:
        raise RuntimeError(f"向左积分失败：{sol_left.message}")
    if not sol_right.success:
        raise RuntimeError(f"向右积分失败：{sol_right.message}")

    # 左侧结果当前顺序是 0->-1，翻转为 -1->0；右侧去掉 x=0 重复点
    x_all = np.concatenate([sol_left.t[::-1], sol_right.t[1:]])
    y_all = np.concatenate([sol_left.y[0][::-1], sol_right.y[0][1:]])
    yp_all = np.concatenate([sol_left.y[1][::-1], sol_right.y[1][1:]])

    info = {
        "nfev_left": int(sol_left.nfev),
        "nfev_right": int(sol_right.nfev),
        "njev_left": int(sol_left.njev),
        "njev_right": int(sol_right.njev),
        "nlu_left": int(sol_left.nlu),
        "nlu_right": int(sol_right.nlu),
    }
    return x_all, y_all, yp_all, info


def print_report(x_all, y_all, yp_all, info, method, with_reference):
    """打印结果摘要。"""
    print("=== 例题 6.13 数值解结果 ===")
    print("方程: y'' - 2y' + y = e^x")
    print("初值: y(0)=1, y'(0)=-1")
    print("区间: [-1,1]（由 x=0 双向积分拼接）")
    print(f"方法: {method}")
    print(
        "求解器统计: "
        f"left(nfev={info['nfev_left']}, njev={info['njev_left']}, nlu={info['nlu_left']}), "
        f"right(nfev={info['nfev_right']}, njev={info['njev_right']}, nlu={info['nlu_right']})"
    )

    # 关键点输出
    y_m1 = float(np.interp(-1.0, x_all, y_all))
    y_0 = float(np.interp(0.0, x_all, y_all))
    y_1 = float(np.interp(1.0, x_all, y_all))
    yp_0 = float(np.interp(0.0, x_all, yp_all))
    print("\n关键点：")
    print(f"  y(-1) = {y_m1:.12f}")
    print(f"  y(0)  = {y_0:.12f}")
    print(f"  y(1)  = {y_1:.12f}")
    print(f"  y'(0) = {yp_0:.12f}")

    if with_reference:
        y_ref = exact_solution(x_all)
        err = y_all - y_ref
        print("\n与解析解对比误差：")
        print(f"  max|error| = {np.max(np.abs(err)):.12e}")
        print(f"  RMSE       = {np.sqrt(np.mean(err ** 2)):.12e}")


def plot_result(x_all, y_all, with_reference):
    """绘制数值解（及可选解析解与误差）。"""
    if with_reference:
        y_ref = exact_solution(x_all)
        err = y_all - y_ref
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.0, 7.2), sharex=True)
    else:
        fig, ax1 = plt.subplots(figsize=(9.0, 5.0))
        ax2 = None

    ax1.plot(x_all, y_all, color="#d62728", linewidth=2.0, label="数值解 y_num")
    if with_reference:
        ax1.plot(x_all, y_ref, color="#1f77b4", linewidth=1.8, linestyle="--", label="解析解 y_exact")
    ax1.set_title("例题 6.13：区间 [-1,1] 上的数值解")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    if with_reference and ax2 is not None:
        ax2.plot(x_all, err, color="#2ca02c", linewidth=1.8, label="误差 y_num - y_exact")
        ax2.axhline(0.0, color="black", linewidth=1.0)
        ax2.set_xlabel("x")
        ax2.set_ylabel("误差")
        ax2.grid(alpha=0.3)
        ax2.legend()
    else:
        ax1.set_xlabel("x")

    plt.tight_layout()
    plt.show()


def solve(method="RK45", n_grid=301, rtol=1e-9, atol=1e-12, with_reference=True, show_plot=True):
    x_all, y_all, yp_all, info = solve_both_directions(
        method=method,
        n_grid=n_grid,
        rtol=rtol,
        atol=atol,
    )
    print_report(x_all, y_all, yp_all, info, method=method, with_reference=with_reference)
    if show_plot:
        plot_result(x_all, y_all, with_reference=with_reference)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.13 二阶线性方程在 [-1,1] 的数值解")
    parser.add_argument(
        "--method",
        type=str,
        choices=["RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA"],
        default="RK45",
        help="solve_ivp 方法，默认 RK45",
    )
    parser.add_argument("--n-grid", type=int, default=301, help="全区间网格点数，默认 301")
    parser.add_argument("--rtol", type=float, default=1e-9, help="相对容差，默认 1e-9")
    parser.add_argument("--atol", type=float, default=1e-12, help="绝对容差，默认 1e-12")
    parser.add_argument("--no-reference", action="store_true", help="不与解析解对比")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        method=args.method,
        n_grid=max(21, args.n_grid),
        rtol=args.rtol,
        atol=args.atol,
        with_reference=not args.no_reference,
        show_plot=not args.no_plot,
    )

