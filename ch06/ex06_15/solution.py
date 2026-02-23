"""
例题 6.15：小水滴横截面方程边值问题数值求解。

方程：
  h'' + (1-h)*(1+(h')^2)^(3/2) = 0, x in [-1,1]
边界：
  h(-1)=0, h(1)=0

运行示例：
  python ch06/ex06_15/solution.py
  python ch06/ex06_15/solution.py --n-grid 121 --tol 1e-7
  python ch06/ex06_15/solution.py --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_bvp
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


def ode_system(x, y):
    """一阶系统：y[0]=h, y[1]=h'。"""
    h = y[0]
    hp = y[1]
    dh = hp
    dhp = -(1.0 - h) * (1.0 + hp ** 2) ** 1.5
    return np.vstack([dh, dhp])


def bc(ya, yb):
    """边界条件：h(-1)=0, h(1)=0。"""
    return np.array([ya[0], yb[0]])


def initial_guess(x, amplitude):
    """构造对称初值猜测。"""
    h0 = amplitude * (1.0 - x ** 2)
    hp0 = -2.0 * amplitude * x
    return np.vstack([h0, hp0])


def solve_bvp_problem(n_grid=101, tol=1e-6, max_nodes=10000, amplitude=0.4):
    """调用 solve_bvp 求解。"""
    if n_grid < 11:
        raise ValueError("n_grid 建议不小于 11。")

    x_mesh = np.linspace(-1.0, 1.0, n_grid)
    y_guess = initial_guess(x_mesh, amplitude=amplitude)

    sol = solve_bvp(
        fun=ode_system,
        bc=bc,
        x=x_mesh,
        y=y_guess,
        tol=tol,
        max_nodes=max_nodes,
    )
    if not sol.success:
        raise RuntimeError(f"边值问题求解失败：{sol.message}")
    return sol


def print_report(sol, n_plot):
    """打印求解摘要。"""
    x_plot = np.linspace(-1.0, 1.0, n_plot)
    y_plot = sol.sol(x_plot)
    h_plot = y_plot[0]
    hp_plot = y_plot[1]

    idx_max = int(np.argmax(h_plot))
    x_max = float(x_plot[idx_max])
    h_max = float(h_plot[idx_max])

    bc_left = float(sol.sol(-1.0)[0])
    bc_right = float(sol.sol(1.0)[0])

    # 方程残差（用数值梯度近似 h''）
    hpp_num = np.gradient(hp_plot, x_plot)
    residual = hpp_num + (1.0 - h_plot) * (1.0 + hp_plot ** 2) ** 1.5

    print("=== 例题 6.15 数值解结果 ===")
    print("方程: h'' + (1-h)*(1+(h')^2)^(3/2) = 0, x∈[-1,1]")
    print("边界: h(-1)=0, h(1)=0")
    print("\n求解器信息：")
    print(f"  status={sol.status}, message={sol.message}")
    print(f"  迭代次数 niter={sol.niter}, 网格节点数={sol.x.size}")

    print("\n边界条件校验：")
    print(f"  h(-1) = {bc_left:.12e}")
    print(f"  h(1)  = {bc_right:.12e}")

    print("\n曲线特征：")
    print(f"  max h(x) = {h_max:.12f} (x={x_max:.6f})")
    print(f"  方程残差 max|R(x)| ≈ {np.max(np.abs(residual)):.6e} (基于数值梯度近似)")

    return x_plot, h_plot


def plot_result(x_plot, h_plot):
    """绘制 h(x) 曲线。"""
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.plot(x_plot, h_plot, color="#1f77b4", linewidth=2.0, label="数值解 h(x)")
    ax.axhline(0.0, color="black", linewidth=1.0)
    ax.set_title("例题 6.15：小水滴横截面 h(x)")
    ax.set_xlabel("x")
    ax.set_ylabel("h(x)")
    ax.grid(alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()


def solve(n_grid=101, tol=1e-6, max_nodes=10000, amplitude=0.4, n_plot=600, show_plot=True):
    sol = solve_bvp_problem(
        n_grid=n_grid,
        tol=tol,
        max_nodes=max_nodes,
        amplitude=amplitude,
    )
    x_plot, h_plot = print_report(sol, n_plot=n_plot)
    if show_plot:
        plot_result(x_plot, h_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.15 小水滴横截面边值问题数值求解")
    parser.add_argument("--n-grid", type=int, default=101, help="初始网格节点数，默认 101")
    parser.add_argument("--tol", type=float, default=1e-6, help="solve_bvp 收敛容差，默认 1e-6")
    parser.add_argument("--max-nodes", type=int, default=10000, help="solve_bvp 最大节点数，默认 10000")
    parser.add_argument("--amplitude", type=float, default=0.4, help="初值猜测振幅，默认 0.4")
    parser.add_argument("--n-plot", type=int, default=600, help="绘图采样点数，默认 600")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_grid=max(11, args.n_grid),
        tol=args.tol,
        max_nodes=max(200, args.max_nodes),
        amplitude=args.amplitude,
        n_plot=max(100, args.n_plot),
        show_plot=not args.no_plot,
    )

