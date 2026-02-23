"""
例题 6.17：非线性微分方程组边值问题数值求解。

方程组：
  u' = 0.5*u*(w-u)/v
  v' = -0.5*(w-u)
  w' = (0.9 - 1000*(w-y) - 0.5*w*(w-u))/z
  z' = 0.5*(w-u)
  y' = -100*(y-w)

边界条件：
  u(0)=1, v(0)=1, w(0)=1, z(0)=-10, w(1)=y(1)

运行示例：
  python ch06/ex06_17/solution.py
  python ch06/ex06_17/solution.py --n-grid 201 --tol 1e-6
  python ch06/ex06_17/solution.py --no-plot
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


def ode_system(_, y):
    """方程组右端，y=[u,v,w,z,yy]。"""
    u = y[0]
    v = y[1]
    w = y[2]
    z = y[3]
    yy = y[4]

    du = 0.5 * u * (w - u) / v
    dv = -0.5 * (w - u)
    dw = (0.9 - 1000.0 * (w - yy) - 0.5 * w * (w - u)) / z
    dz = 0.5 * (w - u)
    dyy = -100.0 * (yy - w)
    return np.vstack([du, dv, dw, dz, dyy])


def bc(ya, yb):
    """边界条件残差。"""
    return np.array(
        [
            ya[0] - 1.0,      # u(0)=1
            ya[1] - 1.0,      # v(0)=1
            ya[2] - 1.0,      # w(0)=1
            ya[3] + 10.0,     # z(0)=-10
            yb[2] - yb[4],    # w(1)=y(1)
        ]
    )


def initial_guess(t):
    """构造初始猜测曲线。"""
    # 基于边界条件的平滑初值；加入小扰动减少退化
    u0 = 1.0 + 0.02 * np.sin(np.pi * t)
    v0 = 1.0 + 0.01 * np.cos(np.pi * t)
    w0 = 1.0 + 0.03 * t * (1.0 - t)
    z0 = -10.0 + 0.05 * np.sin(2.0 * np.pi * t)
    y0 = 1.0 + 0.03 * t * (1.0 - t)
    return np.vstack([u0, v0, w0, z0, y0])


def solve_problem(n_grid=121, tol=1e-5, max_nodes=50000):
    """调用 solve_bvp 求解。"""
    if n_grid < 21:
        raise ValueError("n_grid 建议不小于 21。")

    t_mesh = np.linspace(0.0, 1.0, n_grid)
    y_guess = initial_guess(t_mesh)

    sol = solve_bvp(
        fun=ode_system,
        bc=bc,
        x=t_mesh,
        y=y_guess,
        tol=tol,
        max_nodes=max_nodes,
    )
    if not sol.success:
        raise RuntimeError(f"BVP 求解失败：{sol.message}")
    return sol


def print_report(sol, n_plot=800):
    """打印结果摘要与残差。"""
    t_plot = np.linspace(0.0, 1.0, n_plot)
    y_plot = sol.sol(t_plot)
    u, v, w, z, yy = y_plot

    # 边界校验
    bc_res = bc(sol.sol(0.0), sol.sol(1.0))

    # 方程残差（用数值梯度近似导数）
    du_num = np.gradient(u, t_plot)
    dv_num = np.gradient(v, t_plot)
    dw_num = np.gradient(w, t_plot)
    dz_num = np.gradient(z, t_plot)
    dyy_num = np.gradient(yy, t_plot)

    rhs = ode_system(t_plot, y_plot)
    ode_res = np.vstack([du_num, dv_num, dw_num, dz_num, dyy_num]) - rhs

    print("=== 例题 6.17 数值解结果 ===")
    print("方程组边值问题：u,v,w,z,y over t∈[0,1]")
    print("\n求解器信息：")
    print(f"  status={sol.status}, message={sol.message}")
    print(f"  迭代次数 niter={sol.niter}, 网格节点数={sol.x.size}")

    print("\n边界值：")
    y0 = sol.sol(0.0)
    y1 = sol.sol(1.0)
    print(f"  u(0)={y0[0]:.10f}, v(0)={y0[1]:.10f}, w(0)={y0[2]:.10f}, z(0)={y0[3]:.10f}")
    print(f"  w(1)={y1[2]:.10f}, y(1)={y1[4]:.10f}")

    print("\n边界残差：")
    print(f"  [u(0)-1, v(0)-1, w(0)-1, z(0)+10, w(1)-y(1)] = {bc_res}")
    print(f"  max|bc_res| = {np.max(np.abs(bc_res)):.6e}")

    print("\n方程残差（数值梯度近似，仅供参考）：")
    print(f"  max|ode_res| = {np.max(np.abs(ode_res)):.6e}")

    return t_plot, y_plot


def plot_result(t, y):
    """绘制 5 个状态变量曲线。"""
    labels = ["u(t)", "v(t)", "w(t)", "z(t)", "y(t)"]
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

    fig, axes = plt.subplots(3, 2, figsize=(11.5, 8.0), sharex=True)
    axes = axes.ravel()

    for i in range(5):
        axes[i].plot(t, y[i], color=colors[i], linewidth=1.8)
        axes[i].set_title(labels[i])
        axes[i].grid(alpha=0.3)
        axes[i].set_ylabel(labels[i])

    # 最后一个子图用于展示 w-y 差值，强调边界条件
    axes[5].plot(t, y[2] - y[4], color="#8c564b", linewidth=1.8)
    axes[5].axhline(0.0, color="black", linewidth=1.0)
    axes[5].set_title("w(t)-y(t)")
    axes[5].set_ylabel("w-y")
    axes[5].grid(alpha=0.3)

    axes[4].set_xlabel("t")
    axes[5].set_xlabel("t")
    fig.suptitle("例题 6.17：非线性方程组边值问题数值解", y=0.995)
    plt.tight_layout()
    plt.show()


def solve(n_grid=121, tol=1e-5, max_nodes=50000, n_plot=800, show_plot=True):
    sol = solve_problem(
        n_grid=n_grid,
        tol=tol,
        max_nodes=max_nodes,
    )
    t_plot, y_plot = print_report(sol, n_plot=n_plot)
    if show_plot:
        plot_result(t_plot, y_plot)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.17 非线性微分方程组边值问题")
    parser.add_argument("--n-grid", type=int, default=121, help="初始网格节点数，默认 121")
    parser.add_argument("--tol", type=float, default=1e-5, help="solve_bvp 容差，默认 1e-5")
    parser.add_argument("--max-nodes", type=int, default=50000, help="最大节点数，默认 50000")
    parser.add_argument("--n-plot", type=int, default=800, help="绘图采样点数，默认 800")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        n_grid=max(21, args.n_grid),
        tol=args.tol,
        max_nodes=max(1000, args.max_nodes),
        n_plot=max(200, args.n_plot),
        show_plot=not args.no_plot,
    )

