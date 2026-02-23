"""
例题 6.16：含参数边值问题 y''+mu*y=0 的数值求解。

边界条件：
  y(0)=0, y'(0)=1, y(1)+y'(1)=0

运行示例：
  python ch06/ex06_16/solution.py
  python ch06/ex06_16/solution.py --mode-index 1 --tol 1e-8
  python ch06/ex06_16/solution.py --mode-index 2 --no-plot
"""

import argparse

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.integrate import solve_bvp
    from scipy.optimize import root_scalar
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


def ode_system(_, y, p):
    """一阶系统：y[0]=函数值, y[1]=导数, p[0]=mu。"""
    mu = p[0]
    return np.vstack([y[1], -mu * y[0]])


def bc(ya, yb, _p):
    """边界条件残差。"""
    return np.array([
        ya[0],          # y(0)=0
        ya[1] - 1.0,    # y'(0)=1
        yb[0] + yb[1],  # y(1)+y'(1)=0
    ])


def mode_bracket(mode_index, eps=1e-6):
    """
    tan(k)+k=0 的第 mode_index 个正根区间。
    root in ((n-1/2)pi, n*pi), n=1,2,3,...
    """
    if mode_index < 1:
        raise ValueError("mode_index 必须 >= 1。")
    left = (mode_index - 0.5) * np.pi + eps
    right = mode_index * np.pi - eps
    return left, right


def solve_mode(mode_index=1, n_grid=121, tol=1e-8, max_nodes=20000, n_plot=600):
    """求指定模态的数值解（BVP）并给出解析校验。"""
    left, right = mode_bracket(mode_index)
    k_guess = 0.5 * (left + right)
    mu_guess = k_guess ** 2

    x_mesh = np.linspace(0.0, 1.0, n_grid)
    # 初值猜测采用近似正弦形，满足 y(0)=0、y'(0)≈1
    y_guess = np.vstack([
        np.sin(k_guess * x_mesh) / k_guess,
        np.cos(k_guess * x_mesh),
    ])

    sol = solve_bvp(
        fun=ode_system,
        bc=bc,
        x=x_mesh,
        y=y_guess,
        p=np.array([mu_guess], dtype=float),
        tol=tol,
        max_nodes=max_nodes,
    )
    if not sol.success:
        raise RuntimeError(f"BVP 求解失败：{sol.message}")

    mu_bvp = float(sol.p[0])

    # 解析校验：由 tan(k)+k=0 得 k，再计算 mu_ref=k^2
    f = lambda k: np.tan(k) + k
    root = root_scalar(f, bracket=(left, right), method="brentq")
    k_ref = float(root.root)
    mu_ref = float(k_ref ** 2)

    x_plot = np.linspace(0.0, 1.0, n_plot)
    y_num = sol.sol(x_plot)[0]
    y_ref = np.sin(k_ref * x_plot) / k_ref
    err = y_num - y_ref

    bc_res = bc(sol.sol(0.0), sol.sol(1.0), sol.p)

    return {
        "mode_index": mode_index,
        "mu_bvp": mu_bvp,
        "k_ref": k_ref,
        "mu_ref": mu_ref,
        "x_plot": x_plot,
        "y_num": y_num,
        "y_ref": y_ref,
        "err": err,
        "bc_res": bc_res,
        "niter": int(sol.niter),
        "nodes": int(sol.x.size),
        "status": int(sol.status),
        "message": str(sol.message),
    }


def print_report(res):
    """打印结果摘要。"""
    print("=== 例题 6.16 求解结果（含参数边值问题） ===")
    print("方程: y'' + mu*y = 0")
    print("边界: y(0)=0, y'(0)=1, y(1)+y'(1)=0")
    print(f"模态序号: n = {res['mode_index']}")
    print("\n求解器信息：")
    print(f"  status={res['status']}, message={res['message']}")
    print(f"  迭代次数 niter={res['niter']}, 网格节点数={res['nodes']}")

    print("\n参数 mu：")
    print(f"  mu_bvp = {res['mu_bvp']:.12f}")
    print(f"  mu_ref = {res['mu_ref']:.12f}  (由 tan(k)+k=0 校验)")
    print(f"  |mu_bvp-mu_ref| = {abs(res['mu_bvp'] - res['mu_ref']):.12e}")

    print("\n边界残差：")
    print(f"  y(0)         = {res['bc_res'][0]:.12e}")
    print(f"  y'(0)-1      = {res['bc_res'][1]:.12e}")
    print(f"  y(1)+y'(1)   = {res['bc_res'][2]:.12e}")

    print("\n曲线误差（数值解 vs 解析校验）：")
    print(f"  max|error| = {np.max(np.abs(res['err'])):.12e}")
    print(f"  RMSE       = {np.sqrt(np.mean(res['err'] ** 2)):.12e}")


def plot_result(res):
    """绘制 y(x) 曲线及误差。"""
    x = res["x_plot"]
    y_num = res["y_num"]
    y_ref = res["y_ref"]
    err = res["err"]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9.2, 7.0), sharex=True)

    ax1.plot(x, y_num, color="#1f77b4", linewidth=2.0, label="数值解 y(x)")
    ax1.plot(x, y_ref, color="#d62728", linewidth=1.7, linestyle="--", label="解析校验解")
    ax1.set_title(f"例题 6.16：y(x)（n={res['mode_index']}）")
    ax1.set_ylabel("y")
    ax1.grid(alpha=0.3)
    ax1.legend()

    ax2.plot(x, err, color="#2ca02c", linewidth=1.8, label="误差 y_num-y_ref")
    ax2.axhline(0.0, color="black", linewidth=1.0)
    ax2.set_xlabel("x")
    ax2.set_ylabel("误差")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(mode_index=1, n_grid=121, tol=1e-8, max_nodes=20000, n_plot=600, show_plot=True):
    res = solve_mode(
        mode_index=mode_index,
        n_grid=n_grid,
        tol=tol,
        max_nodes=max_nodes,
        n_plot=n_plot,
    )
    print_report(res)
    if show_plot:
        plot_result(res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题6.16 含参数边值问题数值求解")
    parser.add_argument("--mode-index", type=int, default=1, help="模态序号 n（默认 1）")
    parser.add_argument("--n-grid", type=int, default=121, help="初始网格节点数，默认 121")
    parser.add_argument("--tol", type=float, default=1e-8, help="solve_bvp 容差，默认 1e-8")
    parser.add_argument("--max-nodes", type=int, default=20000, help="最大节点数，默认 20000")
    parser.add_argument("--n-plot", type=int, default=600, help="绘图采样点数，默认 600")
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        mode_index=max(1, args.mode_index),
        n_grid=max(21, args.n_grid),
        tol=args.tol,
        max_nodes=max(500, args.max_nodes),
        n_plot=max(200, args.n_plot),
        show_plot=not args.no_plot,
    )

