"""
例题 5.22：调水调沙试验数据建模

任务：
1) 估计任意时刻排沙量及总排沙量；
2) 拟合排沙量与水流量关系。

运行示例：
  python ch05/ex05_22/solution.py
  python ch05/ex05_22/solution.py --interp cubic --query-hours 6 33 87.5 160 --no-plot
  python ch05/ex05_22/solution.py --int-grid 12001
"""

import argparse
from datetime import datetime, timedelta

try:
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.interpolate import CubicSpline, PchipInterpolator
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


# 表 5.14 数据：每 12 小时记录一次
Q_DATA = np.array(
    [
        1800, 1900, 2100, 2200, 2300, 2400, 2500, 2600,
        2650, 2700, 2720, 2650, 2600, 2500, 2300, 2200,
        2000, 1850, 1820, 1800, 1750, 1500, 1000, 900,
    ],
    dtype=float,
)

C_DATA = np.array(
    [
        32, 60, 75, 85, 90, 98, 100, 102,
        108, 112, 115, 116, 118, 120, 118, 105,
        80, 60, 50, 30, 26, 20, 8, 5,
    ],
    dtype=float,
)

# 24 个时点，对应 0, 12, 24, ..., 276 小时
T_HOURS = np.arange(Q_DATA.size, dtype=float) * 12.0
START_TIME = datetime(2026, 6, 29, 8, 0)


def trapezoid_integral(y, x):
    """兼容 numpy 新旧版本的梯形积分。"""
    trapz_fn = getattr(np, "trapezoid", np.trapz)
    return float(trapz_fn(y, x))


def build_interpolators(method):
    """构造 Q(t)、C(t) 的插值器。"""
    if method == "pchip":
        q_interp = PchipInterpolator(T_HOURS, Q_DATA)
        c_interp = PchipInterpolator(T_HOURS, C_DATA)
    elif method == "cubic":
        q_interp = CubicSpline(T_HOURS, Q_DATA, bc_type="not-a-knot")
        c_interp = CubicSpline(T_HOURS, C_DATA, bc_type="not-a-knot")
    else:
        raise ValueError(f"不支持的插值方式: {method}")
    return q_interp, c_interp


def format_time(hour_from_start):
    """将相对小时转成可读时间。"""
    dt = START_TIME + timedelta(hours=float(hour_from_start))
    return dt.strftime("%m-%d %H:%M")


def evaluate_queries(query_hours, q_interp, c_interp):
    """输出查询时刻的 Q、C、S。"""
    if query_hours is None:
        query_hours = [6.0, 33.0, 87.5, 160.0, 250.0]

    t0 = float(T_HOURS[0])
    t1 = float(T_HOURS[-1])
    result = []
    for h in query_hours:
        h = float(h)
        if h < t0 or h > t1:
            raise ValueError(f"查询时刻 {h} 小时超出区间 [{t0}, {t1}]。")
        qv = float(q_interp(h))
        cv = float(c_interp(h))
        sv = qv * cv
        result.append((h, qv, cv, sv))
    return result


def estimate_total_sediment(q_interp, c_interp, int_grid):
    """估计总排沙量：M = ∫ S(t)dt，注意小时到秒的单位换算。"""
    if int_grid < 101:
        raise ValueError("积分网格点数 int_grid 建议不小于 101。")

    t_dense = np.linspace(float(T_HOURS[0]), float(T_HOURS[-1]), int_grid)
    q_dense = q_interp(t_dense)
    c_dense = c_interp(t_dense)
    s_dense = q_dense * c_dense  # kg/s

    # t 轴单位为小时，先得到 kg/s * h，再乘 3600 变为 kg
    m_kg = trapezoid_integral(s_dense, t_dense) * 3600.0
    return t_dense, q_dense, c_dense, s_dense, m_kg


def r2_score(y_true, y_pred):
    """计算 R^2。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sse = float(np.sum((y_true - y_pred) ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    if sst <= 0:
        return 1.0
    return 1.0 - sse / sst


def fit_quadratic_model(q_data, s_data):
    """对给定 (Q,S) 数据拟合二次模型 S=a0+a1*Q+a2*Q^2。"""
    q_data = np.asarray(q_data, dtype=float)
    s_data = np.asarray(s_data, dtype=float)
    a2, a1, a0 = np.polyfit(q_data, s_data, deg=2)
    s_hat = a0 + a1 * q_data + a2 * (q_data ** 2)
    return {
        "a0": float(a0),
        "a1": float(a1),
        "a2": float(a2),
        "s_hat": s_hat,
        "r2": float(r2_score(s_data, s_hat)),
        "q_min": float(np.min(q_data)),
        "q_max": float(np.max(q_data)),
    }


def quadratic_predict(model_coef, q_values):
    """按给定二次模型系数计算预测值。"""
    q_values = np.asarray(q_values, dtype=float)
    return model_coef["a0"] + model_coef["a1"] * q_values + model_coef["a2"] * (q_values ** 2)


def fit_s_q_relationship():
    """拟合 S-Q 关系：线性模型 + 整体二次模型 + 分段二次模型。"""
    s_obs = Q_DATA * C_DATA

    # 线性模型 S = a + bQ
    b_lin, a_lin = np.polyfit(Q_DATA, s_obs, deg=1)
    s_lin = a_lin + b_lin * Q_DATA
    r2_lin = r2_score(s_obs, s_lin)

    # 整体二次模型
    quad_all = fit_quadratic_model(Q_DATA, s_obs)

    # 以峰值流量为分界进行分段二次拟合：
    # 峰值点仅放在第一段，第二段从峰值后的第一个点开始
    peak_idx = int(np.argmax(Q_DATA))
    q_inc = Q_DATA[: peak_idx + 1]
    s_inc = s_obs[: peak_idx + 1]
    q_dec = Q_DATA[peak_idx + 1:]
    s_dec = s_obs[peak_idx + 1:]
    quad_inc = fit_quadratic_model(q_inc, s_inc)
    quad_dec = fit_quadratic_model(q_dec, s_dec)

    return {
        "s_obs": s_obs,
        "linear": {
            "a": float(a_lin),
            "b": float(b_lin),
            "s_hat": s_lin,
            "r2": float(r2_lin),
        },
        "quadratic": quad_all,
        "peak_index": peak_idx,
        "peak_hour": float(T_HOURS[peak_idx]),
        "peak_flow": float(Q_DATA[peak_idx]),
        "segment_range": {
            "increase": {"q_start": float(q_inc[0]), "q_end": float(q_inc[-1])},
            "decrease": {"q_start": float(q_dec[0]), "q_end": float(q_dec[-1])},
        },
        "segment_quadratic": {
            "increase": quad_inc,
            "decrease": quad_dec,
        },
    }


def print_report(method, int_grid, query_rows, m_kg, fit_result):
    """打印主结果。"""
    m_ton = m_kg / 1000.0
    m_wan_ton = m_ton / 10000.0

    print("=== 例题 5.22 求解结果 ===")
    print(f"插值方式: {method}")
    print(f"积分网格点数: {int_grid}")
    print(f"观测起点: {START_TIME.strftime('%m-%d %H:%M')}")
    print(f"观测终点: {format_time(T_HOURS[-1])}")

    print("\n任意时刻排沙量估计（S=Q*C）：")
    print("  时刻(相对小时)  日期时间      Q(m^3/s)    C(kg/m^3)      S(kg/s)")
    for h, qv, cv, sv in query_rows:
        print(f"  {h:10.2f}   {format_time(h):>11s}   {qv:9.3f}   {cv:10.3f}   {sv:11.3f}")

    print("\n总排沙量估计：")
    print(f"  M = {m_kg:.6f} kg")
    print(f"    = {m_ton:.6f} t")
    print(f"    = {m_wan_ton:.6f} 万t")

    lin = fit_result["linear"]
    quad = fit_result["quadratic"]
    seg_inc = fit_result["segment_quadratic"]["increase"]
    seg_dec = fit_result["segment_quadratic"]["decrease"]
    seg_rng = fit_result["segment_range"]
    print("\n排沙量-流量关系拟合：")
    print(f"  线性模型: S = {lin['a']:.6f} + {lin['b']:.6f} * Q, R^2={lin['r2']:.6f}")
    print(
        "  整体二次模型: "
        f"S = {quad['a0']:.6f} + {quad['a1']:.6f} * Q + {quad['a2']:.10f} * Q^2, "
        f"R^2={quad['r2']:.6f}"
    )
    print("  分段二次模型（按时间先增后减分段，峰值点仅放在增大段）：")
    print(
        f"    增大段 Q: {seg_rng['increase']['q_start']:.0f} -> {seg_rng['increase']['q_end']:.0f}, "
        f"S = {seg_inc['a0']:.6f} + {seg_inc['a1']:.6f} * Q + {seg_inc['a2']:.10f} * Q^2, "
        f"R^2={seg_inc['r2']:.6f}"
    )
    print(
        f"    减小段 Q: {seg_rng['decrease']['q_start']:.0f} -> {seg_rng['decrease']['q_end']:.0f}, "
        f"S = {seg_dec['a0']:.6f} + {seg_dec['a1']:.6f} * Q + {seg_dec['a2']:.10f} * Q^2, "
        f"R^2={seg_dec['r2']:.6f}"
    )


def plot_results(t_dense, q_dense, c_dense, s_dense, fit_result):
    """绘制时序图和 S-Q 拟合图。"""
    s_obs = fit_result["s_obs"]
    seg_inc = fit_result["segment_quadratic"]["increase"]
    seg_dec = fit_result["segment_quadratic"]["decrease"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

    axes[0].plot(t_dense, q_dense, color="#1f77b4", linewidth=2.0, label="Q(t) 插值曲线")
    axes[0].scatter(T_HOURS, Q_DATA, color="#1f77b4", s=28, alpha=0.85, label="Q 观测点")
    axes[0].set_ylabel("Q (m^3/s)")
    axes[0].set_title("调水调沙时序曲线")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(t_dense, c_dense, color="#d62728", linewidth=2.0, label="C(t) 插值曲线")
    axes[1].scatter(T_HOURS, C_DATA, color="#d62728", s=28, alpha=0.85, label="C 观测点")
    axes[1].set_ylabel("C (kg/m^3)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    axes[2].plot(t_dense, s_dense, color="#2ca02c", linewidth=2.0, label="S(t)=Q(t)C(t)")
    axes[2].scatter(T_HOURS, s_obs, color="#2ca02c", s=28, alpha=0.85, label="S 观测点")
    axes[2].set_xlabel("相对起点时间 (小时)")
    axes[2].set_ylabel("S (kg/s)")
    axes[2].grid(alpha=0.3)
    axes[2].legend()

    plt.tight_layout()

    fig2, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 5.2), sharey=True)

    q_inc = Q_DATA[: fit_result["peak_index"] + 1]
    s_inc = s_obs[: fit_result["peak_index"] + 1]
    q_dec = Q_DATA[fit_result["peak_index"] + 1:]
    s_dec = s_obs[fit_result["peak_index"] + 1:]

    q_inc_grid = np.linspace(float(np.min(q_inc)), float(np.max(q_inc)), 220)
    q_dec_grid = np.linspace(float(np.min(q_dec)), float(np.max(q_dec)), 220)
    s_inc_grid = quadratic_predict(seg_inc, q_inc_grid)
    s_dec_grid = quadratic_predict(seg_dec, q_dec_grid)

    ax_left.scatter(q_inc, s_inc, color="#1f77b4", s=42, label="增大段观测点")
    ax_left.plot(
        q_inc_grid,
        s_inc_grid,
        color="#2ca02c",
        linewidth=2.0,
        label=f"二次拟合 R^2={seg_inc['r2']:.4f}",
    )
    ax_left.set_title("增大段拟合")
    ax_left.set_xlabel("Q (m^3/s)")
    ax_left.set_ylabel("S (kg/s)")
    ax_left.grid(alpha=0.3)
    ax_left.legend()

    ax_right.scatter(q_dec, s_dec, color="#ff7f0e", s=42, label="减小段观测点")
    ax_right.plot(
        q_dec_grid,
        s_dec_grid,
        color="#9467bd",
        linewidth=2.0,
        label=f"二次拟合 R^2={seg_dec['r2']:.4f}",
    )
    ax_right.set_title("减小段拟合")
    ax_right.set_xlabel("Q (m^3/s)")
    ax_right.grid(alpha=0.3)
    ax_right.legend()

    fig2.suptitle("排沙量 S 与水流量 Q 的分段二次拟合", y=1.02)
    plt.tight_layout()
    plt.show()


def solve(interp_method="pchip", int_grid=6001, query_hours=None, show_plot=True):
    q_interp, c_interp = build_interpolators(interp_method)
    query_rows = evaluate_queries(query_hours, q_interp, c_interp)
    t_dense, q_dense, c_dense, s_dense, m_kg = estimate_total_sediment(q_interp, c_interp, int_grid=int_grid)
    fit_result = fit_s_q_relationship()

    print_report(
        method=interp_method,
        int_grid=int_grid,
        query_rows=query_rows,
        m_kg=m_kg,
        fit_result=fit_result,
    )

    if show_plot:
        plot_results(t_dense, q_dense, c_dense, s_dense, fit_result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="例题5.22 调水调沙数据建模")
    parser.add_argument(
        "--interp",
        type=str,
        choices=["pchip", "cubic"],
        default="pchip",
        help="时间插值方式：pchip 或 cubic（默认 pchip）",
    )
    parser.add_argument("--int-grid", type=int, default=6001, help="总排沙量积分网格点数，默认 6001")
    parser.add_argument(
        "--query-hours",
        nargs="*",
        type=float,
        default=None,
        help="查询时刻（相对起点小时，可多个），如 --query-hours 6 33 87.5",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    solve(
        interp_method=args.interp,
        int_grid=args.int_grid,
        query_hours=args.query_hours,
        show_plot=not args.no_plot,
    )
