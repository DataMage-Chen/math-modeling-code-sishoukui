"""
习题 5.9：水箱水流量反演

运行示例：
  python ch05/hw05_09/solution.py
  python ch05/hw05_09/solution.py --harmonics 4 --no-plot
  python ch05/hw05_09/solution.py --query-times 0 36000 38000 80000
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


# 常量
E_TO_M = 0.3024                  # 1E = 30.24cm = 0.3024m
LEVEL_UNIT_TO_M = 1e-2 * E_TO_M  # 表中“水位/10^-2E”到米的换算
DIAMETER_E = 57.0
DIAMETER_M = DIAMETER_E * E_TO_M
AREA = np.pi * (DIAMETER_M / 2.0) ** 2
DAY_SEC = 86400.0


# 表 5.18：时间(s), 水位(10^-2E)；None 表示“泵水”标记
RAW_DATA = [
    (0, 3175), (3316, 3110), (6635, 3054), (10619, 2994), (13937, 2947), (17921, 2892),
    (21240, 2850), (25223, 2795), (28543, 2752), (32284, 2697), (35932, None), (39332, None),
    (39435, 3550), (43318, 3445), (44636, 3350), (49953, 3260), (53936, 3167), (57254, 3087),
    (60574, 3012), (64554, 2927), (68535, 2842), (71854, 2767), (75021, 2697), (79254, None),
    (82649, None), (85968, 3475), (89953, 3397), (93270, 3340),
]


def parse_raw_data(raw_data):
    """解析观测点和泵水区间。"""
    obs_t, obs_level_raw = [], []
    pump_marks = []

    for t, level in raw_data:
        if level is None:
            pump_marks.append(float(t))
        else:
            obs_t.append(float(t))
            obs_level_raw.append(float(level))

    if len(pump_marks) % 2 != 0:
        raise ValueError("泵水标记点数量应为偶数（成对的开始/结束）。")

    pump_intervals = []
    for i in range(0, len(pump_marks), 2):
        t1, t2 = pump_marks[i], pump_marks[i + 1]
        if t2 <= t1:
            raise ValueError(f"泵水区间异常: ({t1}, {t2})")
        pump_intervals.append((t1, t2))

    obs_t = np.array(obs_t, dtype=float)
    obs_level_raw = np.array(obs_level_raw, dtype=float)
    obs_h_m = obs_level_raw * LEVEL_UNIT_TO_M
    return obs_t, obs_level_raw, obs_h_m, pump_intervals


def pump_on_flag(t, pump_intervals):
    """判断时刻 t 是否处于泵水状态。"""
    t = np.asarray(t, dtype=float)
    flag = np.zeros_like(t, dtype=bool)
    for start, end in pump_intervals:
        flag |= (t >= start) & (t <= end)
    return flag


def pump_active_duration(t, pump_intervals):
    """计算 [0,t] 内累计泵水时长（秒）。"""
    t = np.asarray(t, dtype=float)
    acc = np.zeros_like(t, dtype=float)
    for start, end in pump_intervals:
        acc += np.clip(np.minimum(t, end) - start, 0.0, end - start)
    return acc


def demand_flow(t, coeff, day_sec=DAY_SEC):
    """
    日周期流出流量模型（傅里叶级数）:
    coeff = [c0, c1, d1, c2, d2, ...]
    """
    t = np.asarray(t, dtype=float)
    c0 = coeff[0]
    y = np.full_like(t, c0, dtype=float)
    omega = 2.0 * np.pi / day_sec

    idx = 1
    k = 1
    while idx + 1 < len(coeff):
        ck = coeff[idx]
        dk = coeff[idx + 1]
        y += ck * np.cos(k * omega * t) + dk * np.sin(k * omega * t)
        idx += 2
        k += 1
    return y


def demand_integral_from_zero(t, coeff, day_sec=DAY_SEC):
    """
    计算 I(t)=∫_0^t f(tau) dtau 的解析表达式。
    """
    t = np.asarray(t, dtype=float)
    c0 = coeff[0]
    omega = 2.0 * np.pi / day_sec

    integral = c0 * t
    idx = 1
    k = 1
    while idx + 1 < len(coeff):
        ck = coeff[idx]
        dk = coeff[idx + 1]
        kw = k * omega
        integral += ck * np.sin(kw * t) / kw
        integral += dk * (1.0 - np.cos(kw * t)) / kw
        idx += 2
        k += 1
    return integral


def predict_height(t, h0, q_pump, coeff, area, pump_intervals):
    """由质量守恒积分得到水位预测值。"""
    pumped = pump_active_duration(t, pump_intervals)
    used = demand_integral_from_zero(t, coeff)
    return h0 + (q_pump * pumped - used) / area


def interval_overlap(a1, a2, b1, b2):
    """判断区间是否有重叠。"""
    return max(a1, b1) < min(a2, b2)


def initial_guess(obs_t, obs_h_m, pump_intervals, harmonics):
    """根据非泵水阶段的有限差分斜率构造初值。"""
    f_samples = []
    t_samples = []

    for i in range(obs_t.size - 1):
        t1, t2 = obs_t[i], obs_t[i + 1]
        cross_pump = any(interval_overlap(t1, t2, s, e) for s, e in pump_intervals)
        if cross_pump:
            continue

        dt = t2 - t1
        if dt <= 0:
            continue
        dh = obs_h_m[i + 1] - obs_h_m[i]
        f_est = -AREA * dh / dt  # 泵关时 A*dh/dt = -f
        if np.isfinite(f_est) and f_est > 0:
            f_samples.append(float(f_est))
            t_samples.append(float((t1 + t2) * 0.5))

    if len(f_samples) == 0:
        c0 = 0.02
        q_p = 0.20
        coeff = np.zeros(1 + 2 * harmonics, dtype=float)
        coeff[0] = c0
        return q_p, coeff

    f_samples = np.array(f_samples, dtype=float)
    t_samples = np.array(t_samples, dtype=float)

    # 先做一个傅里叶线性回归初值
    omega = 2.0 * np.pi / DAY_SEC
    cols = [np.ones_like(t_samples)]
    for k in range(1, harmonics + 1):
        cols.append(np.cos(k * omega * t_samples))
        cols.append(np.sin(k * omega * t_samples))
    design = np.column_stack(cols)
    coef_ls, *_ = np.linalg.lstsq(design, f_samples, rcond=None)

    coeff = np.array(coef_ls, dtype=float)
    coeff[0] = max(coeff[0], 1e-6)  # c0 至少为正

    max_f = float(np.max(f_samples))
    q_p = max(max_f * 1.5, coeff[0] + 0.05, 0.05)
    return q_p, coeff


def fit_model(obs_t, obs_h_m, pump_intervals, harmonics=3, regularization=1e-3):
    """拟合模型参数 [q_p, c0, c1, d1, ...]。"""
    h0 = float(obs_h_m[0])
    q_init, coeff_init = initial_guess(obs_t, obs_h_m, pump_intervals, harmonics=harmonics)
    x0 = np.concatenate(([q_init], coeff_init))

    # 参数下界：q_p>0, c0>=0，其余谐波项不受限
    lower = np.concatenate(([1e-8, 1e-8], np.full(2 * harmonics, -np.inf)))
    upper = np.full_like(lower, np.inf, dtype=float)

    def residual(params):
        q_p = float(params[0])
        coeff = np.array(params[1:], dtype=float)
        h_hat = predict_height(obs_t, h0, q_p, coeff, AREA, pump_intervals)
        data_res = h_hat - obs_h_m

        # 轻微正则抑制高频谐波震荡
        reg = regularization * coeff[1:]
        return np.concatenate([data_res, reg])

    res = least_squares(
        residual,
        x0=x0,
        bounds=(lower, upper),
        method="trf",
        max_nfev=60000,
        x_scale="jac",
    )

    q_hat = float(res.x[0])
    coeff_hat = np.array(res.x[1:], dtype=float)
    h_hat_obs = predict_height(obs_t, h0, q_hat, coeff_hat, AREA, pump_intervals)

    resid = obs_h_m - h_hat_obs
    sse = float(np.sum(resid ** 2))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sst = float(np.sum((obs_h_m - np.mean(obs_h_m)) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else 1.0

    return {
        "success": bool(res.success),
        "message": str(res.message),
        "status": int(res.status),
        "nfev": int(res.nfev),
        "q_pump": q_hat,
        "coeff": coeff_hat,
        "h0": h0,
        "h_hat_obs": h_hat_obs,
        "sse": sse,
        "rmse": rmse,
        "r2": r2,
    }


def print_report(result, obs_t, obs_level_raw, obs_h_m, pump_intervals, query_times):
    """打印核心结果。"""
    q_p = result["q_pump"]
    coeff = result["coeff"]

    print("=== 习题 5.9 求解结果 ===")
    print(f"水箱直径 D = {DIAMETER_M:.6f} m, 横截面积 A = {AREA:.6f} m^2")
    print("泵水区间（s）:")
    for s, e in pump_intervals:
        print(f"  [{s:.0f}, {e:.0f}]，持续 {e - s:.0f} s")

    print("\n优化状态：")
    print(f"  success={result['success']}, status={result['status']}, nfev={result['nfev']}")
    print(f"  message={result['message']}")

    print("\n估计参数：")
    print(f"  泵水常流量 q_p = {q_p:.10f} m^3/s = {q_p * 1000.0:.3f} L/s")
    print(f"  流出模型基准项 c0 = {coeff[0]:.10f} m^3/s")
    idx = 1
    k = 1
    while idx + 1 < coeff.size:
        print(f"  谐波{k}: c{k}={coeff[idx]:+.10e}, d{k}={coeff[idx+1]:+.10e}")
        idx += 2
        k += 1

    print("\n水位拟合误差（单位：m）：")
    print(f"  SSE={result['sse']:.12e}, RMSE={result['rmse']:.12e}, R^2={result['r2']:.12f}")

    t_dense = np.linspace(float(obs_t.min()), float(obs_t.max()), 1600)
    f_dense = demand_flow(t_dense, coeff)
    max_f = float(np.max(f_dense))
    min_f = float(np.min(f_dense))
    print("\n流出流量统计：")
    print(f"  min f(t) = {min_f:.10f} m^3/s = {min_f * 1000.0:.3f} L/s")
    print(f"  max f(t) = {max_f:.10f} m^3/s = {max_f * 1000.0:.3f} L/s")
    print(f"  假设(3)校验 max f(t) < q_p ? {max_f < q_p}")

    print("\n任意时刻估计（含泵水期间）：")
    print("   t(s)   泵状态      f(t)[L/s]    q_in[L/s]   净入流(q_in-f)[L/s]")
    q_times = np.array(query_times, dtype=float)
    f_q = demand_flow(q_times, coeff)
    pump_flag = pump_on_flag(q_times, pump_intervals)
    q_in = np.where(pump_flag, q_p, 0.0)
    for t, flag, fv, qv in zip(q_times, pump_flag, f_q, q_in):
        state = "泵开" if flag else "泵关"
        net = (qv - fv) * 1000.0
        print(f"{t:7.0f}   {state:>3s}     {fv * 1000.0:10.3f}   {qv * 1000.0:10.3f}         {net:10.3f}")

    h_fit_raw = result["h_hat_obs"] / LEVEL_UNIT_TO_M
    max_abs_level_err = float(np.max(np.abs(h_fit_raw - obs_level_raw)))
    print("\n水位点拟合（原表单位 10^-2E）最大绝对误差：")
    print(f"  max|h_hat-h_obs| = {max_abs_level_err:.6f}")


def plot_result(result, obs_t, obs_level_raw, pump_intervals):
    """绘制水位拟合与流量曲线。"""
    q_p = result["q_pump"]
    coeff = result["coeff"]
    h0 = result["h0"]

    t_min = float(obs_t.min())
    t_max = float(obs_t.max())
    t_dense = np.linspace(t_min, t_max, 1800)

    h_dense = predict_height(t_dense, h0, q_p, coeff, AREA, pump_intervals)
    h_dense_raw = h_dense / LEVEL_UNIT_TO_M
    f_dense_lps = demand_flow(t_dense, coeff) * 1000.0
    q_in_dense_lps = np.where(pump_on_flag(t_dense, pump_intervals), q_p * 1000.0, 0.0)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True)

    # 水位图
    ax1.scatter(obs_t, obs_level_raw, color="#1f77b4", s=35, zorder=4, label="观测水位")
    ax1.plot(t_dense, h_dense_raw, color="#d62728", linewidth=2.0, label="模型拟合水位")
    for i, (s, e) in enumerate(pump_intervals):
        ax1.axvspan(s, e, color="#ffbb78", alpha=0.28, label="泵水区间" if i == 0 else None)
    ax1.set_ylabel("水位 (10^-2 E)")
    ax1.set_title("习题 5.9：水位拟合与流量反演")
    ax1.grid(alpha=0.3)
    ax1.legend()

    # 流量图
    ax2.plot(t_dense, f_dense_lps, color="#2ca02c", linewidth=2.0, label="估计流出流量 f(t)")
    ax2.plot(t_dense, q_in_dense_lps, color="#9467bd", linewidth=1.8, linestyle="--", label="泵入流量 q_in(t)")
    for i, (s, e) in enumerate(pump_intervals):
        ax2.axvspan(s, e, color="#ffbb78", alpha=0.28, label="泵水区间" if i == 0 else None)
    ax2.set_xlabel("时间 t (s)")
    ax2.set_ylabel("流量 (L/s)")
    ax2.grid(alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.show()


def solve(harmonics=3, regularization=1e-3, query_times=None, query_step=21600.0, show_plot=True):
    obs_t, obs_level_raw, obs_h_m, pump_intervals = parse_raw_data(RAW_DATA)
    result = fit_model(
        obs_t=obs_t,
        obs_h_m=obs_h_m,
        pump_intervals=pump_intervals,
        harmonics=harmonics,
        regularization=regularization,
    )

    if query_times is None:
        t_end = float(obs_t.max())
        query_times = list(np.arange(0.0, t_end + 1e-9, query_step))
        for s, e in pump_intervals:
            query_times.extend([s, 0.5 * (s + e), e])
        query_times = sorted(set(float(t) for t in query_times))

    print_report(
        result=result,
        obs_t=obs_t,
        obs_level_raw=obs_level_raw,
        obs_h_m=obs_h_m,
        pump_intervals=pump_intervals,
        query_times=query_times,
    )

    if show_plot:
        plot_result(result, obs_t, obs_level_raw, pump_intervals)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="习题5.9 水箱流量反演")
    parser.add_argument("--harmonics", type=int, default=3, help="傅里叶谐波阶数 K，默认 3")
    parser.add_argument(
        "--regularization",
        type=float,
        default=1e-3,
        help="谐波项L2正则系数，默认 1e-3",
    )
    parser.add_argument(
        "--query-times",
        nargs="*",
        type=float,
        default=None,
        help="指定查询时刻（秒），如 --query-times 0 36000 38000",
    )
    parser.add_argument(
        "--query-step",
        type=float,
        default=21600.0,
        help="未显式给出查询时刻时，默认按该步长自动查询（秒），默认 21600",
    )
    parser.add_argument("--no-plot", action="store_true", help="不显示图像")
    args = parser.parse_args()

    if args.harmonics < 1:
        raise SystemExit("harmonics 必须 >= 1")

    solve(
        harmonics=args.harmonics,
        regularization=args.regularization,
        query_times=args.query_times,
        query_step=args.query_step,
        show_plot=not args.no_plot,
    )

