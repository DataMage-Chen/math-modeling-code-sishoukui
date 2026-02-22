"""
习题 2.10：三阶段面试排序（排列流水调度 MILP）。

运行：
  python ch02/hw02_10/solution.py
"""

from gurobipy import GRB, Model, quicksum


def format_clock(minutes_after_8):
    total = 8 * 60 + int(round(minutes_after_8))
    hh = total // 60
    mm = total % 60
    return f"{hh:02d}:{mm:02d}"


def build_data():
    students = ["甲", "乙", "丙", "丁"]
    stages = ["秘书初试", "主管复试", "经理面试"]
    durations = {
        "甲": [14, 16, 21],
        "乙": [19, 17, 10],
        "丙": [10, 15, 12],
        "丁": [9, 12, 13],
    }
    return students, stages, durations


def decode_order(y, students, positions):
    order = []
    for k in positions:
        chosen = None
        for s in students:
            if y[s, k].X > 0.5:
                chosen = s
                break
        if chosen is None:
            raise RuntimeError(f"位置{k+1}未找到对应同学。")
        order.append(chosen)
    return order


def build_schedule(order, stages, durations):
    n = len(order)
    m = len(stages)
    start = [[0] * m for _ in range(n)]
    end = [[0] * m for _ in range(n)]

    for i, stu in enumerate(order):
        for j in range(m):
            prev_job = end[i - 1][j] if i > 0 else 0
            prev_stage = end[i][j - 1] if j > 0 else 0
            start[i][j] = max(prev_job, prev_stage)
            end[i][j] = start[i][j] + durations[stu][j]
    return start, end


def solve():
    students, stages, durations = build_data()
    positions = range(len(students))
    stage_ids = range(len(stages))

    model = Model("hw02_10_flowshop")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    y = model.addVars(students, positions, vtype=GRB.BINARY, name="y")
    c = model.addVars(positions, stage_ids, lb=0.0, vtype=GRB.CONTINUOUS, name="C")

    for s in students:
        model.addConstr(quicksum(y[s, k] for k in positions) == 1, name=f"assign_{s}")

    for k in positions:
        model.addConstr(quicksum(y[s, k] for s in students) == 1, name=f"pos_{k+1}")

    for k in positions:
        for j in stage_ids:
            p_kj = quicksum(durations[s][j] * y[s, k] for s in students)
            model.addConstr(c[k, j] >= p_kj, name=f"base_{k+1}_{j+1}")
            if k > 0:
                model.addConstr(c[k, j] >= c[k - 1, j] + p_kj, name=f"seq_{k+1}_{j+1}")
            if j > 0:
                model.addConstr(c[k, j] >= c[k, j - 1] + p_kj, name=f"flow_{k+1}_{j+1}")

    model.setObjective(c[len(students) - 1, len(stages) - 1], GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    order = decode_order(y, students, positions)
    start, end = build_schedule(order, stages, durations)
    makespan = end[-1][-1]

    print("=== 习题 2.10 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最短总用时: {int(round(makespan))} 分钟")
    print(f"最早离开时间: {format_clock(makespan)}")
    print(f"面试顺序: {' -> '.join(order)}")

    print("详细时刻表（起点 08:00）：")
    for i, stu in enumerate(order):
        segments = []
        for j, stage in enumerate(stages):
            segments.append(
                f"{stage} {format_clock(start[i][j])}-{format_clock(end[i][j])}"
            )
        print(f"  {stu}: " + " | ".join(segments))


if __name__ == "__main__":
    solve()
