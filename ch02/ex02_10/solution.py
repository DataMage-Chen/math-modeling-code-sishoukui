"""
例题 2.10：比赛项目顺序优化（0-1 整数规划，TSP 路径转化）。

题意：给定 40 名运动员对 14 个项目的报名关系，安排项目顺序，
使相邻两项之间“同时报名两项的运动员人数”总和最小（即连续参赛总人次最小）。

运行示例：
  python ch02/ex02_10/solution.py
  python ch02/ex02_10/solution.py --show-overlap
"""

import argparse

from gurobipy import GRB, Model, quicksum


SIGNUP_MATRIX = [
    [0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0],  # 运动员 1
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0],  # 运动员 2
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 运动员 3
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 运动员 4
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1],  # 运动员 5
    [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 运动员 6
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],  # 运动员 7
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 运动员 8
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],  # 运动员 9
    [1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 运动员 10
    [0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 运动员 11
    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0],  # 运动员 12
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],  # 运动员 13
    [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 运动员 14
    [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0],  # 运动员 15
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0],  # 运动员 16
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # 运动员 17
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 运动员 18
    [0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 运动员 19
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 运动员 20
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],  # 运动员 21
    [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 运动员 22
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0],  # 运动员 23
    [0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],  # 运动员 24
    [1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # 运动员 25
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # 运动员 26
    [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0],  # 运动员 27
    [0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # 运动员 28
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0],  # 运动员 29
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # 运动员 30
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0],  # 运动员 31
    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],  # 运动员 32
    [0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # 运动员 33
    [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],  # 运动员 34
    [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # 运动员 35
    [0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # 运动员 36
    [1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],  # 运动员 37
    [0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1],  # 运动员 38
    [0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0],  # 运动员 39
    [0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0],  # 运动员 40
]


def validate_signup_matrix(signup_matrix):
    athlete_count = len(signup_matrix)
    if athlete_count == 0:
        raise ValueError("报名矩阵不能为空。")

    event_count = len(signup_matrix[0])
    for row in signup_matrix:
        if len(row) != event_count:
            raise ValueError("报名矩阵每一行的项目数必须一致。")
        if any(value not in (0, 1) for value in row):
            raise ValueError("报名矩阵只能包含 0/1。")

    return athlete_count, event_count


def build_overlap_matrix(signup_matrix):
    athlete_count, event_count = validate_signup_matrix(signup_matrix)
    overlap = [[0 for _ in range(event_count)] for _ in range(event_count)]

    for i in range(event_count):
        for j in range(event_count):
            if i == j:
                continue
            overlap[i][j] = sum(
                signup_matrix[m][i] * signup_matrix[m][j] for m in range(athlete_count)
            )
    return overlap


def print_overlap_matrix(overlap):
    event_count = len(overlap)
    print("相邻冲突权重矩阵 W（元素为同时报名两项的人数）：")
    header = "     " + " ".join(f"{j + 1:>3d}" for j in range(event_count))
    print(header)
    for i in range(event_count):
        row_text = " ".join(f"{overlap[i][j]:>3d}" for j in range(event_count))
        print(f"{i + 1:>3d}: {row_text}")


def solve(show_overlap=False):
    signup_matrix = SIGNUP_MATRIX
    athlete_count, event_count = validate_signup_matrix(signup_matrix)
    overlap = build_overlap_matrix(signup_matrix)
    dummy = event_count
    nodes = list(range(event_count + 1))
    real_events = list(range(event_count))

    if show_overlap:
        print_overlap_matrix(overlap)
        print()

    arcs = [(i, j) for i in nodes for j in nodes if i != j]

    model = Model("ex02_10_schedule_order")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    x = model.addVars(arcs, vtype=GRB.BINARY, name="x")
    u = model.addVars(real_events, lb=1.0, ub=float(event_count), name="u")

    for i in nodes:
        model.addConstr(
            quicksum(x[i, j] for j in nodes if j != i) == 1,
            name=f"out_{i + 1}",
        )
    for j in nodes:
        model.addConstr(
            quicksum(x[i, j] for i in nodes if i != j) == 1,
            name=f"in_{j + 1}",
        )

    for i in real_events:
        for j in real_events:
            if i == j:
                continue
            model.addConstr(
                u[i] - u[j] + event_count * x[i, j] <= event_count - 1,
                name=f"mtz_{i + 1}_{j + 1}",
            )

    model.setObjective(
        quicksum(
            overlap[i][j] * x[i, j]
            for i in real_events
            for j in real_events
            if i != j
        ),
        GRB.MINIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"优化结束，状态码：{model.status}")
        return

    successor = {}
    for i in nodes:
        for j in nodes:
            if i != j and x[i, j].X > 0.5:
                successor[i] = j
                break

    sequence = []
    visited = set()
    current = successor[dummy]
    while current != dummy:
        if current in visited:
            raise RuntimeError("提取到重复节点，可能存在子回路。")
        visited.add(current)
        sequence.append(current)
        current = successor[current]

    if len(sequence) != event_count:
        raise RuntimeError("提取的比赛顺序长度不正确。")

    pair_counts = []
    for k in range(event_count - 1):
        i = sequence[k]
        j = sequence[k + 1]
        pair_counts.append((i, j, overlap[i][j]))

    total_person_times = sum(item[2] for item in pair_counts)
    athlete_once_count = 0
    for m in range(athlete_count):
        has_consecutive = any(
            signup_matrix[m][i] == 1 and signup_matrix[m][j] == 1 for i, j, _ in pair_counts
        )
        if has_consecutive:
            athlete_once_count += 1

    sequence_display = " -> ".join(str(event + 1) for event in sequence)

    print("=== 例题 2.10 求解结果 ===")
    print(f"ObjVal: {model.ObjVal:.10f}")
    print(f"ObjBound: {model.ObjBound:.10f}")
    print(f"MIPGap: {model.MIPGap:.3e}")
    print(f"最优连续参赛人次: {total_person_times}")
    print(f"比赛项目顺序: {sequence_display}")
    print("相邻项目冲突（共同报名人数）：")
    for i, j, count in pair_counts:
        print(f"  项目{i + 1} -> 项目{j + 1}: {count}")

    print("统计信息：")
    print(f"  总运动员数: {athlete_count}")
    print(f"  至少出现一次连续参赛的运动员人数: {athlete_once_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解例题 2.10（比赛顺序优化）。")
    parser.add_argument(
        "--show-overlap",
        action="store_true",
        help="先打印 14x14 的相邻冲突权重矩阵。",
    )
    args = parser.parse_args()
    solve(show_overlap=args.show_overlap)
