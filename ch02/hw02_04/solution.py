"""
习题 2.4：体操团体赛阵容优化。

功能：
1) 按最低分估算求最优阵容；
2) 按均值估算求最优阵容；
3) 第(2)问按“同学版”思路建模：
   在满足总分 >= 门槛的前提下，选择一个具体得分场景 z_{i,e,k}，
   使该场景联合概率（乘积）最大。
4) 对第(2)问得到的阵容，再做非正态近似的精确离散评估：
   计算 P(S>=门槛)、E[S] 与 90% 把握水平。

运行示例：
  python ch02/hw02_04/solution.py
  python ch02/hw02_04/solution.py --target 236.2
  python ch02/hw02_04/solution.py --no-chance
"""

import argparse
import math
from collections import defaultdict

from gurobipy import GRB, Model, quicksum


EVENTS = ["高低杠", "平衡木", "跳马", "自由体操"]
ATHLETES = list(range(1, 11))
SCALE = 10

# 表 2.7 数据：每个项目、每个运动员对应 4 个（得分, 概率）取值
SCORE_DISTS = {
    1: {
        "高低杠": [(8.4, 0.15), (9.5, 0.5), (9.2, 0.25), (9.4, 0.1)],
        "平衡木": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "跳马": [(9.1, 0.1), (9.3, 0.1), (9.5, 0.6), (9.8, 0.2)],
        "自由体操": [(8.7, 0.1), (8.9, 0.2), (9.1, 0.6), (9.9, 0.1)],
    },
    2: {
        "高低杠": [(9.3, 0.1), (9.5, 0.1), (9.6, 0.6), (9.8, 0.2)],
        "平衡木": [(8.4, 0.15), (9.0, 0.5), (9.2, 0.25), (9.4, 0.1)],
        "跳马": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "自由体操": [(8.9, 0.1), (9.1, 0.1), (9.3, 0.6), (9.6, 0.2)],
    },
    3: {
        "高低杠": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "平衡木": [(8.1, 0.1), (9.1, 0.5), (9.3, 0.3), (9.5, 0.1)],
        "跳马": [(8.4, 0.15), (9.5, 0.5), (9.2, 0.25), (9.4, 0.1)],
        "自由体操": [(9.5, 0.1), (9.7, 0.1), (9.8, 0.6), (10.0, 0.2)],
    },
    4: {
        "高低杠": [(8.1, 0.1), (9.1, 0.5), (9.3, 0.3), (9.5, 0.1)],
        "平衡木": [(8.7, 0.1), (8.9, 0.2), (9.1, 0.6), (9.9, 0.1)],
        "跳马": [(9.0, 0.1), (9.4, 0.1), (9.5, 0.5), (9.7, 0.3)],
        "自由体操": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
    },
    5: {
        "高低杠": [(8.4, 0.15), (9.5, 0.5), (9.2, 0.25), (9.4, 0.1)],
        "平衡木": [(9.0, 0.1), (9.2, 0.1), (9.4, 0.6), (9.7, 0.2)],
        "跳马": [(8.3, 0.1), (8.7, 0.1), (8.9, 0.6), (9.3, 0.2)],
        "自由体操": [(9.4, 0.1), (9.6, 0.1), (9.7, 0.6), (9.9, 0.2)],
    },
    6: {
        "高低杠": [(9.4, 0.1), (9.6, 0.1), (9.7, 0.6), (9.9, 0.2)],
        "平衡木": [(8.7, 0.1), (8.9, 0.2), (9.1, 0.6), (9.9, 0.1)],
        "跳马": [(8.5, 0.1), (8.7, 0.1), (8.9, 0.5), (9.1, 0.3)],
        "自由体操": [(8.4, 0.15), (9.5, 0.5), (9.2, 0.25), (9.4, 0.1)],
    },
    7: {
        "高低杠": [(9.5, 0.1), (9.7, 0.1), (9.8, 0.6), (10.0, 0.2)],
        "平衡木": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "跳马": [(8.3, 0.1), (8.7, 0.1), (8.9, 0.6), (9.3, 0.2)],
        "自由体操": [(8.4, 0.1), (8.8, 0.1), (9.2, 0.6), (9.8, 0.2)],
    },
    8: {
        "高低杠": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "平衡木": [(8.8, 0.05), (9.2, 0.05), (9.8, 0.5), (10.0, 0.4)],
        "跳马": [(8.7, 0.1), (8.9, 0.2), (9.1, 0.6), (9.9, 0.1)],
        "自由体操": [(8.2, 0.1), (9.3, 0.5), (9.5, 0.3), (9.8, 0.1)],
    },
    9: {
        "高低杠": [(8.4, 0.15), (9.5, 0.5), (9.2, 0.25), (9.4, 0.1)],
        "平衡木": [(8.4, 0.1), (8.8, 0.1), (9.2, 0.6), (9.8, 0.2)],
        "跳马": [(8.4, 0.1), (8.8, 0.2), (9.0, 0.6), (10.0, 0.1)],
        "自由体操": [(9.3, 0.1), (9.5, 0.1), (9.7, 0.5), (9.9, 0.3)],
    },
    10: {
        "高低杠": [(9.0, 0.1), (9.2, 0.1), (9.4, 0.6), (9.7, 0.2)],
        "平衡木": [(8.1, 0.1), (9.1, 0.5), (9.3, 0.3), (9.5, 0.1)],
        "跳马": [(8.2, 0.1), (9.2, 0.5), (9.4, 0.3), (9.6, 0.1)],
        "自由体操": [(9.1, 0.1), (9.3, 0.1), (9.5, 0.6), (9.8, 0.2)],
    },
}


def validate_distributions():
    for i in ATHLETES:
        for event in EVENTS:
            probs = [prob for _, prob in SCORE_DISTS[i][event]]
            if abs(sum(probs) - 1.0) > 1e-9:
                raise ValueError(f"概率和不为 1：运动员{i}-{event}")


def build_score_table(mode):
    table = {}
    for i in ATHLETES:
        for event in EVENTS:
            pairs = SCORE_DISTS[i][event]
            if mode == "min":
                value = min(score for score, _ in pairs)
            elif mode == "mean":
                value = sum(score * prob for score, prob in pairs)
            else:
                raise ValueError(f"未知模式: {mode}")
            table[i, event] = value
    return table


def create_base_model(model_name):
    model = Model(model_name)
    model.Params.OutputFlag = 0
    model.Params.MIPGap = 1e-9
    model.Params.MIPGapAbs = 1e-9

    a = model.addVars(ATHLETES, vtype=GRB.BINARY, name="all_around")
    x = model.addVars(ATHLETES, EVENTS, vtype=GRB.BINARY, name="x")

    model.addConstr(quicksum(a[i] for i in ATHLETES) == 4, name="all_around_count")

    for event in EVENTS:
        model.addConstr(
            quicksum(x[i, event] for i in ATHLETES) == 6,
            name=f"event_size_{event}",
        )

    for i in ATHLETES:
        for event in EVENTS:
            model.addConstr(x[i, event] >= a[i], name=f"all_req_{i}_{event}")

        # 全能选手 4 项全参加；单项选手 1~3 项
        model.addConstr(
            quicksum(x[i, event] for event in EVENTS) <= 3 + a[i],
            name=f"max_events_{i}",
        )
        model.addConstr(
            quicksum(x[i, event] for event in EVENTS) >= 1 + 3 * a[i],
            name=f"min_events_{i}",
        )

    return model, a, x


def extract_lineup_from_x(a, x):
    all_around = sorted(i for i in ATHLETES if a[i].X > 0.5)
    event_members = {
        event: sorted(i for i in ATHLETES if x[i, event].X > 0.5) for event in EVENTS
    }
    per_athlete_events = {
        i: [event for event in EVENTS if x[i, event].X > 0.5] for i in ATHLETES
    }
    lineup = {
        "all_around": all_around,
        "event_members": event_members,
        "per_athlete_events": per_athlete_events,
    }
    return lineup


def print_lineup_detail(title, lineup, score_table=None):
    print(title)
    print(f"  全能选手（4人）: {lineup['all_around']}")

    for event in EVENTS:
        members = lineup["event_members"][event]
        if score_table is None:
            print(f"  {event}: {members}")
        else:
            event_score = sum(score_table[i, event] for i in members)
            print(f"  {event}: {members}，估算分={event_score:.4f}")

    print("  各运动员参赛项目数：")
    for i in ATHLETES:
        events = lineup["per_athlete_events"][i]
        print(f"    运动员{i}: {len(events)} 项 -> {events}")


def solve_deterministic(mode):
    score_table = build_score_table(mode=mode)
    model, a, x = create_base_model(model_name=f"hw02_04_{mode}")
    model.setObjective(
        quicksum(score_table[i, event] * x[i, event] for i in ATHLETES for event in EVENTS),
        GRB.MAXIMIZE,
    )
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    lineup = extract_lineup_from_x(a, x)
    result = {
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
        "lineup": lineup,
        "score_table": score_table,
    }
    return result


def convolve_distributions(dist_a, dist_b):
    result = defaultdict(float)
    for score_a, prob_a in dist_a.items():
        for score_b, prob_b in dist_b.items():
            result[score_a + score_b] += prob_a * prob_b
    return dict(result)


def solve_chance_most_likely_scenario(target, mip_gap=1e-6):
    """
    第(2)问“同学版”建模：
    在总分不低于 target 的约束下，选择一个具体得分场景，
    使该场景联合概率（乘积）最大。
    """
    model = Model("hw02_04_most_likely_scenario")
    model.Params.OutputFlag = 0
    model.Params.MIPGap = mip_gap
    model.Params.MIPGapAbs = mip_gap

    a = model.addVars(ATHLETES, vtype=GRB.BINARY, name="all_around")
    x = model.addVars(ATHLETES, EVENTS, vtype=GRB.BINARY, name="x")
    z = model.addVars(ATHLETES, EVENTS, range(4), vtype=GRB.BINARY, name="z")

    for event in EVENTS:
        model.addConstr(
            quicksum(x[i, event] for i in ATHLETES) == 6,
            name=f"event_size_{event}",
        )

    model.addConstr(quicksum(a[i] for i in ATHLETES) == 4, name="all_around_count")

    for i in ATHLETES:
        total_events = quicksum(x[i, event] for event in EVENTS)
        # 4a_i <= sum_e x_ie <= 3 + a_i
        model.addConstr(4 * a[i] <= total_events, name=f"lower_events_{i}")
        model.addConstr(total_events <= 3 + a[i], name=f"upper_events_{i}")

        for event in EVENTS:
            # 若参加该项目，则在4个离散得分中恰选其一；不参加则全为0
            model.addConstr(
                quicksum(z[i, event, k] for k in range(4)) == x[i, event],
                name=f"pick_score_{i}_{event}",
            )

    # 场景总分不低于门槛
    model.addConstr(
        quicksum(
            SCORE_DISTS[i][event][k][0] * z[i, event, k]
            for i in ATHLETES
            for event in EVENTS
            for k in range(4)
        )
        >= target,
        name="target_score",
    )

    # max product(prob) 等价于 max sum(log(prob))
    log_obj = quicksum(
        math.log(SCORE_DISTS[i][event][k][1]) * z[i, event, k]
        for i in ATHLETES
        for event in EVENTS
        for k in range(4)
    )
    model.setObjective(log_obj, GRB.MAXIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        return None

    lineup = extract_lineup_from_x(a, x)

    scenario_items = []
    scenario_score = 0.0
    scenario_log_prob = 0.0
    for event in EVENTS:
        for i in lineup["event_members"][event]:
            chosen_k = None
            for k in range(4):
                if z[i, event, k].X > 0.5:
                    chosen_k = k
                    break
            if chosen_k is None:
                raise RuntimeError(f"未找到场景值：运动员{i}-{event}")
            score, prob = SCORE_DISTS[i][event][chosen_k]
            scenario_items.append(
                {"athlete": i, "event": event, "score": score, "prob": prob}
            )
            scenario_score += score
            scenario_log_prob += math.log(prob)

    result = {
        "obj": model.ObjVal,
        "obj_bound": model.ObjBound,
        "mip_gap": model.MIPGap,
        "lineup": lineup,
        "scenario_items": scenario_items,
        "scenario_score": scenario_score,
        "scenario_log_prob": scenario_log_prob,
        "scenario_joint_prob": math.exp(scenario_log_prob),
    }
    return result


def exact_distribution_for_lineup(lineup):
    selected_distributions = []
    for event in EVENTS:
        for i in lineup["event_members"][event]:
            selected_distributions.append(SCORE_DISTS[i][event])

    # 用“十分位整数”避免浮点键误差：例如 236.2 -> 2362
    distribution = {0: 1.0}
    for pairs in selected_distributions:
        next_distribution = defaultdict(float)
        for score10, prob_acc in distribution.items():
            for score, prob in pairs:
                new_score10 = score10 + int(round(score * SCALE))
                next_distribution[new_score10] += prob_acc * prob
        distribution = dict(next_distribution)
    return distribution


def guaranteed_level(distribution, confidence=0.9):
    if not (0 < confidence < 1):
        raise ValueError("confidence 必须在 (0,1) 之间。")

    keys = sorted(distribution.keys())
    survival = 1.0
    best = keys[0]
    for key in keys:
        if survival >= confidence - 1e-12:
            best = key
        survival -= distribution[key]
    return best / SCALE


def evaluate_lineup(lineup, target, confidence):
    distribution = exact_distribution_for_lineup(lineup)
    target10 = int(round(target * SCALE))

    expected = sum(score10 * prob for score10, prob in distribution.items()) / SCALE
    win_prob = sum(prob for score10, prob in distribution.items() if score10 >= target10)
    level = guaranteed_level(distribution, confidence=confidence)

    result = {
        "expected": expected,
        "win_prob": win_prob,
        "confidence_level": level,
        "state_count": len(distribution),
    }
    return result


def solve(target=236.2, confidence=0.9, chance_mipgap=1e-6, run_chance=True):
    validate_distributions()

    min_result = solve_deterministic(mode="min")
    mean_result = solve_deterministic(mode="mean")

    if min_result is None or mean_result is None:
        print("确定性模型未得到最优解，请检查求解器状态。")
        return

    print("=== 习题 2.4 (1) 最低分估算 ===")
    print(f"ObjVal: {min_result['obj']:.10f}")
    print(f"ObjBound: {min_result['obj_bound']:.10f}")
    print(f"MIPGap: {min_result['mip_gap']:.3e}")
    print_lineup_detail(
        title="最低分估算下的最优阵容：",
        lineup=min_result["lineup"],
        score_table=min_result["score_table"],
    )

    print("\n=== 习题 2.4 (1) 均值估算 ===")
    print(f"ObjVal: {mean_result['obj']:.10f}")
    print(f"ObjBound: {mean_result['obj_bound']:.10f}")
    print(f"MIPGap: {mean_result['mip_gap']:.3e}")
    print_lineup_detail(
        title="均值估算下的最优阵容：",
        lineup=mean_result["lineup"],
        score_table=mean_result["score_table"],
    )

    if not run_chance:
        return

    chance_result = solve_chance_most_likely_scenario(target=target, mip_gap=chance_mipgap)
    if chance_result is None:
        print("\n第(2)问模型未得到最优解，请检查求解器状态。")
        return

    eval_result = evaluate_lineup(
        lineup=chance_result["lineup"],
        target=target,
        confidence=confidence,
    )

    print("\n=== 习题 2.4 (2) 最可能达标场景模型 ===")
    print(f"目标门槛: {target:.1f}")
    print(f"ObjVal: {chance_result['obj']:.10f}")
    print(f"ObjBound: {chance_result['obj_bound']:.10f}")
    print(f"MIPGap: {chance_result['mip_gap']:.3e}")
    print(f"最优场景对数联合概率: {chance_result['scenario_log_prob']:.10f}")
    print(f"最优场景联合概率: {chance_result['scenario_joint_prob']:.12e}")
    print(f"最优场景总分: {chance_result['scenario_score']:.4f}")
    print_lineup_detail(
        title="该最可能场景对应阵容：",
        lineup=chance_result["lineup"],
        score_table=build_score_table("mean"),
    )
    print("该最可能场景（仅展示参赛的 24 个项目成绩）：")
    for event in EVENTS:
        event_items = [item for item in chance_result["scenario_items"] if item["event"] == event]
        event_items.sort(key=lambda item: item["athlete"])
        text = ", ".join(
            f"运{item['athlete']}={item['score']:.1f}(p={item['prob']:.2f})"
            for item in event_items
        )
        print(f"  {event}: {text}")

    print("对该阵容做精确离散评估（非正态近似）：")
    print(f"  夺冠概率 P(S >= {target:.1f}) = {eval_result['win_prob']:.6%}")
    print(f"  得分前景 E[S] = {eval_result['expected']:.6f}")
    print(
        f"  {confidence:.0%} 把握可战胜的对手水平约为: "
        f"{eval_result['confidence_level']:.1f}"
    )
    print(f"  分布状态数（卷积后）: {eval_result['state_count']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解习题 2.4（体操团体赛阵容优化）。")
    parser.add_argument(
        "--target",
        type=float,
        default=236.2,
        help="夺冠门槛分数（默认 236.2）。",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.9,
        help="战胜对手把握度（默认 0.9）。",
    )
    parser.add_argument(
        "--chance-mipgap",
        type=float,
        default=1e-6,
        help="第(2)问最可能达标场景模型的 MIPGap（默认 1e-6）。",
    )
    parser.add_argument(
        "--no-chance",
        action="store_true",
        help="仅求解第(1)问，不运行第(2)问概率模型。",
    )
    args = parser.parse_args()

    if not (0 < args.confidence < 1):
        raise ValueError("confidence 必须在 (0,1) 区间。")
    if args.chance_mipgap < 0:
        raise ValueError("chance-mipgap 不能为负。")

    solve(
        target=args.target,
        confidence=args.confidence,
        chance_mipgap=args.chance_mipgap,
        run_chance=not args.no_chance,
    )
