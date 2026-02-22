"""
例题 2.8：使用蒙特卡洛法估计曲边三角形面积。

运行示例：
  python ch02/ex02_08/solution.py
  python ch02/ex02_08/solution.py --samples 500000 --seed 2026
"""

import argparse
import random


def is_inside_region(x_value, y_value):
    return y_value <= x_value * x_value and y_value <= 12.0 - x_value


def solve(samples=200000, seed=42):
    if samples <= 0:
        raise ValueError("samples 必须为正整数。")

    random.seed(seed)

    x_low, x_high = 0.0, 12.0
    y_low, y_high = 0.0, 12.0
    box_area = (x_high - x_low) * (y_high - y_low)

    hit_count = 0
    for _ in range(samples):
        x_value = random.uniform(x_low, x_high)
        y_value = random.uniform(y_low, y_high)
        if is_inside_region(x_value, y_value):
            hit_count += 1

    area_estimate = box_area * hit_count / samples

    # 解析面积：∫(0->3)x^2 dx + ∫(3->12)(12-x) dx = 49.5
    exact_area = 49.5
    abs_error = abs(area_estimate - exact_area)
    rel_error = abs_error / exact_area

    print("=== 例题 2.8 蒙特卡洛求解结果 ===")
    print(f"样本数 N: {samples}")
    print(f"随机种子: {seed}")
    print(f"外接矩形面积: {box_area:.6g}")
    print(f"命中点数 m: {hit_count}")
    print(f"面积估计值 S_hat: {area_estimate:.10f}")
    print(f"解析面积 S: {exact_area:.10f}")
    print(f"绝对误差: {abs_error:.10f}")
    print(f"相对误差: {rel_error:.6%}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="求解例题 2.8（蒙特卡洛法估计面积）。")
    parser.add_argument(
        "--samples",
        type=int,
        default=200000,
        help="蒙特卡洛随机样本数（默认 200000）。",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子（默认 42）。",
    )
    args = parser.parse_args()

    solve(samples=args.samples, seed=args.seed)

