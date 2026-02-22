"""
例题 3.5：求 f(x,y)=x^3-y^3+3x^2+3y^2-9x 在 (0,0) 附近的极值。

运行示例：
  python ch03/ex03_05/solution.py
  python ch03/ex03_05/solution.py --radius 1.5
"""

import argparse
import math


def f_value(x, y):
    """目标函数值。"""
    return x**3 - y**3 + 3 * x**2 + 3 * y**2 - 9 * x


def classify_point(x, y):
    """
    用 Hessian 二阶判别分类驻点。
    本题 fxy=0，Hessian 为对角阵，判别可直接看 fxx 与 fyy 符号。
    """
    fxx = 6 * x + 6
    fyy = -6 * y + 6

    if fxx > 0 and fyy > 0:
        return "局部极小点"
    if fxx < 0 and fyy < 0:
        return "局部极大点"
    return "鞍点"


def stationary_points():
    """由 fx=0, fy=0 的分解式直接得到全部驻点。"""
    xs = [1.0, -3.0]
    ys = [0.0, 2.0]
    return [(x, y) for x in xs for y in ys]


def solve(radius):
    points = stationary_points()

    print("=== 例题 3.5 驻点与极值分析 ===")
    print("函数: f(x,y)=x^3-y^3+3x^2+3y^2-9x")
    print("驻点明细：")
    for x, y in points:
        val = f_value(x, y)
        dist = math.hypot(x, y)
        ctype = classify_point(x, y)
        print(
            f"  (x,y)=({x:.6f},{y:.6f}), f={val:.6f}, "
            f"到原点距离={dist:.6f}, 类型={ctype}"
        )

    near_points = [(x, y) for x, y in points if math.hypot(x, y) <= radius]
    print(f"\n按“附近半径 r={radius:.6f}”筛选：")
    if not near_points:
        print("  该邻域内没有驻点。")
        return

    for x, y in near_points:
        print(
            f"  邻域内驻点: (x,y)=({x:.6f},{y:.6f}), "
            f"f={f_value(x, y):.6f}, 类型={classify_point(x, y)}"
        )

    # 在邻域内驻点上，给出最小/最大函数值（仅用于本题“附近”说明）
    best_min = min(near_points, key=lambda p: f_value(p[0], p[1]))
    best_max = max(near_points, key=lambda p: f_value(p[0], p[1]))
    print("\n邻域内驻点比较：")
    print(
        f"  最小值驻点: ({best_min[0]:.6f},{best_min[1]:.6f}), "
        f"f={f_value(best_min[0], best_min[1]):.6f}"
    )
    print(
        f"  最大值驻点: ({best_max[0]:.6f},{best_max[1]:.6f}), "
        f"f={f_value(best_max[0], best_max[1]):.6f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="例题3.5：多元函数在(0,0)附近的极值分析"
    )
    parser.add_argument(
        "--radius",
        type=float,
        default=1.5,
        help="“附近”邻域半径（默认 1.5）",
    )
    args = parser.parse_args()
    solve(radius=args.radius)
