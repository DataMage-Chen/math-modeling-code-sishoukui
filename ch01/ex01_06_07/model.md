# 例题 1.6/1.7 模型

## 例题 1.6（一般形式）
最小化：

- |x1| + |x2| + ... + |xn|

约束条件：

- A x <= b

其中 x = [x1, x2, ..., xn]^T，A 与 b 为给定矩阵与向量。

### 线性化思路
引入辅助变量 u_i >= 0，并令：

- x_i <= u_i
- -x_i <= u_i

则 |x_i| 可用 u_i 表示，目标变为最小化 sum u_i。

## Gurobi 直接支持绝对值的两种写法
Gurobi 提供通用约束，可直接表示绝对值（内部仍会线性化）：

写法一：通用约束

```python
y = model.addVar(lb=0.0)
model.addGenConstrAbs(y, x, name="abs_x")  # y = |x|
```

写法二：在表达式中使用 abs_

```python
import gurobipy as gp
model.setObjective(gp.abs_(x) + gp.abs_(y), GRB.MINIMIZE)
```

## 例题 1.7（具体数值）
最小化：

- |x1| + 2|x2| + 3|x3| + 4|x4|

约束条件：

- x1 - x2 - x3 + x4 <= -2
- x1 - x2 + x3 - 3x4 <= -1
- x1 - x2 - 2x3 + 3x4 <= -1/2

## 决策变量
- x_i：原变量（可为任意实数）
- u_i：辅助变量，u_i >= 0

## 目标函数（线性化后）
例题 1.7 线性化目标：

- u1 + 2u2 + 3u3 + 4u4

## 约束条件（线性化后）
对每个 i：

- x_i <= u_i
- -x_i <= u_i
- u_i >= 0

并满足例题 1.7 的三条线性不等式约束。
