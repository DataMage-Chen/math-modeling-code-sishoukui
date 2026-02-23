# 习题 5.6 模型（非线性最小二乘参数识别）

## 题意重述
给定函数

- `g(x) = 10a / (10b + (a-10b) * exp(-a*sin(x)))`

取 `a=1.1, b=0.01`，在 `x=1,2,...,20` 处生成观测值 `y_i=g(x_i)`，并完成：

1. 用“lsqcurvefit 思路”拟合 `hat_g(x)`；
2. 用“fittype + fit 思路”拟合 `hat_g(x)`。

## 已知数据与符号说明
- 自变量：`x_i = i, i=1,...,20`
- 观测值：`y_i = g(x_i; a_true, b_true)`，其中 `a_true=1.1, b_true=0.01`
- 拟合模型：`hat_g(x; a,b)=10a/(10b+(a-10b)exp(-a*sin(x)))`
- 参数：`a,b` 为待估计量。

## 数学模型
参数估计统一写成非线性最小二乘问题：

- `min_{a,b} Σ_i [hat_g(x_i;a,b)-y_i]^2`

其中残差定义为：

- `r_i(a,b)=hat_g(x_i;a,b)-y_i`

## 方法选择与理由
- 方法一（对应 MATLAB `lsqcurvefit`）：使用 `scipy.optimize.least_squares` 直接最小化残差向量；
- 方法二（对应 MATLAB `fittype + fit`）：使用 `scipy.optimize.curve_fit` 进行非线性回归；
- 两种方法求得参数后，比较参数误差与拟合误差（SSE、RMSE、`R^2`）。

## 结果与校验
- 输出两种方法得到的 `(a,b)`；
- 输出与真值 `(1.1,0.01)` 的偏差；
- 输出拟合指标（SSE、RMSE、`R^2`）；
- 在同一图中绘制观测点与两种方法拟合曲线，检查吻合程度。

## 说明
- 计算 `sin(x)` 时按弧度制处理（与 `numpy.sin` 一致）。

