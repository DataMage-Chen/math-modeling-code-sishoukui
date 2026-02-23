# 例题 7.14 模型（中位数估计量 MSE 的 Bootstrap 估计）

## 题意重述
设铅的升华热是分布函数 \(F\) 的连续型随机变量，总体中位数 \(\theta\) 未知。给出一组样本数据，令样本中位数
\[
M=M(X)
\]
作为 \(\theta\) 的估计，要求估计
\[
MSE=E[(M-\theta)^2].
\]

## 难点
总体分布 \(F\) 未知，且中位数估计量的抽样分布不易写出解析式，因此直接计算 \(MSE\) 困难。

## Bootstrap 估计思路
采用非参数 Bootstrap，用经验分布 \(\hat F_n\) 近似总体分布 \(F\)：

1. 原样本中位数记为 \(M_{\text{hat}}\)（作为 \(\theta\) 的替代）；
2. 从原样本中有放回抽样，生成 \(B\) 个重抽样样本；
3. 第 \(b\) 个重抽样样本中位数记为 \(M_b^*\)；
4. 用
\[
\widehat{MSE}_{boot}
=\frac1B\sum_{b=1}^{B}\left(M_b^*-M_{\text{hat}}\right)^2
\]
作为 \(MSE\) 的 Bootstrap 估计。

同时可分解为
\[
\widehat{MSE}_{boot}
\approx \widehat{Var}_{boot}(M^*)+\widehat{Bias}_{boot}(M^*)^2,
\]
其中
\[
\widehat{Bias}_{boot}(M^*)=\bar M^*-M_{\text{hat}}.
\]

## 实现说明
`solution.py` 默认：
- 重抽样次数 \(B=100000\)；
- 固定随机种子以便复现；
- 输出 \(\widehat{MSE}_{boot}\)、方差项、偏差项及分解校验；
- 可选绘制 Bootstrap 中位数分布图。

## 说明
Bootstrap 估计不增加原始信息量，而是用计算近似估计量的抽样不确定性。样本较小时结果应结合背景知识与敏感性分析谨慎解释。
