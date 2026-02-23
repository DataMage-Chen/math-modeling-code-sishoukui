# 例题 7.13 模型（中位数估计的 Bootstrap 标准误）

## 题意重述
某基金年回报率样本（连续型总体、分布函数 \(F\) 未知）为：
\[
18.2,\ 9.5,\ 12.0,\ 21.1,\ 10.2.
\]
以样本中位数作为总体中位数 \(\theta\) 的估计，求该中位数估计的标准误差的 Bootstrap 估计。

## 点估计
样本中位数估计量：
\[
\hat\theta=\mathrm{median}(X_1,\dots,X_n).
\]
本题 \(n=5\)，排序后样本中位数为 \(12.0\)。

## Bootstrap 思路
由于总体分布未知且中位数估计量的解析标准误不便直接求，采用非参数 Bootstrap：

1. 从原样本 \(\{x_1,\dots,x_n\}\) 有放回抽样，得到一个重抽样样本；
2. 计算该重抽样样本中位数 \(\hat\theta^*\)；
3. 重复上述步骤 \(B\) 次，得到 \(\hat\theta_1^*,\dots,\hat\theta_B^*\)；
4. 用这些 Bootstrap 中位数的样本标准差估计标准误：
\[
\widehat{SE}_{boot}(\hat\theta)
=
\sqrt{\frac{1}{B-1}\sum_{b=1}^{B}\left(\hat\theta_b^*-\bar{\hat\theta}^*\right)^2}.
\]

## 说明
- \(B\) 取值越大，估计通常越稳定（常用 \(10^4\sim10^5\)）；
- 本题样本量较小，Bootstrap 分布会呈离散特征，这属于正常现象。

## 实现说明
`solution.py` 默认：
- 重抽样次数 `B=100000`；
- 随机种子 `seed=2026`；
- 输出样本中位数和 Bootstrap 标准误估计；
- 可选 `--plot` 绘制 Bootstrap 中位数分布图。
