# 习题 6.9 模型（小船过河航线）

## 题意重述
河宽为 \(d\)。小船从近岸点 \(A\) 出发，目标是对岸正对点 \(B\)。  
水流速度为 \(v_1\)，小船在静水中速度为 \(v_2\)，记
\[
k=\frac{v_1}{v_2}.
\]
要求：
1. 建立航线方程并求解析解；
2. 对 \(d=100\text{ m}, v_1=1\text{ m/s}, v_2=2\text{ m/s}\) 做数值求解并与解析解比较。

## 坐标与运动方程
取 \(x\) 轴沿顺流方向，\(y\) 轴垂直河岸从近岸指向对岸。  
于是
- \(A=(0,0)\)；
- \(B=(0,d)\)。

小船始终将船头指向 \(B\)，因此其相对水体速度方向为 \(\overrightarrow{PB}\)。设
\[
r=\sqrt{x^2+(d-y)^2}.
\]
则地面参考系下速度分量为
\[
\begin{cases}
\dot x = v_1 - v_2\dfrac{x}{r},\\[6pt]
\dot y = v_2\dfrac{d-y}{r}.
\end{cases}
\]

## 航线解析解
令
\[
u=d-y,\qquad p=\frac{x}{u}.
\]
由方程可推得
\[
u\frac{dp}{du}=-k\sqrt{1+p^2}.
\]
积分并利用初值 \(x(0)=0,y(0)=0\Rightarrow p=0,u=d\)，得到
\[
\operatorname{arsinh}(p)=k\ln\frac{d}{u}.
\]
即
\[
\frac{x}{u}=\sinh\!\left(k\ln\frac{d}{u}\right),
\]
从而航线解析表达（以 \(y\) 为自变量）为
\[
x=(d-y)\sinh\!\left(k\ln\frac{d}{d-y}\right),\quad 0\le y<d.
\]

## 渡河时间解析式
由
\[
\dot u=-\frac{v_2}{\sqrt{1+p^2}}
\]
并代入 \(p=\sinh\!\left(k\ln\frac{d}{u}\right)\)，积分得
\[
t(y)=\frac{d}{2v_2}
\left[
\frac{1-\left(\frac{d-y}{d}\right)^{1-k}}{1-k}
+
\frac{1-\left(\frac{d-y}{d}\right)^{1+k}}{1+k}
\right].
\]
令 \(y=d\) 得总渡河时间
\[
T=\frac{d}{v_2(1-k^2)}.
\]
因此该策略有限时间到达 \(B\) 的条件为 \(k<1\)（即 \(v_2>v_1\)）。

## 数值求解与比较
`solution.py` 中：
1. 用 `solve_ivp` 对上述二维常微分方程组积分，事件 \(y=d\) 触发停止；
2. 输出数值渡河时间 \(T_{\text{num}}\) 与解析值 \(T_{\text{ana}}\) 比较；
3. 在同一 \(y\) 网格上比较数值航线与解析航线，给出最大误差与 RMSE；
4. 输出任意时刻位置（数值与解析）对照，并绘制航迹图。

## 本题指定参数（第 2 问）
- \(d=100\text{ m}\)
- \(v_1=1\text{ m/s}\)
- \(v_2=2\text{ m/s}\)
- \(k=0.5\)

解析渡河时间为
\[
T=\frac{100}{2(1-0.5^2)}=\frac{100}{1.5}\approx 66.67\text{ s}.
\]
