# 卡尔曼滤波作业

<p align='right'>——518021910971 裴奕博</p>

- In this lecture, we introduced the Gain Matrix K of the Kalman Filter. However, we do not provide a detailed explanation of K. Would you please provide a brief derivation of K?

#### 定义

若系统的状态的预测向量$\mathbf{x_k}$,观测值向量$\mathbf{z_k}$。假设两者均满足高斯分布，两者的均值和方差分别为$(\mu_0,\Sigma_0)$和$(\mu_1,\Sigma_1)$。在Kalman滤波中，我们将预测值的分布与观测值的分布相乘，两者相乘之后的分布仍然是一个高斯分布，而他们的均值和方差由$(\mu_0,\Sigma_0)$和$(\mu_1,\Sigma_1)$加权求得，此处的权重即为增益矩阵$K$。

#### $K$表达式的推导：

- 公式1：若$\operatorname{Cov}(x) =\Sigma$，则$\operatorname{Cov}(\mathbf{A} x) =\mathbf{A} \Sigma \mathbf{A}^{T}$
- 公式2：若两个高斯分布的均值和方差分别为$(\mu_0,\Sigma_0)$和$(\mu_1,\Sigma_1)$，则两者乘积的分布仍然满足高斯分布，且满足：
$$
\begin{array}{l}

  \vec{\mu}^{\prime}=\overrightarrow{\mu_{0}}+\mathbf{K}\left(\overrightarrow{\mu_{1}}-\overrightarrow{\mu_{0}}\right) \\
  \Sigma^{\prime}=\Sigma_{0}-\mathbf{K} \Sigma_{0}\\
  其中\mathbf{K}=\Sigma_{0}\left(\Sigma_{0}+\Sigma_{1}\right)^{-1} \\
  \end{array}
$$


- 若系统的状态的预测向量$\mathbf{x_k}$,$\mathbf{x_k}$的协方差矩阵为$\mathbf{P_k}$，观测值向量为$\mathbf{z_k}$。两者均满足高斯分布，由公式1可知满足
  
  $$
  \begin{array}{l}
  \vec{\mu}_{\text {1}}=\mathbf{H}_{k} \hat{\mathbf{x}}_{k} \\
  \mathbf{\Sigma}_{\text {1}}=\mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T}
  \end{array}
  $$
  其中，$\mathbf{H_k}$为传感器矩阵。
  
- 此外，又假设传感器的读数满足
  $$
  \begin{array}{l}
  \vec{\mu}_{\text {0}}=\mathbf{z}_{k}\\
  \mathbf{\Sigma}_{\text {0}}=\mathbf{R}_{k}
  \end{array}
  $$
  
  

- 因此我们得到了两组满足高斯分布的预测值分布$(\mu_0,\Sigma_0)$和$(\mu_1,\Sigma_1)$，为了在这两者中找到最优解，我们将它们相乘，代入公式2就有：
  $$
  \begin{aligned}
  \mathbf{H}_{k} \hat{\mathbf{x}}_{k}^{\prime} &=\mathbf{H}_{k} \hat{\mathbf{x}}_{k} +\mathbf{A}\left(\mathbf{z_{k}}-\mathbf{H}_{k} \hat{\mathbf{x}}_{k}\right) \\
  \mathbf{H}_{k} \mathbf{P}_{k}^{\prime} \mathbf{H}_{k}^{T} &=\mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T} -\mathbf{A} \mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T}
  \end{aligned}
  $$
  其中$\mathbf{A}=\mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T}(\mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T}+\mathbf{R_k})^{-1}$。将A代入并约去公因子$H_k$便可得到结果
  $$
  \begin{aligned}
  \hat{\mathbf{x}}_{k}^{\prime} &= \hat{\mathbf{x}}_{k} +\mathbf{K}\left(\mathbf{z_{k}}-\mathbf{H}_{k} \hat{\mathbf{x}}_{k}\right) \\
  \mathbf{P}_{k}^{\prime}&=(\mathbf{I}-\mathbf{K} \mathbf{H}_{k}) \mathbf{P}_{k} \\
  其中\mathbf{K}&=\mathbf{P}_{k} \mathbf{H}_{k}^{T}(\mathbf{H}_{k} \mathbf{P}_{k} \mathbf{H}_{k}^{T}+\mathbf{R_k})^{-1}
  \end{aligned}
  $$
  

  K即为增益矩阵。