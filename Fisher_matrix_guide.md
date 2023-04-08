# Fisher Matrix Quick Guide

参考[Report of the DARK ENERGY TASK FORCE](https://arxiv.org/abs/astro-ph/0609591), Page 94. 

故事的开始，我们手里有一组数据$\{x_i,y_i\}$，对应的$y_i$的误差为$\sigma_i$，对应这个数据有一个模型$f_i = f(x_i|\theta)$。根据高斯分布的原理，有
$$P(y|\theta)\propto \exp(-\frac{1}{2}\chi^2)=\exp(-\frac{1}{2}\sum_i\frac{(f_i-y_i)^2}{\sigma_i^2})$$

根据贝叶斯统计的原理，如果我们使用uniform prior的话，$P(y|\theta) \propto P(\theta|y)$。现在假定拟合得到的参数为true value，在参数空间中，在true value附近作展开，研究在正确参数附近的概率分布，以确定拟合参数的不确定度。

设$\theta_i = \bar{\theta}_i+\delta\theta_i$，可以得到
$$ \chi^2(\theta) = \chi^2(\bar{\theta})+\sum_j \partial_{\theta_j}\chi^2\delta \theta_j+\frac{1}{2}\sum_{j,k}(\partial^2_{\theta_j\theta_k}\chi^2)\delta \theta_j\delta \theta_k + \cdots$$

由于是在true value附近展开，一阶项认为是0，于是得到
$$ P(y|\theta)\propto \exp(-\frac{1}{2}\chi^2) \propto \exp(-\frac{1}{4}\sum_{j,k}(\partial^2_{\theta_j\theta_k}\chi^2)\delta \theta_j\delta \theta_k) = \exp(-\frac{1}{2}F_{jk}\delta \theta_j\delta \theta_k)$$

$F_{jk}$即为我们的Fisher矩阵。经过简单的计算之后，
$$ F_{jk} = \sum_i\frac{1}{\sigma_i^2}\frac{\partial f_i}{\partial \theta_j}\frac{\partial f_i}{\partial \theta_k} $$

Fisher matrix的方便之处在于，根据多元正态分布的一般形式，
$$\rho(x)\propto \exp(-\frac{1}{2}x^T\Sigma^{-1}x)=\exp(-\frac{1}{2}\Sigma^{-1}_{jk}x_jx_k)$$
$\Sigma$是协方差矩阵。于是我们就可以看出，对Fisher矩阵求逆即可得到协方差矩阵。Fisher Matrix Forecast在计算观测对宇宙学参数的约束的时候非常常用。