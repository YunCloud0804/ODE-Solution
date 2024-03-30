# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from scipy.integrate import quad
'''
def MLf(z, a):
    """Mittag-Leffler function
    """
    k = np.arange(100).reshape(-1, 1)
    E = z**k / gamma(a*k + 1)
    return np.sum(E, axis=0)

x = np.arange(-50, 10, 0.1)

plt.figure(figsize=(10,5))
for i in range(5):
    print(MLf(x, i))
    plt.plot(x, MLf(x, i), label="alpha = "+str(i))
plt.legend()
plt.ylim(-5, 5); plt.xlim(-55, 15); plt.grid()
plt.show()

从x = -35开始，在a = 1的橙色线中可以看到最好的不稳定性，但是a = 0(蓝线)也存在问题.更改要加和的项数(即j)会更改发生不稳定性的x.

这是怎么回事?如何避免这种情况?

推荐答案
如果a = 0，则您使用的MLf的系列定义仅在| z |< 1时适用.实际上，当基数z的绝对值大于1时，幂z**k不断增加，并且级数发散.观察其第100个或另一个部分和是没有意义的，这些和与区间-1< 1之外的函数无关. & 1.对于a = 0，只需使用公式1/(1-z).

该函数为exp(z)，从技术上讲，它由所有z的幂级数z**k / k!表示.但是，对于大的负z而言，该幂级数经历了灾难性的重要性丧失:各个术语都非常庞大，例如，(-40)**40/factorial(40)大于1e16，但是它们的和很小(exp(-40)几乎为零).由于1e16接近双精度极限，因此输出被截断/舍入操作的噪声所控制.

通常，从效率和精度的角度来看，通过添加c(k) * z**k来评估多项式并不是最好的选择. Horner的方案已经在NumPy中实现，使用它可以简化代码:

k = np.arange(100)
return np.polynomial.polynomial.polyval(z, 1/gamma(a*k + 1))
但是，这不会将序列保存为exp(z)，其数值问题超出了NumPy的范围. 

您可以使用mpmath进行评估，以提高准确性(mpmath支持任意高精度的浮点运算)并失去速度(不编译代码，不进行矢量化). 

或者当a = 1时，您可以从MLf中返回exp(z).

该系列收敛，但又损失了灾难性的精度；现在没有明确的公式可以使用.前面提到的mpmath是一个选项:设置非常高的精度(mp.dps = 50)，并希望它足以对系列求和.另一种选择是寻找另一种计算函数的方法. 

环顾四周，我发现了"Mittag-Leffler函数及其导数" Rudolf Gorenflo，Joulia Loutchko&尤里·卢奇科(Yuri Luchko).我从中得出公式(23)，并将其用于负z和0 << 1.
'''

def MLF2(z, alpha, beta):
    """Mittag-Leffler function E(alpha, 1)(z)
    """
    z = np.atleast_1d(z)
    if alpha == 0:
        return 1/(1 - z)
    elif alpha == 1:
        return np.exp(z)
    elif alpha > 1 or all(z > 0):
        k = np.arange(100)
        return np.polynomial.polynomial.polyval(z, 1/gamma(alpha*k + beta))
'''
    # a helper for tricky case, from Gorenflo, Loutchko & Luchko
    def _MLf(z, a):
        if z < 0:
            f = lambda x: (np.exp(-x*(-z)**(1/a)) * x**(a-1)*np.sin(np.pi*a)
                          / (x**(2*a) + 2*x**a*np.cos(np.pi*a) + 1))
            return 1/np.pi * quad(f, 0, np.inf)[0]
        elif z == 0:
            return 1
        else:
            return MLf(z, a)
    return np.vectorize(_MLf)(z, a)
'''

x = np.arange(-10, 50, 0.1)

plt.figure(figsize=(10,5))
for i in range(1,10):
    print(MLF2(x,i,i))
    plt.plot(x, MLF2(x, i, i), label="alpha = beta "+str(i))
plt.legend()
plt.ylim(-5, 100); plt.xlim(-15, 60); plt.grid()
plt.show()