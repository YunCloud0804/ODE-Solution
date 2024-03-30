import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from scipy.special import gamma
from scipy.integrate import quad,dblquad,nquad
"""
用神经网络模拟微分方程,f(x)'=f(x),初始条件f(0) = 1
"""

class Net(nn.Module):
    def __init__(self, NL, NN): # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数，
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()
        self.input_layer = nn.Linear(1, NN)
        self.hidden_layer = nn.Linear(NN,int(NN/2)) ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优。更多情况有待我实验验证。
        self.output_layer = nn.Linear(int(NN/2), 1)

    def forward(self, x):
        out = torch.tanh(self.input_layer(x))
        out = torch.tanh(self.hidden_layer(out))
        out_final = self.output_layer(out)
        return out_final

#mittag-leffler函数

def MLF2(z, alpha, beta):
    """Mittag-Leffler function E(alpha, 1)(z)
    """
    z = z.detach().numpy()
    if alpha == 0:
        return 1/(1 - z)
    elif alpha == 1:
        return np.exp(z)
    elif alpha > 1 or all(z > 0):
        k = np.arange(100)
        return np.polynomial.polynomial.polyval(z, 1/gamma(alpha*k + beta))



#积分的使用

#print (quad(lambda x:np.exp(-x), 0, np.inf))

#神经网络

net=Net(10,30) # 10层 30个
mse_cost_function = torch.nn.MSELoss(reduction='mean') # Mean squared error 均方误差求cost
optimizer = torch.optim.Adam(net.parameters(),lr=1e-4)  # 优化器


#导数和方程的定义
def ode_01(x,net):
    y=net(x)
    y_x = autograd.grad(y, x,grad_outputs=torch.ones_like(net(x)),create_graph=True)[0]
    #print(x)
#Mittag-Leffler部分
    e=autograd.Variable(torch.from_numpy(MLF2(x,1,1)).float(), requires_grad=True)
#积分部分
    x2 = x.detach().numpy()
    nn = np.size(x2)

    int = np.zeros(n)
    temp2 = [int]
    intt = np.transpose(temp2)

 #   pt_x_in2 = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)
  #  for i in range(0, nn):
 #       intt[i], err = quad(lambda t: y_x[i]*pt_x_in2[i], 0, x2[i])  # 0-inf exp(-x)积分 lambda x：预留关键字x
  #  intt = autograd.Variable(torch.from_numpy(intt).float(), requires_grad=True)

    return y_x+x**2*y-y*x+1/(x+1)                                                                  # 定义方程


# requires_grad=True).unsqueeze(-1)



#主程序
#plt.ion()  # 动态图


iterations=1000000
i=0
n=2000
for epoch in range(iterations):

    optimizer.zero_grad()  # 梯度归0

    ## 初始条件的损失函数
    x_0 = torch.zeros(n, 1)
    y_0 = net(x_0)
    mse_i = mse_cost_function(y_0, 2*torch.ones(n, 1))  # f(0) - 1 = 0    #初始条件（ones的 列向量）y初值

    ## 方程的损失函数
    x = np.arange(0,2,2/n)
    temp = [x]
    x_in = np.transpose(temp)                                          #x取值
  #  x_in = np.random.uniform(low=0.0, high=2.0, size=(2, 1))
    pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)  # x 自动求导（tensor）
    pt_y_colection=ode_01(pt_x_in,net)
#    print(pt_x_in)
#    print(pt_y_colection)
    pt_all_zeros= autograd.Variable(torch.from_numpy(np.zeros((n,1))).float(), requires_grad=False)
    mse_f=mse_cost_function(pt_y_colection, pt_all_zeros)  # y-y' = 0

    loss = mse_i + mse_f   #loss函数
    loss.backward()  # 反向传播
    optimizer.step()  # 优化下一步。This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    if epoch%5000==0 :
        i=i+1
     #  y = torch.exp(pt_x_in)  # y 真实值 真实值的
        y_train0 = net(pt_x_in) # y 预测值
        print(f'Epoch: {epoch}, line number: {i-1}, "Traning Loss:", loss: {loss.item()}')
     #   print(pt_y_colection)
     #   print(f'times {epoch}  -  loss: {loss.item()} - y_0: {y_0}')
          #  plt.figure(i)
         #   plt.plot(pt_x_in.detach().numpy(), y.detach().numpy(),linewidth=1.0)                  #真实值的画图
        plt.plot(pt_x_in.detach().numpy(), y_train0.detach().numpy(),linewidth=1.0)   #模拟结果
        plt.pause(0.1)
        plt.grid(True)
            #plt.ion()
        if loss <= 1e-4 :
            break
plt.pause(0)
