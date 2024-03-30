import torch

# Domain and Sampling 定义和方程表达式
def interior(n=1000):
    x = torch.rand(n, 1)  # x定义域
  #  y = torch.rand(n, 1)
    cond = 2**x   #方程表达式
    return x.requires_grad_(True), cond # y.requires_grad_(True), cond

# Loss 损失函数
loss = torch.nn.MSELoss()

# 定义函数导数，任意整数阶
def gradients(u, x, order=1):
    if order == 1:
        return torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                                   create_graph=True,
                                   only_inputs=True, )[0]
    else:
        return gradients(gradients(u, x), x, order=order - 1)

# 定义等式左边
def l_interior(u):
    x,  cond = interior()
    ux = u(x, dim=1))
    return loss(gradients(ux, x, 1), cond)