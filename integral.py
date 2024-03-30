import numpy as np
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch import autograd
from scipy.special import gamma
from scipy.integrate import quad,dblquad,nquad
import math
def main():
    x = np.arange(0, 2, 0.5)
    n=np.size(x)


    temp = [x]
    x_in = np.transpose(temp)
    int = np.zeros(n)
    temp2 = [int]
    intt = np.transpose(temp2)

    pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)

    print(x_in)
    print(pt_x_in)

    for i in range (0,n):
        intt[i],err = quad(lambda t: pt_x_in[i], 0, x_in[i])       #0-inf exp(-x)积分 lambda x：预留关键字x

    int = autograd.Variable(torch.from_numpy(intt).float(), requires_grad=True)
    print(int)

if __name__=="__main__":
    main()