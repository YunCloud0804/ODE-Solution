import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma
from torch import autograd

x = np.arange(0,2,0.5)
temp = [x]
x_in = np.transpose(temp)

#print(x_in)
#print(temp2)
pt_x_in = autograd.Variable(torch.from_numpy(x_in).float(), requires_grad=True)

x = pt_x_in.detach().numpy()

print(x)
