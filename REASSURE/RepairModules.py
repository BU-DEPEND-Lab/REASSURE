import torch, torch.nn as nn, numpy as np
from scipy.optimize import linprog


class SupportNet(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, n):
        """
        SupportNN is a neural network that almost only active on polytope {x|Ax<=b}.
        :param A, b: H-rep of a polytope.
        :param n: To control the side effect. Choose a large number until the 100% repair rate.
        """
        # Ax <= b
        super(SupportNet, self).__init__()
        assert len(A) == len(b)
        A = torch.tensor(A)
        b = torch.tensor(b)
        assert len(A.size()) == 2
        layer = nn.Linear(*A.size())
        layer.weight = torch.nn.Parameter(-A)
        layer.bias = torch.nn.Parameter(b)
        self.layer, self.l, self.n = layer, len(A), n

    def forward(self, x):
        s = - self.l + 1
        y = self.layer(x)
        s += torch.sum(nn.ReLU()(self.n * y + 1), -1, keepdim=True) - torch.sum(nn.ReLU()(self.n * y), -1, keepdim=True)
        return nn.ReLU()(s)


class SingleRegionRepairNet(nn.Module):
    def __init__(self, g, c, d, input_boundary):
        super(SingleRegionRepairNet, self).__init__()
        self.K = 0
        for i in range(len(d)):
            self.K = max(self.K, abs(
                linprog(c=c[i], A_ub=input_boundary[0], b_ub=input_boundary[1], bounds=[None, None]).fun+d[i]))
            self.K = max(self.K, abs(
                linprog(c=-c[i], A_ub=input_boundary[0], b_ub=input_boundary[1], bounds=[None, None]).fun-d[i]))
        c, d = torch.from_numpy(c).float(), torch.from_numpy(d).float()
        self.p = nn.Linear(*c.size())
        self.p.weight = torch.nn.Parameter(c)
        self.p.bias = torch.nn.Parameter(d)
        self.g = g

    def forward(self, x):
        return nn.ReLU()(self.p(x) + self.K*self.g(x)-self.K) \
               - nn.ReLU()(-self.p(x) + self.K*self.g(x)-self.K)


class NetSum(torch.nn.Module):
    def __init__(self, target_net: torch.nn.Module, sub_nets: list):
        super(NetSum, self).__init__()
        self.target_net = target_net
        self.sub_nets = sub_nets

    def forward(self, x):
        out = self.target_net(x)
        for sub_net in self.sub_nets:
            if len(x.size()) >= 3:
                out += sub_net(x.view(len(x), -1))
            else:
                out += sub_net(x)
        return out