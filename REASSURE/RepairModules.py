import torch, torch.nn as nn, numpy as np
from scipy.optimize import linprog


class SupportNet(nn.Module):
    def __init__(self, A: np.ndarray, b: np.ndarray, n):
        """
        SupportNN is a neural network that almost only active on polytope {x|Ax<=b}.
        :param A, b: H-rep of a polytope.
        :param n: To control the side effect. Choose a large number until the 100% repair rate.
        """
        super(SupportNet, self).__init__()
        assert len(A) == len(b)
        self.A = torch.tensor(A, dtype=torch.float32)
        self.b = torch.tensor(b, dtype=torch.float32)
        assert len(self.A.size()) == 2
        self.layer = nn.Linear(*self.A.size())
        self.layer.weight = torch.nn.Parameter(-self.A)
        self.layer.bias = torch.nn.Parameter(self.b)
        self.n = n

    def forward(self, x):
        y = self.layer(x)
        s = torch.sum(torch.relu(self.n * y + 1), -1, keepdim=True) - torch.sum(torch.relu(self.n * y), -1, keepdim=True)
        return torch.relu(s - self.A.shape[0] + 1)


class SingleRegionRepairNet(nn.Module):
    def __init__(self, g, c, d, input_boundary):
        super(SingleRegionRepairNet, self).__init__()
        c, d = torch.tensor(c, dtype=torch.float32), torch.tensor(d, dtype=torch.float32)
        self.p = nn.Linear(*c.shape)
        self.p.weight = nn.Parameter(c)
        self.p.bias = nn.Parameter(d)
        self.g = g
        self.K = self._compute_K(c, d, input_boundary)

    def _compute_K(self, c, d, input_boundary):
        K = 0
        for i in range(d.shape[0]):
            K = max(K, abs(linprog(c=c[i], A_ub=input_boundary[0], b_ub=input_boundary[1], bounds=[None, None]).fun + d[i]))
            K = max(K, abs(linprog(c=-c[i], A_ub=input_boundary[0], b_ub=input_boundary[1], bounds=[None, None]).fun - d[i]))
        return K

    def forward(self, x):
        return torch.relu(self.p(x) + self.K * self.g(x) - self.K) - torch.relu(-self.p(x) + self.K * self.g(x) - self.K)



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