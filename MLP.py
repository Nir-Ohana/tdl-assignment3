import math
import sys

import torch
from torch import nn


class MLP(nn.Module):

    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1 / in_features
        bound = math.sqrt(k)
        # toch.rand returns a tensor samples uniformly in [0, 1).
        # we scaling it to [l = -bound, r = bound] using the formula: (l - r) * torch.rand(x, y) + r
        self.W = (-2 * bound) * torch.rand(out_features, in_features) + bound

    def forward(self, x):
        _, input_num = x.shape
        if input_num != self.in_features:
            sys.exit(f'Wrong x size: {x}')
        y = x @ self.W.t()
        return y


if __name__ == '__main__':
    m = MLP(20, 30)
    input = torch.randn(128, 20)
    output = m(input)
    print(output.size())
