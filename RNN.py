import math
import sys

import torch
from torch import nn


class RNN(nn.Module):

    def __init__(self, in_features, out_features, non_linearity="tanh"):
        super(RNN, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.non_linearity = non_linearity
        k = 1 / out_features
        bound = math.sqrt(k)
        # toch.rand returns a tensor samples uniformly in [0, 1).
        # we scaling it to [l = -bound, r = bound] using the formula: (l - r) * torch.rand(x, y) + r
        self.W = (-2 * bound) * torch.rand(out_features, in_features) + bound
        self.U = (-2 * bound) * torch.rand(out_features, out_features) + bound

    def forward(self, x, b_prev=None):

        # check validity of x
        _, input_num = x.shape
        if input_num != self.in_features:
            sys.exit(f'Wrong x size: {x}')

        # check validity of b_prev
        if b_prev is not None:
            _, output_num = b_prev.shape
            if output_num != self.out_features:
                sys.exit(f'Wrong b size: {b_prev}')
        else:
            b_prev = torch.zeros(x.shape[0], self.out_features)
        a = (x @ self.W.t()) + (b_prev @ self.U.t())

        if self.non_linearity == "tanh":
            b = torch.tanh(a)
        else:
            b = torch.relu(a)

        return b


if __name__ == '__main__':
    rnn = RNN(10, 20)
    input = torch.randn(6, 3, 10)
    hx = torch.randn(3, 20)
    output = []
    for i in range(6):
        hx = rnn(input[i], hx)
        print(hx)
        output.append(hx)
