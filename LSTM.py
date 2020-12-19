import math
import sys

import torch
from torch import nn


class LSTM(nn.Module):

    def __init__(self, in_features, out_features):
        super(LSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1 / out_features
        bound = math.sqrt(k)
        # toch.rand returns a tensor samples uniformly in [0, 1).
        # we scaling it to [l = -bound, r = bound] using the formula: (l - r) * torch.rand(x, y) + r
        self.Wi = (-2 * bound) * torch.rand(out_features, in_features) + bound
        self.Ui = (-2 * bound) * torch.rand(out_features, out_features) + bound
        self.Wf = (-2 * bound) * torch.rand(out_features, in_features) + bound
        self.Uf = (-2 * bound) * torch.rand(out_features, out_features) + bound
        self.Wg = (-2 * bound) * torch.rand(out_features, in_features) + bound
        self.Ug = (-2 * bound) * torch.rand(out_features, out_features) + bound
        self.Wo = (-2 * bound) * torch.rand(out_features, in_features) + bound
        self.Uo = (-2 * bound) * torch.rand(out_features, out_features) + bound

    def forward(self, x, state=None):
        h_prev = state[0] if state is not None else None
        c_prev = state[1] if state is not None else None

        # check validity of x
        _, input_num = x.shape
        if input_num != self.in_features:
            sys.exit(f'Wrong x size: {x}')

        # check validity of b_prev
        if h_prev is not None:
            _, output_num = h_prev.shape
            if output_num != self.out_features:
                sys.exit(f'Wrong h size: {h_prev}')
        else:
            h_prev = torch.zeros(x.shape[0], self.out_features)
            c_prev = torch.zeros(x.shape[0], self.out_features)

        activation_func = nn.Sigmoid()
        i = activation_func((x @ self.Wi.t()) + (h_prev @ self.Ui.t()))
        f = activation_func((x @ self.Wf.t()) + (h_prev @ self.Uf.t()))
        g = torch.tanh((x @ self.Wg.t()) + (h_prev @ self.Ug.t()))
        o = activation_func((x @ self.Wo.t()) + (h_prev @ self.Uo.t()))
        c = (f * c_prev) + (i * g)
        h = o * torch.tanh(c_prev)
        return h, c


if __name__ == '__main__':
    rnn = LSTM(10, 20)
    input = torch.randn(3, 10)
    hx = torch.randn(3, 20)
    cx = torch.randn(3, 20)
    output = []
    for i in range(3):
        hx, cx = rnn(input[i], (hx, cx))
        output.append(hx)


