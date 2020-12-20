import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import sys
import math

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(np.__version__)
print(torch.__version__)

# Set the seed of PRNG manually for reproducibility
seed = 1234
torch.manual_seed(seed)
np.random.seed(seed)


# Copy data
def copy_data(T, K, batch_size):
    seq = np.random.randint(1, high=9, size=(batch_size, K))
    zeros1 = np.zeros((batch_size, T))
    zeros2 = np.zeros((batch_size, K - 1))
    zeros3 = np.zeros((batch_size, K + T))
    marker = 9 * np.ones((batch_size, 1))

    x = torch.LongTensor(np.concatenate((seq, zeros1, marker, zeros2), axis=1))
    y = torch.LongTensor(np.concatenate((zeros3, seq), axis=1))

    return x, y


# one hot encoding
def onehot(out, input):
    out.zero_()
    in_unsq = torch.unsqueeze(input, 2)
    out.scatter_(2, in_unsq, 1)


# Class for handling copy data
class Model(nn.Module):
    def __init__(self, m, k, architecture):
        super(Model, self).__init__()

        self.m = m
        self.k = k
        self.architecture = architecture
        self.rnn = architecture(m + 1, k).cuda(device)
        self.V = nn.Linear(k, m).cuda(device)

        # loss for the copy data
        self.loss_func = nn.CrossEntropyLoss()

    def forward(self, inputs):
        state = torch.zeros(inputs.size(0), self.k, requires_grad=False)

        outputs = []
        for input in torch.unbind(inputs, dim=1):
            state = self.rnn(input, state)
            outputs.append(self.V(state))

        return torch.stack(outputs, dim=1)

    def loss(self, logits, y):
        return self.loss_func(logits.view(-1, 9), y.view(-1))


T = 5
K = 3

batch_size = 128
iter = 5000
n_train = iter * batch_size
n_classes = 9
hidden_size = 64
n_characters = n_classes + 1
lr = 1e-3
print_every = 20


class MLP(nn.Module):

    def __init__(self, in_features, out_features):
        super(MLP, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1 / in_features
        bound = math.sqrt(k)
        # toch.rand returns a tensor samples uniformly in [0, 1).
        # we scaling it to [l = -bound, r = bound] using the formula: (l - r) * torch.rand(x, y) + r
        self.W = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound

    def forward(self, x):
        _, input_num = x.shape
        if input_num != self.in_features:
            sys.exit(f'Wrong x size: {x}')
        y = x @ self.W.t()
        return y


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
        self.W = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound
        self.U = (-2 * bound) * torch.rand(out_features, out_features).cuda(device) + bound

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


class LSTM(nn.Module):

    def __init__(self, in_features, out_features):
        super(LSTM, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        k = 1 / out_features
        bound = math.sqrt(k)
        # toch.rand returns a tensor samples uniformly in [0, 1).
        # we scaling it to [l = -bound, r = bound] using the formula: (l - r) * torch.rand(x, y) + r
        self.Wi = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound
        self.Ui = (-2 * bound) * torch.rand(out_features, out_features).cuda(device) + bound
        self.Wf = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound
        self.Uf = (-2 * bound) * torch.rand(out_features, out_features).cuda(device) + bound
        self.Wg = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound
        self.Ug = (-2 * bound) * torch.rand(out_features, out_features).cuda(device) + bound
        self.Wo = (-2 * bound) * torch.rand(out_features, in_features).cuda(device) + bound
        self.Uo = (-2 * bound) * torch.rand(out_features, out_features).cuda(device) + bound

    def forward(self, x, state=None):
        h_prev = state[0] if state is not None else None
        c_prev = state[1] if state is not None else None

        # check validity of x
        _, input_num = x.shape
        if input_num != self.in_features:
            sys.exit(f'Wrong x size: {x}')

        if h_prev is None or c_prev is None:
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


def main(architecture):
    # create the training data
    X, Y = copy_data(T, K, n_train)
    print('{}, {}'.format(X.shape, Y.shape))

    ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
    onehot(ohX, X[:batch_size])
    print('{}, {}'.format(X[:batch_size].shape, ohX.shape))

    model = Model(n_classes, hidden_size, architecture)
    model.train()

    opt = torch.optim.RMSprop(model.parameters(), lr=lr)

    for step in range(iter):
        bX = X[step * batch_size: (step + 1) * batch_size]
        bY = Y[step * batch_size: (step + 1) * batch_size]

        onehot(ohX, bX)

        opt.zero_grad()
        logits = model(ohX)
        loss = model.loss(logits, bY)
        loss.backward()
        opt.step()

        if step % print_every == 0:
            print('Step={}, Loss={:.4f}'.format(step, loss.item()))


if __name__ == "__main__":
    main(MLP)
    main(RNN)
    main(LSTM)
