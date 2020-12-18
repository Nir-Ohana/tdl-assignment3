import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

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
	zeros2 = np.zeros((batch_size, K-1))
	zeros3 = np.zeros((batch_size, K+T))
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
	def __init__(self, m, k):
		super(Model, self).__init__()

		self.m = m
		self.k = k

		self.rnn = nn.RNNCell(m+1, k)
		self.V = nn.Linear(k, m)

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

def main():
	# create the training data
	X, Y = copy_data(T, K, n_train)
	print('{}, {}'.format(X.shape, Y.shape))

	ohX = torch.FloatTensor(batch_size, T + 2 * K, n_characters)
	onehot(ohX, X[:batch_size])
	print('{}, {}'.format(X[:batch_size].shape, ohX.shape))

	model = Model(n_classes, hidden_size)
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
	main()
