import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *
from .dlutils import *

## Simple SlimGAN model
class StepGAN_simulator_16(nn.Module):
	def __init__(self):
		super(StepGAN_simulator_16, self).__init__()
		self.name = 'StepGAN_simulator_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_hidden = 32
		self.n_window = 1 # SlimGAN w_size = 1
		self.n = self.n_window * self.n_feats + self.n_hosts ** 2
		self.generator = nn.Sequential(
			SlimmableLinear(self.n, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_feats, 1), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			SlimmableLinear(self.n, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, 1, 1), nn.Sigmoid(),
		)

	def forward(self, t, s):
		## Generate
		z = self.generator(torch.cat((t.view(-1), s.view(-1))))
		## Discriminator
		real_score = self.discriminator(torch.cat((t.view(-1), s.view(-1))).view(1,-1))
		fake_score = self.discriminator(torch.cat((z.view(-1), s.view(-1))).view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)

## Simple SlimGAN model
class StepGAN_framework_16(nn.Module):
	def __init__(self):
		super(StepGAN_framework_16, self).__init__()
		self.name = 'StepGAN_framework_16'
		self.lr = 0.0001
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_hidden = 32
		self.n_window = 1 # SlimGAN w_size = 1
		self.n = self.n_window * self.n_feats + self.n_hosts ** 2
		self.generator = nn.Sequential(
			SlimmableLinear(self.n, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_feats, 1), nn.Sigmoid(),
		)
		self.discriminator = nn.Sequential(
			SlimmableLinear(self.n, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, self.n_hidden, 1), nn.LeakyReLU(True),
			SlimmableLinear(self.n_hidden, 1, 1), nn.Sigmoid(),
		)

	def forward(self, t, s):
		## Generate
		z = self.generator(torch.cat((t.view(-1), s.view(-1))))
		## Discriminator
		real_score = self.discriminator(torch.cat((t.view(-1), s.view(-1))).view(1,-1))
		fake_score = self.discriminator(torch.cat((z.view(-1), s.view(-1))).view(1,-1))
		return z.view(-1), real_score.view(-1), fake_score.view(-1)