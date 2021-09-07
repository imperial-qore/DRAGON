import torch
import torch.nn as nn
import torch.nn.functional as F
from .constants import *

## Simple Multi-Head Self-Attention Model
class DRAGON_simulator_16(nn.Module):
	def __init__(self):
		super(DRAGON_simulator_16, self).__init__()
		self.name = 'DRAGON_simulator_16'
		self.lr = 0.0002
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 1 # w_size = 3
		self.n_latent = self.n_feats # 10
		self.n_hidden = 32
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.discriminator = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats + self.n_hosts ** 2, self.n_latent), nn.LeakyReLU(True),
			nn.Linear(self.n_latent, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, t, s):
		score = self.discriminator(torch.cat((t.view(-1), s.view(-1))))
		return score.view(-1)

## Simple Multi-Head Self-Attention Model
class DRAGON_framework_16(nn.Module):
	def __init__(self):
		super(DRAGON_framework_16, self).__init__()
		self.name = 'DRAGON_framework_16'
		self.lr = 0.0002
		self.n_hosts = 16
		self.n_feats = 3 * self.n_hosts
		self.n_window = 1 # w_size = 3
		self.n_latent = self.n_feats # 10
		self.n_hidden = 32
		self.n = self.n_window * self.n_feats + self.n_hosts * self.n_hosts
		self.discriminator = nn.Sequential(
			nn.Linear(self.n_window * self.n_feats + self.n_hosts ** 2, self.n_latent), nn.LeakyReLU(True),
			nn.Linear(self.n_latent, self.n_hidden), nn.LeakyReLU(True),
			nn.Linear(self.n_hidden, 1), nn.Sigmoid(),
		)

	def forward(self, t, s):
		score = self.discriminator(torch.cat((t.view(-1), s.view(-1))))
		return score.view(-1)