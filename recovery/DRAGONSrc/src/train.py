from .constants import *
from .utils import *
import torch.nn as nn
from tqdm import tqdm
from .gen import *
from .plotter import *

mse_loss = nn.MSELoss(reduction = 'mean')

def custom_loss(model, pred_state, true_state):
	aloss = mse_loss(pred_state.view(-1), torch.tensor(true_state, dtype=torch.double))
	return aloss

def backprop(epoch, model, optimizer, train_time_data, train_schedule_data, stats, norm_series, thresholds, training = True):
	aloss_list = []
	bcel = nn.BCELoss(reduction = 'mean')
	real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
	real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
	for i in tqdm(range(train_time_data.shape[0]), leave=False, position=1):
		state, schedule = train_time_data[i], train_schedule_data[i]
		real_data, fake_data = state, gen(model, state, schedule, real_label, bcel, 1e-3)
		real, fake = model(real_data, schedule), model(fake_data, schedule)
		loss = (bcel(real, real_label) + bcel(fake, fake_label)) / 2
		aloss_list.append(loss.item())
		if training:
			optimizer.zero_grad(); loss.backward(); optimizer.step()
	tqdm.write(f'Epoch {epoch},\tLoss = {np.mean(aloss_list)}')
	return np.mean(aloss_list)

# Accuracy 
def anomaly_accuracy(pred_state, target_anomaly, thresholds, model_plotter):
	correct = 0; res_list = []; tp, fp, tn, fn = 0, 0, 0, 0
	anomaly_any_dim, _ = check_anomalies(pred_state.view(1, -1).detach().clone().numpy(), thresholds, model_plotter.env)
	anomaly_any_dim = anomaly_any_dim[0] + 0
	for i, res in enumerate(anomaly_any_dim):
		res_list.append(res)
		if res == target_anomaly[i]:
			correct += 1
			if target_anomaly[i] == 1: tp += 1
			else: tn += 1
		else:
			if target_anomaly[i] == 1: fn += 1
			else: fp += 1
	model_plotter.update_anomaly(res_list, target_anomaly, correct/pred_state.shape[0])
	return correct*4/pred_state.shape[1], tp, tn, fp, fn

def accuracy(model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, model_plotter):
	anomaly_correct = 0; tpl, tnl, fpl, fnl = [], [], [], []
	bcel = nn.BCELoss(reduction = 'mean')
	real_label, fake_label = torch.tensor([0.9]), torch.tensor([0.1]) # label smoothing
	real_label, fake_label = real_label.type(torch.DoubleTensor), fake_label.type(torch.DoubleTensor)
	for i, d in enumerate(train_time_data):
		state, schedule = train_time_data[i], train_schedule_data[i]
		pred_state = gen(model, state, schedule, real_label, bcel, 1e-3)
		# pred_state = model(train_time_data[i], train_schedule_data[i])
		model_plotter.update_lines(pred_state.view(-1), train_time_data[i][-1])
		res, tp, tn, fp, fn = anomaly_accuracy(pred_state, anomaly_data[i], thresholds, model_plotter)
		anomaly_correct += min(1, res)
		tpl.append(tp); tnl.append(tn); fpl.append(fp); fnl.append(fn)
	tp, fp, tn, fn = np.mean(tpl), np.mean(fpl), np.mean(tnl), np.mean(fnl)
	p, r = tp/(tp+fp), tp/(tp+fn)
	tqdm.write(f'P = {p}, R = {r}, F1 = {2 * p * r / (p + r)}')
	return anomaly_correct / len(train_time_data), 2 * p * r / (p + r)