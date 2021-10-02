import os
import torch
import numpy as np
from .constants import *
from .models import *

def convert_to_windows(data, model):
	data = torch.tensor(data).double()
	windows = []; w_size = model.n_window
	for i, g in enumerate(data): 
		if i >= w_size: w = data[i-w_size:i]
		else: w = torch.cat([data[0].repeat(w_size-i, 1), data[0:i]])
		windows.append(w)
	return torch.stack(windows)

def check_anomalies(data, thresholds, env='s'):
	anomaly_per_dim = data > thresholds
	checkdim = 2 if 'f' in env else 3
	anomaly_which_dim, anomaly_any_dim = [], []
	for i in range(0, data.shape[1], 3):
		anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
		anomaly_any_dim.append(np.logical_or.reduce(anomaly_per_dim[:, i:i+checkdim], axis=1))
	anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
	anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
	return anomaly_any_dim, anomaly_which_dim

def form_test_dataset(data):
	thresholds = np.percentile(data, PERCENTILES, axis=0) 
	if 1 - max(thresholds) < 1e-5: thresholds *= percentile_multiplier
	anomaly_any_dim, anomaly_which_dim = check_anomalies(data, thresholds)
	return anomaly_any_dim + 0, anomaly_which_dim, thresholds

def load_npyfile(folder, fname):
	path = os.path.join(folder, fname)
	if not os.path.exists(path):
		raise Exception('Data not found ' + path)
	return np.load(path)

def load_dataset(folder, model):
	time_data = load_npyfile(folder, data_filename)
	time_data = normalize_time_data(time_data) # Normalize data
	train_schedule_data = torch.tensor(load_npyfile(folder, schedule_filename)).double()
	train_time_data = convert_to_windows(time_data, model)
	anomaly_data, class_data, thresholds = form_test_dataset(time_data)
	return train_time_data, train_schedule_data, anomaly_data, class_data, thresholds

def save_model(folder, fname, model, optimizer, epoch, accuracy_list):
	path = os.path.join(folder, fname)
	# if 'DeepFT' in model.name: print(model.prototype)
	torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'accuracy_list': accuracy_list}, path)

def load_model(folder, fname, modelname):
	import recovery.StepGANSrc.src.models
	path = os.path.join(folder, fname)
	model_class = getattr(recovery.StepGANSrc.src.models, modelname)
	model = model_class().double()
	optimizer = torch.optim.Adam(model.parameters() , lr=model.lr, weight_decay=1e-5)
	if os.path.exists(path):
		print(f"{color.GREEN}Loading pre-trained model: {model.name}{color.ENDC}")
		checkpoint = torch.load(path)
		model.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		epoch = checkpoint['epoch']
		accuracy_list = checkpoint['accuracy_list']
	else:
		print(f"{color.GREEN}Creating new model: {model.name}{color.ENDC}")
		epoch = -1; accuracy_list = []
	return model, optimizer, epoch, accuracy_list

def convertToOneHot(dat, cpu_old, HOSTS):
    alloc = []
    for i in dat:
        oneHot = [0] * HOSTS; alist = i.tolist()[-HOSTS:]
        oneHot[alist.index(max(alist))] = 1; alloc.append(oneHot)
    new_dat_oneHot = torch.cat((cpu_old, torch.FloatTensor(alloc)), dim=1)
    return new_dat_oneHot

# Misc
def normalize_time_data(time_data):
	return time_data / (np.max(time_data, axis = 0) + 1e-8) 

def normalize_test_time_data(time_data, train_time_data):
	return (time_data / (np.max(train_time_data, axis = 0) + 1e-8))

def run_simulation(stats, schedule_data):
    datapoint = stats.runStateSimulation(schedule_data)
    return datapoint

def get_classes(embeddings, model):
	class_list = []
	for e in embeddings:
		if (e == 0).all().item():
			class_list.append(-1); continue
		distances = np.array([(torch.mean((e - p)**2)).item() for p in model.prototype])
		class_list.append(np.argmin(distances))
	return class_list

def freeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = False

def unfreeze(model):
	for name, p in model.named_parameters():
		p.requires_grad = True

class color:
	HEADER = '\033[95m'
	BLUE = '\033[94m'
	GREEN = '\033[92m'
	RED = '\033[93m'
	FAIL = '\033[91m'
	ENDC = '\033[0m'
	BOLD = '\033[1m'
	UNDERLINE = '\033[4m'