import sys
sys.path.append('recovery/DRAGONSrc/')

import numpy as np
from copy import deepcopy
from .Recovery import *
from .DRAGONSrc.src.constants import *
from .DRAGONSrc.src.utils import *
from .DRAGONSrc.src.train import *

class DRAGONRecovery(Recovery):
    def __init__(self, hosts, env):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.model_name = f'DRAGON_{self.env_name}_{hosts}'
        self.model_loaded = False

    def load_model(self):
        # Load training time series and thresholds
        self.train_time_data = normalize_time_data(np.load(data_folder + self.env_name + '/' + data_filename))
        self.thresholds = np.percentile(self.train_time_data, PERCENTILES, axis=0) 
        if self.env_name == 'simulator': self.thresholds *= percentile_multiplier
        # Load encoder model
        self.model, self.optimizer, self.epoch, self.accuracy_list = \
            load_model(model_folder, f'{self.model_name}.ckpt', self.model_name)
        # Train the model is not trained
        if self.epoch == -1: self.train_model()
        # Freeze encoder
        freeze(self.model); self.model_loaded = True

    def train_model(self):
        self.model_plotter = Model_Plotter(self.env_name, self.model_name)
        folder = os.path.join(data_folder, self.env_name); norm_series = self.train_time_data
        train_time_data, train_schedule_data, anomaly_data, class_data, thresholds = load_dataset(folder, self.model)
        for self.epoch in tqdm(range(self.epoch+1, self.epoch+num_epochs+1), position=0):
            aloss = backprop(self.epoch, self.model, self.optimizer, train_time_data, train_schedule_data, self.env.stats, norm_series, thresholds)
            anomaly_score, f1 = accuracy(self.model, train_time_data, train_schedule_data, anomaly_data, class_data, thresholds, self.model_plotter)
            tqdm.write(f'Epoch {self.epoch},\tAScore = {anomaly_score}')
            self.accuracy_list.append((aloss, anomaly_score, f1))
            self.model_plotter.plot(self.accuracy_list, self.epoch)
            save_model(model_folder, f'{self.model_name}.ckpt', self.model, self.optimizer, self.epoch, self.accuracy_list)

    def getObjective(self, cid, hostID):
        energy = self.env.stats.runSimpleSimulation([(cid, hostID)])
        return energy

    def optimize_decision(self, state, schedule):
        bcel = nn.BCELoss(reduction = 'mean')
        real_label = torch.tensor([0.9]).type(torch.DoubleTensor)
        result, oldResult = {}, {} 
        bestScore, oldScore = 1000, 10000
        while bestScore < oldScore:
            oldScore, oldResult = bestScore, result
            # update schedule based on recovery migrations
            for cid in result:
                orig_host = torch.argmax(schedule[cid])
                schedule[cid][orig_host] = 0
                schedule[cid][result[cid]] = 1
            # get anomalies
            pred_state = gen(self.model, state, schedule, real_label, bcel, 1e-3)
            pred_state = pred_state.view(1, -1).detach().clone().numpy()
            anomaly_any_dim, _ = check_anomalies(pred_state, self.thresholds, self.env_name)
            hostlist = np.where(anomaly_any_dim[0])[0].tolist()
            # select MMT containers
            selectedContainerIDs = []
            for hostID in hostlist:
                containerIDs = self.env.getContainersOfHost(hostID)
                if containerIDs:
                    containerIPS = [self.env.containerlist[cid].getRAM()[0] for cid in containerIDs]
                    selectedContainerIDs.append(containerIDs[np.argmin(containerIPS)])
            scorecount = 0
            # select target hosts based on co-simulated ficitious play
            for cid in containerIDs:
                scores = [self.getObjective(cid, hostID) for hostID, _ in enumerate(self.env.hostlist)]
                result[cid] = np.argmin(scores)
                scorecount += np.min(scores)
            scorecount = scorecount / (len(containerIDs) + 1e-4)
            migration_overhead = len(containerIDs) / (self.env.getNumActiveContainers() + 1e-4)
            bestScore = scorecount + migration_overhead
        return result

    def get_data(self):
        schedule_data = torch.tensor(self.env.scheduler.result_cache).double()
        time_data = self.env.stats.time_series
        time_data = normalize_test_time_data(time_data, self.train_time_data)
        if time_data.shape[0] >= self.model.n_window: time_data = time_data[-self.model.n_window:]
        time_data = convert_to_windows(time_data, self.model)[-1]
        return time_data, schedule_data

    def run_model(self, time_series, original_decision):
        if not self.model_loaded: self.load_model()
        state, schedule = self.get_data()
        result = self.optimize_decision(state, schedule)
        prev_alloc = {}
        for c in self.env.containerlist:
            oneHot = [0] * len(self.env.hostlist)
            if c: prev_alloc[c.id] = c.getHostID()
        decision = []
        for cid, hid in original_decision:
            new_host = result[cid] if cid in result else hid
            if prev_alloc[cid] != new_host: decision.append((cid, new_host))
        return decision

