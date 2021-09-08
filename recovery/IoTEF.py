import numpy as np
from copy import deepcopy
from .Recovery import *

class IoTEFRecovery(Recovery):
    def __init__(self, hosts, env):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.train_time_data = self.normalize_time_data(np.load('recovery/DRAGONSrc/data/' + self.env_name + '/' + 'time_series.npy'))
        self.thresholds = np.percentile(self.train_time_data, 98, axis=0) 

    def check_anomalies(self, data, thresholds):
        anomaly_per_dim = data > thresholds
        anomaly_which_dim, anomaly_any_dim = [], []
        for i in range(0, data.shape[1], 3):
            anomaly_which_dim.append(np.argmax(data[:, i:i+3] + 0, axis=1))
            anomaly_any_dim.append(np.logical_or.reduce(anomaly_per_dim[:, i:i+3], axis=1))
        anomaly_any_dim = np.stack(anomaly_any_dim, axis=1)
        anomaly_which_dim = np.stack(anomaly_which_dim, axis=1)
        return anomaly_any_dim, anomaly_which_dim

    def normalize_time_data(self, time_data):
        return time_data / (np.max(time_data, axis = 0) + 1e-8) 

    def normalize_test_time_data(self, time_data, train_time_data):
        return (time_data / (np.max(train_time_data, axis = 0) + 1e-8))

    def recover_decision(self, state, original_decision):
        anomaly_any_dim, _ = self.check_anomalies(state[1].reshape(1, -1), self.thresholds)
        host_selection = []
        for hid, anomaly in enumerate(anomaly_any_dim[0]):
            if anomaly: host_selection.append(hid)
        if host_selection == []:
            return original_decision
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.env.scheduler.FirstFitPlacement(container_selection)
        container_alloc = [-1] * len(self.env.hostlist)
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision)
        for cid, hid in target_selection:
            if container_alloc[cid] != hid:
                decision_dict[cid] = hid
        return list(decision_dict.items())

    def get_data(self):
        time_data = self.env.stats.time_series
        time_data = self.normalize_test_time_data(time_data, self.train_time_data)
        return time_data

    def run_model(self, time_series, original_decision):
        state = self.get_data()
        return self.recover_decision(state, original_decision)

