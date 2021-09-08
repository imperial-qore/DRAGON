import numpy as np
from sklearn_extensions.fuzzy_kmeans import FuzzyKMeans
from .Recovery import *

class FDMRRecovery(Recovery):
    def __init__(self, hosts, env, training = False):
        super().__init__()
        self.hosts = hosts
        self.env_name = 'simulator' if env == '' else 'framework'
        self.kmeans = FuzzyKMeans(k=2)
        self.detections = (0, 0)

    def recover_decision(self, pred_values, original_decision):
        kmean_lists = pred_values.reshape(self.hosts, 3)
        kmeans = self.kmeans.fit(kmean_lists)
        self.detections = (self.detections[0], self.detections[1] + 1)
        # Single label found
        if len(np.unique(kmeans.labels_)) == 1:
            return original_decision
        # Clusters too close
        if abs(np.sum(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1])) < (0.025 * self.hosts):
            return original_decision
        self.detections = (self.detections[0] + 1, self.detections[1])
        if np.sum(kmeans.cluster_centers_[0] - kmeans.cluster_centers_[1]) < 0:
            higher, lower = 0, 1
        else:
            higher, lower = 1, 0
        host_selection = []
        for host_embedding in range(kmean_lists.shape[1]):
            if kmeans.labels_[host_embedding] == higher:
                host_selection.append(host_embedding)
        container_selection = self.env.scheduler.MMTContainerSelection(host_selection)
        target_selection = self.env.scheduler.LeastFullPlacement(container_selection)
        container_alloc = [-1] * len(self.env.hostlist)
        for c in self.env.containerlist:
            if c and c.getHostID() != -1: 
                container_alloc[c.id] = c.getHostID()
        decision_dict = dict(original_decision)
        for cid, hid in target_selection:
            if container_alloc[cid] != hid:
                decision_dict[cid] = hid
        print(f"Percent Detection: {int(100 * self.detections[0] / self.detections[1])}, \
            Hosts: {len(host_selection)}")
        return list(decision_dict.items())


    def run_model(self, time_series, original_decision):
        return self.recover_decision(time_series[1], original_decision)

