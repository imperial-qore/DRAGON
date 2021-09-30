[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/DRAGON/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDRAGON&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/DRAGON/workflows/AIoT-Benchmarks/badge.svg)](https://github.com/imperial-qore/DRAGON/actions)
![Docker pulls AIoTBench](https://img.shields.io/docker/pulls/shreshthtuli/aiotbench?label=docker%20pulls%3AAIoTBench)

# DRAGON

DRAGON: Decentralized Fault Remediation using Generative Optimization Nets in Edge Federations.

We consider multiple edge nodes. 16 clusters of 1 rpi. 4 clusters of 4 rpis. 4 clusters of 2, 2, 4, 8 rpis. 
Each cluster has a rpi node as master. If the master becomes unresponsive, another node becomes master (with least utilization).

At each cluster, we perform scheduling using GOBI + PreGAN to assume fault-tolerant scheduling.
But due to non-stationarity, there might still be faults. Each node (including master) runs GON based 
anomaly detection on its own systems. Now, if there is an anomaly in any node of a cluster, we classify that
as an anomaly in the cluster. For any anomaly in a cluster, the master co-simulates the migration of its
container(s) to another cluster (pulling the resource predictions from other nodes updated periodically).


Thus, every cluster proposes a set of migrations, these are compiled together and then we perform a single-step co-simulated fictitious play. Like x to y to z is changed to x to z.
Co-simulation gives resource characteristics after the migrations, and particularly Fitness = QoS + Sum(anomalies) - Migration overhead.
If any migration reduces this score (not possible in single node clusters), it is not performed and load-balancing (some existing) is performed within the cluster. 
Again anomaly detection takes place and we find migration decision again. 
These are added to the previous one and we again perform a single step simulation from original timestep, should not give anomalies now as the required migrations have been added.
Would not give cycles: (proof) x to y to z to x would never usually happen. If x to y caused anomaly in y and z led to anomalies, it would become x to x. The final QoS is lower than
the QoS of no migration. So even if such a thing happens, we ignore cycles in the final decision.

DRAGON+: Now instead of checking the fitness of only the next state, we find the decision using DRAGON, next time-series window using GON to give us the next to next state. We continue this for say 5-10 timesteps and take a discounted cummulative fitness score. 


To speed up the scheduling time compared to trasditional GON, (1) we use topology based attention, (2) we use second-order Adahessian optimization and (3) we start from W_t to construct W_{t+1} instead of a random noise sample.

## Figures

0. Motivational example (without FT {memory application mid, performance low}, with memory intensive FT {memory high application low, performance low}, with memory efficient FT {memory low application mid, performance hi}). Plot mem app + model, swap % (dot), response time. Change RAM usage using ulimit and cgcreate to limit memory and swap usage, Swappiness = 0.
1. System Model (three different use cases)
2. GON model diagram (inputs: scheduling decision, charcteristics window; feats: topology GTN based attention)
3. Visualization of Attention (with topology and CPU utilization)
4. @FTSAD dataset statistics (table)
5. @Loss curves on FTSAD dataset. (loss, detection/diagnosis accuracy). F1 and F1/GB with memory for FTSAD datasets.
6. @Comparison (detection/diagnosis accuracy table)
7. @QoS comparison (response time, energy, sla, migration counts, scheduling time) for each use case.
8. @Ablation Analysis (w/o Topology attention (accuracy, memory, time slightly), w/o adahessian (time), w/o starting point (time))
9. @Sensitivity Analysis (Detection/diagnosis/memory/scheduling time with window size, learning rate)

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
