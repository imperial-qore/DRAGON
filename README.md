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
Co-simulation gives resource characteristics after the migrations, and particularly = QoS + Sum(anomalies) - Migration overhead.
If any migration reduces this score (not possible in single node clusters), it is not performed and load-balancing (some existing) is performed within the cluster. 
Again anomaly detection takes place and we find migration decision again. 
These are added to the previous one and we again perform a single step simulation from original timestep, should not give anomalies now as the required migrations have been added.
Would not give cycles: (proof) x to y to z to x would never usually happen. If x to y caused anomaly in y and z led to anomalies, it would become x to x. The final QoS is lower than
the QoS of no migration. So even if such a thing happens, we ignore cycles in the final decision.

## Surrogate Model

Takes scheduling decision and time-series input and creates a reconstruction. 

## License

BSD-3-Clause. 
Copyright (c) 2021, Shreshth Tuli.
All rights reserved.

See License file for more details.
