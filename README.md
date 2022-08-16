[![License](https://img.shields.io/badge/License-BSD%203--Clause-red.svg)](https://github.com/imperial-qore/DRAGON/blob/master/LICENSE)
![Python 3.7, 3.8](https://img.shields.io/badge/python-3.7%20%7C%203.8-blue.svg)
[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fimperial-qore%2FDRAGON&count_bg=%23FFC401&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com)
[![Actions Status](https://github.com/imperial-qore/DRAGON/workflows/AIoT-Benchmarks/badge.svg)](https://github.com/imperial-qore/DRAGON/actions)
![Docker pulls AIoTBench](https://img.shields.io/docker/pulls/shreshthtuli/aiotbench?label=docker%20pulls%3AAIoTBench)

# DRAGON

DRAGON: Decentralized Fault Remediation using Generative Optimization Nets in Edge Federations.

Edge Federation is a new computing paradigm that seamlessly interconnects the resources of multiple edge service providers.  A key challenge in such systems is the deployment of latency-critical and AI based resource-intensive applications in constrained devices. To address this challenge, we propose a novel memory-efficient deep learning based model, namely generative optimization networks (GON). Unlike GANs, GONs use a single network to both discriminate input and generate samples, significantly reducing their memory footprint. Leveraging the low memory footprint of GONs, we propose a decentralized fault-tolerance method called DRAGON that runs simulations (as per a digital modeling twin) to quickly predict and optimize the performance of the edge federation. Extensive experiments with real-world edge computing benchmarks on multiple Raspberry-Pi based federated edge configurations show that DRAGON can outperform the baseline methods in fault-detection and Quality of Service (QoS) metrics. Specifically, the proposed method gives higher F1 scores for fault-detection than the best deep learning (DL) method, while consuming lower memory than the heuristic methods. This allows for improvement in energy consumption, response time and service level agreement violations by up to 74, 63 and 82 percent, respectively. 

## Quick Test
Clone repo.
```console
git clone https://github.com/imperial-qore/DRAGON.git
cd CAROL/
```
Install dependencies.
```console
sudo apt -y update
python3 -m pip --upgrade pip
python3 -m pip install matplotlib scikit-learn
python3 -m pip install -r requirements.txt
python3 -m pip install torch==1.7.1+cpu torchvision==0.8.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
export PATH=$PATH:~/.local/bin
```
Change line 116 in `main.py` to use one of the implemented fault-tolerance techniques: `DRAGONRecovery`, `DRAGON2Recovery`, `PBFMRecovery`, `FDMRRecovery`, `TBAFTRecovery`, `MedusaRecovery`, `IoTEFRecovery`, `TopoMADRecovery` or `StepGANRecovery` and run the code using the following command.
```console
python3 main.py
````

## External Links
| Items | Contents | 
| --- | --- |
| **Pre-print** | (coming soon) |
| **Contact**| Shreshth Tuli ([@shreshthtuli](https://github.com/shreshthtuli))  |
| **Funding**| Imperial President's scholarship |

## Cite this work
Our work is accepted in IEEE TNSM. Cite our work using the bibtex entry below.
```bibtex
@inproceedings{tuli2022dragon,
  title={{DRAGON: Decentralized Fault Tolerance in Edge Federations}},
  author={Tuli, Shreshth and Casale, Giuliano and Jennings, Nicholas R},
  booktitle={IEEE Transactions on Network and Service Management (TNSM)},
  year={2022},
  organization={IEEE}
}
```
## License

BSD-3-Clause. 
Copyright (c) 2022, Shreshth Tuli.
All rights reserved.

See License file for more details.
