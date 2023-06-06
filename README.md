# Reinforcement Learning algorithms in YAWNING-TITAN
Python implementation of project exploring reinforcement algorithms using YAWNING TITAN for network cyber defence.

## About The Project
We employed YAWNING-TITAN (**YT**), an abstract, graph based cyber-security simulation environment to train intelligent agents for autonomous cyber operations. We have use model-free reinforcement learning algorithms from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for training and deploying in a set of different networks with increased complexity, status change and challenge. 

## Abstract
The increasing number of network simulators have opened opportunities to explore and apply state-of-the-art algorithms to understand and measure the capabilities of such techniques in numerous sectors. On this regard, the recently released Yawning Titan is one example of a simplistic, but not less detailed, representation of a cyber network scenario where it is possible to train agents guided by reinforcement learning algorithms and measure their effectiveness in trying to stop an infection. In this paper, we explore how different reinforcement learning algorithms lead the training of various agents in different example and realistic networks.We asses how we can deploy such agents in a set of networks, focusing in particular on the resilience of the agents in exploring networks with complex starting states, increased number of routes connecting the nodes and different level of challenge

![alt text](img src="https://github.com/A-acuto/RLYawningTitan/blob/main/figures/exploration_RL_models_nodes_updates_paper_fix.png" width=90% heigth=90%)

# Project structure
- Networks: directory containing the examples networks to run this example
- logs_dir
- paper_plots: directory containing plots from the paper.
- yawning_titan: YAWNING TITAN modified version to run this example, this is based on the V-0.1.1 release.



#### 3. Install `yawning-titan` into the venv along with all of it's dependencies


## Authors
[Alberto Acuto](https://www.linkedin.com/in/albeacu/), [Simon Maskell](http://www.simonmaskell.com/) &

## Cite This Work
If you would like to cite this work please use this:
```bibtex
```


If you would like to include a citation for **YT** in your work, please cite the paper published at the ICML 2022 ML4Cyber Workshop.
```bibtex
@inproceedings{inproceedings,
 author = {Andrew, Alex and Spillard, Sam and Collyer, Joshua and Dhir, Neil},
 year = {2022},
 month = {07},
 title = {Developing Optimal Causal Cyber-Defence Agents via Cyber Security Simulation},
 maintitle = {International Conference on Machine Learning (ICML)},
 booktitle = {Workshop on Machine Learning for Cybersecurity (ML4Cyber)}
}
```

# License

YAWNING-TITAN is released under MIT license. Please see [LICENSE](LICENSE) for details.
**YT** was publicly released on 20th July 2022 under MIT licence. It will continue to be developed through the Autonomous
Resilient Cyber Defence (ARCD) project, overseen by Dstl.
