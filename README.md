# Defending the unknown: Exploring reinforcement learning agents’ deployment in realistic, unseen networks
Python implementation of project exploring reinforcement algorithms using YAWNING TITAN software for network cyber defence.

## About The Project
We employed [YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN) (**YT**), an abstract, graph based cyber-security simulation environment to train intelligent agents for autonomous cyber operations. We have use model-free reinforcement learning algorithms from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for training and deploying in a set of different networks with increased complexity, status change and challenge. The main focus lies in the deployment of realistic agents in unseen networks. This work was presented at [CAMLIS](https://www.camlis.org/) conference in October 2023, this repository is publicly available but the project will not be updated with new development. 

## Abstract
The increasing number of network simulators has opened opportunities to explore and apply state-of-the-art algorithms to understand and measure the capabilities of such techniques in numerous sectors. In this regard, the recently released Yawning Titan is one example of a simplistic, but not less detailed, representation of a cyber network scenario where it is possible to train agents guided by reinforcement learning algorithms and measure their effectiveness in trying to stop an infection. In this paper, we explore how different reinforcement learning algorithms lead the training of various agents in different examples and realistic networks. We assess how we can deploy such agents in a set of networks, focusing in particular on the resilience of the agents in exploring networks with complex starting states, increased number of routes connecting the nodes and different levels of challenge, aiming to evaluate the deployment performances in realistic networks never seen before.

<img src="https://github.com/A-acuto/RLYawningTitan/blob/main/figures/exploration_RL_models_nodes_updates_paper_fix.png" width=60% heigth=60%>

# Project structure
- Networks: directory containing the examples networks to run the codes present in the repository;
- logs_dir: directory containing the trained models, information about the training performances;
- figures: directory containing plots from the paper/poster;
- utils : directory containing general codes needed to run the examples and deal with the models and Yawning Titan;
- yawning_titan: YAWNING TITAN modified version to run this example, this is based on the V-0.1.1 release.

Codes:
- train_agents.py : code that simulates the training procedure, from generating (or loading) the networks to setting the RL algorithms to train and save them;


## Authors
[Alberto Acuto](https://www.linkedin.com/in/albeacu/)<sup>1</sup>, [Simon Maskell](http://www.simonmaskell.com/)<sup>1</sup> & Jack D. <sup>2</sup>  
<sup>1</sup> University of Liverpool, School of Electrical engineering, electronics and computer science  
<sup>2</sup> Alan Turing Institute  

## Community engagement
The content of this repository, codes and results of this project can be freely accessed and reviewed, models can be re-deployed and the authors are happy to help with implementation and queries.

## Cite This Work
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

And for this specific project, please use this BibTex entry 
```bibtex
@misc{misc,
 author = {Acuto, Alberto and Maskell, Simon and D.,Jack},
 year = {2023},
 month = {10},
 title = {Defending the unknown: Exploring reinforcement learning agents’ deployment in realistic, unseen networks},
 howpublished = {GitHub},
 url ={https://github.com/A-acuto/RLYawningTitan}
}
```
# License
YAWNING-TITAN is released under MIT license. Please see [LICENSE](LICENSE) for details.
**YT** was publicly released on 20th July 2022 under MIT licence. It will continue to be developed through the Autonomous
Resilient Cyber Defence (ARCD) project, overseen by Dstl.
