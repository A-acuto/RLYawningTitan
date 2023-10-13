# Defending the unknown: Exploring reinforcement learning agents’ deployment in realistic, unseen networks
Python implementation of project exploring reinforcement algorithms using YAWNING TITAN software for network cyber defence.

## About The Project
We employed [YAWNING-TITAN](https://github.com/dstl/YAWNING-TITAN) (**YT**), an abstract, graph based cyber-security simulation environment to train intelligent agents for autonomous cyber operations. We have use model-free reinforcement learning algorithms from [Stable Baselines3](https://github.com/DLR-RM/stable-baselines3) for training and deploying in a set of different networks with increased complexity, status change and challenge. The main focus lies in the deployment of realistic agents in unseen networks. This work was presented at [CAMLIS](https://www.camlis.org/) conference in October 2023, this repository is publicly available but the project will not be updated with new development. You can see the poster presented at CAMLIS here (add link to research gate)

## Abstract
The increasing number of network simulators has opened opportunities to explore and apply state-of-the-art algorithms to understand and measure the capabilities of such techniques in numerous sectors. In this regard, the recently released Yawning Titan is one example of a simplistic, but not less detailed, representation of a cyber network scenario where it is possible to train agents guided by reinforcement learning algorithms and measure their effectiveness in trying to stop an infection. In this paper, we explore how different reinforcement learning algorithms lead the training of various agents in different examples and realistic networks. We assess how we can deploy such agents in a set of networks, focusing in particular on the resilience of the agents in exploring networks with complex starting states, increased number of routes connecting the nodes and different levels of challenge, aiming to evaluate the deployment performances in realistic networks never seen before.

<img src="https://github.com/A-acuto/RLYawningTitan/blob/main/figures/exploration_RL_models_nodes_updates_paper_fix.png" width=60% heigth=60%>

# Project structure

To explore in more detail what this project comprehends the best way to start is to run the Jupiter Notebook 
[RL_in_Yawning_titan](https://github.com/A-acuto/RLYawningTitan/blob/main/RL_in_Yawning_Titan_notebook.ipynb). 
This notebooks shows the several steps present in this project, enhancing the data provided with plots showing
the training performances, the evaluation of the models in a standard scenario and when we explore the 
agents' resilience by modifying the network. Finally, we present how the various agents perform in the zero-shot
deployment on unseen, realistic networks. This notebook also links all the codes present in the repository. 

Codes:
-
- [train_agents.py](https://github.com/A-acuto/RLYawningTitan/blob/main/train_agents.py) : code that simulates the training procedure, from generating (or loading) the networks to setting the RL algorithms to train and save them;
- [show_training_performances.py](https://github.com/A-acuto/RLYawningTitan/blob/main/show_training_performances.py) : code to plot the 
plot the training performances of the various algorithms, using the output of the Monitor files; 
- [evaluate_agents_perfomances.py](https://github.com/A-acuto/RLYawningTitan/blob/main/evaluate_agents_perfomances.py) : code to test the models
collecting the rewards obtained in seeded random network to produce statistics of the overall performances of the algorithms
- [check_network_statistics.py](https://github.com/A-acuto/RLYawningTitan/blob/main/check_network_statistics.py) : code to check the general network statistics like clustering;
- [show_agents_deployment_varying_hyperpars.py](https://github.com/A-acuto/RLYawningTitan/blob/main/show_agents_deployment_varying_hyperpars.py) : code to plot the RL algorithms mean scores while changing the hyper-parameters;
- [show_summary_plot_extensions.py](https://github.com/A-acuto/RLYawningTitan/blob/main/show_summary_plot_extensions.py) : code to plot in a single summary figure the mean performances of the various agents while modifying the network
with nodes compromised or isolated, performances against a weaker or stronger red agent and on network with fewer of more edges per node.
- [show_deployment_performances.py](https://github.com/A-acuto/RLYawningTitan/blob/main/show_deployment_perfomances.py) : code to plot the results from the deployment on realistic networks and compare the results with the scores obtained
by a random agent and the same trained RL on the synthetic networks used for training.
- [RL_plottings.py](https://github.com/A-acuto/RLYawningTitan/blob/main/RL_plottings.py) : code containing the plotting routines used to generate the plots on the project.
- [RL_utils.py](https://github.com/A-acuto/RLYawningTitan/blob/main/RL_utils.py) : code containing some ad-hoc functions used in the project for data management and handling.
- 
Directories:
- Networks: directory containing the examples networks to run the codes present in the repository;
- logs_dir: directory containing the trained models, information about the training performances;
- figures: directory containing plots from the paper/poster;
- results_data: directory containing the raw data used to produce the plots;
- utils : Yawning titan directory utils not used;
- yawning_titan: YAWNING TITAN modified version to run this example, this is based on the V-0.1.1 release.


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