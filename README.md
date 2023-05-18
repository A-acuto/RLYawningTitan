![Python](https://img.shields.io/badge/python-3.8%20%7C%203.9%20%7C%203.10-blue)
![GitHub](https://img.shields.io/github/license/dstl/YAWNING-TITAN)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/dstl/YAWNING-TITAN/Python%20package)
![GitHub Workflow Status](https://img.shields.io/github/workflow/status/dstl/YAWNING-TITAN/build-sphinx-to-github-pages?label=docs)
![GitHub release (latest SemVer)](https://img.shields.io/github/v/release/dstl/YAWNING-TITAN)
![GitHub commits since latest release (by SemVer)](https://img.shields.io/github/commits-since/dstl/YAWNING-TITAN/latest)
![GitHub issue custom search](https://img.shields.io/github/issues-search?label=active%20bug%20issues&query=repo%3Adstl%2FYAWNING-TITAN%20is%3Aopen%20label%3Abug)
![GitHub issue custom search](https://img.shields.io/github/issues-search?label=active%20feature%20requests&query=repo%3Adstl%2FYAWNING-TITAN%20is%3Aopen%20label%3Afeature_request)
![GitHub Discussions](https://img.shields.io/github/discussions/dstl/YAWNING-TITAN)

# Reinforcement Learning in YAWNING-TITAN


## About The Project
YAWNING-TITAN (**YT**) is an abstract, graph based cyber-security simulation environment that supports the training of
intelligent agents for autonomous cyber operations. YAWNING-TITAN currently only supports defensive autonomous agents
who face off against probabilistic red agents.

**YT** has been designed with the following things in mind:
- Simplicity over complexity
- Minimal Hardware Requirements
- Operating System agnostic
- Support for a wide range of algorithms
- Enhanced agent/policy evaluation support
- Flexible environment and game rule configuration
- Generation of evaluation episode visualisations (gifs)

**YT** was publicly released on 20th July 2022 under MIT licence. It will continue to be developed through the Autonomous
Resilient Cyber Defence (ARCD) project, overseen by Dstl.

## Contributing to YAWNING-TITAN
Found a bug, have an idea/feature you'd like to suggest, or just want to get involved with the YT community, please read
our [How to contribute to YAWNING-TITAN?](CONTRIBUTING.md) guidelines.




#### 3. Install `yawning-titan` into the venv along with all of it's dependencies

```bash
python3 -m pip install -e .
```


This will install all the dependencies including algorithm libraries. These libraries
all use `torch`. If you'd like to install `tensorflow` for use with Rllib, you can do this manually
or install `tensorflow` as an optional dependency by postfixing the command in step 3 above with the `[tensorflow]` extra. Example:

```bash
python3 -m pip install -e .[tensorflow]
```
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

# License

YAWNING-TITAN is released under MIT license. Please see [LICENSE](LICENSE) for details.
