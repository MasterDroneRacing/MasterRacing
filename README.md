# MasterRacing

[![IsaacSim](https://img.shields.io/badge/IsaacSim-5.0.0-silver.svg)](https://docs.omniverse.nvidia.com/isaacsim/latest/overview.html)
[![Isaac Lab](https://img.shields.io/badge/IsaacLab-1.2.0-silver)](https://isaac-sim.github.io/IsaacLab)
[![Python](https://img.shields.io/badge/python-3.10-blue.svg)](https://docs.python.org/3/whatsnew/3.10.html)
[![Windows platform](https://img.shields.io/badge/platform-windows--64-orange.svg)](https://www.microsoft.com/en-us/)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://pre-commit.com/)
[![License](https://img.shields.io/badge/license-MIT-yellow.svg)](https://opensource.org/license/mit)

[![IMAGE ALT TEXT HERE](docs/assets/cover.png)](https://youtu.be/qxpx5dXtpNk)

## Abstract
```
Most reinforcement learning (RL)-based methods for drone racing target fixed, obstacle-free tracks, leaving the generalization to unknown, cluttered environments largely unaddressed.  This challenge stems from the need to balance racing speed and collision avoidance, limited feasible space causing policy exploration trapped in local optima during training, and perceptual ambiguity between gates and obstacles in depth maps-especially when gate positions are only coarsely specified.
To overcome these issues, we propose a two-phase learning framework: an initial soft-collision training phase that preserves policy exploration for high-speed flight, followed by a hard-collision refinement phase that enforces robust obstacle avoidance. An adaptive, noise-augmented curriculum with an asymmetric actor-critic architecture gradually shifts the policy’s reliance from privileged gate-state information to depth-based visual input. We further impose Lipschitz constraints and integrate a track-primitive generator to enhance motion stability and cross-environment generalization.
We evaluate our framework through extensive simulation and ablation studies, and validate it in real-world experiments on a computationally constrained quadrotor. The system achieves agile flight while remaining robust to gate-position errors, developing a generalizable drone racing framework with the capability to operate in diverse, partially unknown and cluttered environments.
```
## Installation

- Install [IsaacSim and IsaacLab](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html#local-installation), we recommend using Anaconda to install them.

- Clone our repository.

```bash
git clone https://github.com/MasterDroneRacing/MasterRacing.git
```

- Install our repository.

```bash
cd path_to_MasterRacing
conda activate virtual_environment (contain isaaclab repo)
bash ./install.sh
```
- Setup vscode.
```bash
python .vscode/tools/setup_vscode.py --isaac_path path_to_isaaclab_repo
```
- Verify installation.

```bash
# verify isaaclab installation
python standalone/rsl_rl/train.py --task=Template-Isaac-Velocity-Rough-Anymal-D-v0
```

- Training and evluation.
1. Click 'run and debug' button in vscode.
2. Drop down menu, select stage 0, and then click on start debugging.
3. See corrsponding options in [launch.json](https://github.com/MasterDroneRacing/MasterRacing/tree/main/.vscode/launch.json).

## License
The MasterRacing framework is released under [MIT License](https://github.com/MasterDroneRacing/MasterRacing/blob/main/LICENCE). The license files of its dependencies and assets are present in the [here](https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses).

Note that our framework requires Isaac Sim, which includes components under proprietary licensing terms. Please see the [Isaac Sim license](https://github.com/isaac-sim/IsaacLab/blob/main/docs/licenses/dependencies/isaacsim-license.txt) for information on Isaac Sim licensing.


## Common Issues

**1.Pylance Missing Indexing of Extensions**

In some VsCode versions, the indexing of part of the extensions is missing. In this case, add the path to your extension in `.vscode/settings.json` under the key `"python.analysis.extraPaths"`.

```json
{
    "python.analysis.extraPaths": [
        "<path-to-ext-repo>/exts/MasterRacing"
    ]
}
```

**2.Pylance Crash**
If you encounter a crash in `pylance`, it is probable that too many files are indexed and you run out of memory.
A possible solution is to exclude some of omniverse packages that are not used in your project.
To do so, modify `.vscode/settings.json` and comment out packages under the key `"python.analysis.extraPaths"`
Some examples of packages that can likely be excluded are:

```json
"<path-to-isaac-sim>/extscache/omni.anim.*"         // Animation packages
"<path-to-isaac-sim>/extscache/omni.kit.*"          // Kit UI tools
"<path-to-isaac-sim>/extscache/omni.graph.*"        // Graph UI tools
"<path-to-isaac-sim>/extscache/omni.services.*"     // Services tools
...
```
## Citation
When referencing our MasterRacing in your research, please cite the following paper:
```
@ARTICLE{kulkarni2025@aerialgym,
  author={Feng Yu, Yu Hu, Yang Su, Yang Deng, Linzuo Zhang, and Danping Zou},
  journal={IEEE Robotics and Automation Letters}, 
  title={Mastering Diverse, Unknown, and Cluttered Tracks for Robust
Vision-Based Drone Racing}, 
  year={2025},
  volume={},
  number={},
  pages={1-8},
  keywords={Aerial Systems: Perception and Autonomy,Collision Avoidance, Reinforcement Learning}}
```

## Acknowledgement
We gratefully acknowledge the authors of IsaacLab for their foundational contributions:
```
@article{mittal2025isaaclab,
  title={Isaac Lab: A GPU-Accelerated Simulation Framework for Multi-Modal Robot Learning},
  author={Mayank Mittal and Pascal Roth and James Tigue and Antoine Richard and Octi Zhang and Peter Du and Antonio Serrano-Muñoz and Xinjie Yao and René Zurbrügg and Nikita Rudin and Lukasz Wawrzyniak and Milad Rakhsha and Alain Denzler and Eric Heiden and Ales Borovicka and Ossama Ahmed and Iretiayo Akinola and Abrar Anwar and Mark T. Carlson and Ji Yuan Feng and Animesh Garg and Renato Gasoto and Lionel Gulich and Yijie Guo and M. Gussert and Alex Hansen and Mihir Kulkarni and Chenran Li and Wei Liu and Viktor Makoviychuk and Grzegorz Malczyk and Hammad Mazhar and Masoud Moghani and Adithyavairavan Murali and Michael Noseworthy and Alexander Poddubny and Nathan Ratliff and Welf Rehberg and Clemens Schwarke and Ritvik Singh and James Latham Smith and Bingjie Tang and Ruchik Thaker and Matthew Trepte and Karl Van Wyk and Fangzhou Yu and Alex Millane and Vikram Ramasamy and Remo Steiner and Sangeeta Subramanian and Clemens Volk and CY Chen and Neel Jawale and Ashwin Varghese Kuruttukulam and Michael A. Lin and Ajay Mandlekar and Karsten Patzwaldt and John Welsh and Huihua Zhao and Fatima Anes and Jean-Francois Lafleche and Nicolas Moënne-Loccoz and Soowan Park and Rob Stepinski and Dirk Van Gelder and Chris Amevor and Jan Carius and Jumyung Chang and Anka He Chen and Pablo de Heras Ciechomski and Gilles Daviet and Mohammad Mohajerani and Julia von Muralt and Viktor Reutskyy and Michael Sauter and Simon Schirm and Eric L. Shi and Pierre Terdiman and Kenny Vilella and Tobias Widmer and Gordon Yeoman and Tiffany Chen and Sergey Grizan and Cathy Li and Lotus Li and Connor Smith and Rafael Wiltz and Kostas Alexis and Yan Chang and David Chu and Linxi "Jim" Fan and Farbod Farshidian and Ankur Handa and Spencer Huang and Marco Hutter and Yashraj Narang and Soha Pouya and Shiwei Sheng and Yuke Zhu and Miles Macklin and Adam Moravanszky and Philipp Reist and Yunrong Guo and David Hoeller and Gavriel State},
  journal={arXiv preprint arXiv:2511.04831},
  year={2025},
  url={https://arxiv.org/abs/2511.04831}
}
```
