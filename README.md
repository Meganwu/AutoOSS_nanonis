# AutoOSS_nanonis


<!-- Badges -->
[![Paper](https://img.shields.io/badge/Paper-arXiv-blue)](https://doi.org/10.1021/jacs.4c14757)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/USERNAME/REPO)](https://github.com/Meganwu/AutoOSS_nanonis)
[![Documentation Status](https://readthedocs.org/projects/YOURDOC/badge/?version=latest)](https://YOURDOC.readthedocs.io/en/latest/?badge=latest)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.xxxxx.svg)](https://doi.org/10.5281/zenodo.13761822)
[![GitHub Release](https://img.shields.io/github/v/release/USERNAME/REPO)](https://github.com/USERNAME/REPO/releases)

---

# 📑 Table of Contents
- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Releases](#releases)
- [Citation](#citation)
- [License](#license)




# About AutoOSS
<img src="./Image/total_architecture_zn_color_hel_font.png" alt="Workflow" style="width:90%;">

We developed the framework of AutoOSS (Autonomous on-surface synthesis) to automate chemical reactions (break bond, move fragments, form bond and assebmle structures) in scanning tunneling microscopy based on Nanonis V5 (part function based on Createc is also avaialble on GitHub https://github.com/Meganwu/AutoOSS_nanonis). It comprises the remote connection, target dection module, interpetation module (image classifiers to identify reactants and products), decision-making module to optimize manipulation parameters for each function as well as miscellaneous analysis scritps. 


## Project Structure
```
.
|   LICENSE
|   README.md
\---AutoOSS
    |   __init__.py
    +---action_modules
    |   |   diss_module.py
    |   |   move_module.py
    |   |   scan_module.py
    |   +---assem_module
    |   |   |   assem_utils.py
    |   |   |   dynamic_rrt.py
    |   |   |   path_planning.py          
    +---env_modules
    |   |   basic_params.py
    |   |   dissociate_env.py
    |   |   episode_memory.py
    |   |   rrt.py
    |   |   __init__.py         
    +---img_modules
    |   |   img_attrib.py
    |   |   img_conv.py
    |   |   img_net_framework.py          
    +---img_signal_classifier
    |       boss_curr_opt.py
    |       boss_opt.py
    |       boss_opt_2_class.py
    |       boss_opt_2_messy.py
    |       boss_opt_2_target.py
    |       boss_opt_3_class.py
    |       boss_signal_opt.py
    |       curr_data.py
    |       img_data.py
    |       img_data_2_class.py
    |       img_data_2_messy.py
    |       img_data_2_target.py
    |       img_data_3_class.py
    |       topo_data.py      
    +---model_params
    |       img_classifier_best.pth
    |       messy_classifier_2_best.pth
    |       product_classifier_2_best.pth
    |       product_classifier_3_best.pth    
    +---nanonisTCP       
    +---rl_modules
    |   |   actor_critic_net.py
    |   |   ddpg_agent.py
    |   |   gaussianpolicy.py
    |   |   initi_update.py
    |   |   ppo_agent.py
    |   |   qnetwork.py
    |   |   replay_memory.py
    |   |   sac_agent.py        
    +---task_script
    |   |   analysis_result.py
    |   |   collect_images.py
    |   |   diss_mols.py
    |   |   move_mols.py
    |   |   utils.py      
    +---test_utils
    |   |   detect_mol_test.py     
    +---utils
    |   |   detect_mols.py
    |   |   extract_img_from_sxm.py
    |   |   nanonis_sxm.py
    |   |   utils.py
          
```

# Installation

## Install from package

1. Clone the repository:
   ```sh
   git clone https://github.com/Meganwu/AutoOSS_nanonis.git

2. Navigate to the main directory
   cd AutoOSS_nanonis

3. Install dependenceies
   pip install -r requirements.txt

## Install from 'conda install'

conda install -c your-anaconda-username your-package-name




# Usage

## env_module


It consists of the interface to remote connection to STM/AFM software to monitor STM, target detection.



## rl_module

The reinforcement learning module aims to optimize manipulation parameters.

## img_module

<img src="./Image/resnet18.png" alt="ResNet18 architecture" style="width:90%;">

Neural network models based on ResNet18 can be applied to identify reactants and products, where bayesian optimization technique is used to optimize hyperparameters like learning rate.

## params
The optimized neural network parameters of image classifiers were uploaded to evalute the protrusion in STM images.

## scan_module

It enables long-duration scanning along a predefined path and automatically detects target molecules, then zooms in to scan them in detail.

## dissociate_module

It introduces active learning or reinforcement learning method to look for proper dissociation parameters by multiple modes (constant current, constant height, pulse).

## move_module

It enables to explore proper manipulation parameters ((constant current, constant height, pulse) to move molecules along precise distance and directions by active learning, reinforcement learning and path planing algorithm.


## associate_module

It introduces active learning or reinforcement learning method to look for proper association parameters by multiple modes (constant current, constant height, pulse).


## assemble_module

It combines all functions above and allow more complicated tasks like assembling designed structures.



## task_script
It includes the script to show all dissociation cases with images and signal curves, to submit tasks, and all analyses in the manuscript. 


# License
Distributed under the MIT License. More details are shown in LICENSE.
