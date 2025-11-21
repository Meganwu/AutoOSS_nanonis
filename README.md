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

## dissociate_module

## move_module


## associate_module

## assemble_module



## task_script
It includes the script to show all dissociation cases with images and signal curves, to submit tasks, and all analyses in the manuscript. 


# License
Distributed under the MIT License. More details are shown in LICENSE.
