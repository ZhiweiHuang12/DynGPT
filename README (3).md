

# DynGPT: a generative AI framework for learning stochastic dynamics through state transition networks

<p align="center">
  <img src="https://github.com/ZhiweiHuang12/DynGPT/blob/main/docs/_static/image/logo.jpg" alt="logo">
</p>

## Brief introduction
DynGPT is a GPT-driven framework designed to solve generalized multi-dimensional state transition networks (STNs) and learn stochastic dynamics from static observed data. our framework is extensible user-friendly; Users can solve their specific STNs by simply providing configuration files, allowing for the subsequent inference of the corresponding observed data. DynGPT consists of two core modules: DynGPT-Solver and DynGPT-Inferrer.

## DynGPT-Solver: Solving the stationary distribution of the state transition networks 
Solving the steady-state distribution of complex STNs remains challenging, as analytical and numerical methods often demand substantial computational resources or trade accuracy for efficiency. To address this, DynGPT-Solver employs an autoregressive transformer-based architecture to efficiently solve the joint stationary distribution of multi-dimensional STNs. 

![logo](https://github.com/ZhiweiHuang12/DynGPT/blob/main/docs/_static/image/solver.jpg)

## DynGPT-Inferrer: Learning stochastic dynamic from static observed data
Understanding how stochasticity enhances robustness and flexibility is a key focus in systems dynamics research, requiring the modeling and inference of underlying stochastic mechanisms from observed data. Leveraging a trained neural network,  DynGPT-Inferrer efficiently estimates the Bayesian posterior distributions of STNs parameters by utilizing automatic differentiation and neural approximate Bayesian computation.

![logo](https://github.com/ZhiweiHuang12/DynGPT/blob/main/docs/_static/image/inferrer.jpg)

## Installation

Run the following command to create a new environment  and activate the environment. 

```bash
conda create --name dyngpt_env python=3.8
conda activate dyngpt_env
```

Download the code and use the `cd` command to navigate to the directory containing the `setup.py` file

```bash
git clone this-repo-url
cd DynGPT
```

Run the following command to install the package:

```bash
pip install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0 --extra-index-url https://download.pytorch.org/whl/cu113
pip install -r requirements.txt
pip install .
```

## Documentation and tutorials
For how to use DynGPT, please see DynGPT documentation that is available through the link https://dyngpt.readthedocs.io/
## Applications
DynGPT can be used for the following tasks:

- Solving the generalized state transition networks efficiently.

- Inferring the multi-dimensional observed data to estimate the parameters of the corresponding state transition networks. 
