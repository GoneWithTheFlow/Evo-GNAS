# Evo-GNAS: Evolutionary Graph Neural Architecture Search for Quadratic Unconstrained Binary Optimization
## Overview
This work proposes Evo-GNAS, an Evolutionary Graph Neural Architecture Search framework specifically designed to solve classic NP-hard combinatorial optimization problems formulated as Quadratic Unconstrained Binary Optimization (QUBO), such as Maximum Cut (Max-Cut) and Maximum Independent Set (MIS). 
<img width="1137" height="542" alt="Algorithms" src="https://github.com/user-attachments/assets/0636af0a-8bd3-42fc-bf77-9c5e92343495" />
The QUBO matrix is mapped to a weighted graph data structure, where nodes represent decision variables and edge weights denote quadratic interaction coefficients. Given the input QUBO matrix, the iterative search process of Evo-GNAS consists of the following steps:

1.Explores the optimal GNN network topology at the upper level  
2.Optimizes the GNN operations at the lower level  
3.Decode discrete gene sequences via genotype decoding to obtain executable GNN networks  
4.Evaluate the performance of each GNN individual in the population according to the QUBO objective function  
5.Implement the evolutionary loop and perform survival-of-the-fittest selection based on fitness scores. The retained high-quality architectures serve as parents for the subsequent evolutionary cycle.

## Environments
Versions of PyTorch, NumPy, PyTorch Geometric (PyG) and their dependencies are required. All numerical experiments in this study were conducted on an Ubuntu 22.04 LTS server equipped with an NVIDIA RTX 4090 GPU. Ensure that Python 3.11.14, PyTorch 2.1.0 (with CUDA 12.1 and NVCC 13.0), PyTorch Geometric 2.7.0, and NumPy 2.3.5 are installed. Please note that we have provided a file which defines the full environment required to run this code. requirements.txt
```
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

pip install -r requirements.txt
```

## Running
### test
We have integrated the optimal GNN architecture derived from the Max-Cut problem into this demo. You can reproduce the experimental results by executing the following commands:

```bash
# Navigate to the demo directory
cd QUBO-cut/demo

# Run the Max-Cut demo
python cut_70.py
```
### Run the complete code
This project requires the installation and configuration of PostgreSQL 15.14.  
Run the following commands to launch the full Evo-GNAS pipeline in the background:
```
setsid python run_evolutionary_search.py > log_ea.log 2>&1 &  
setsid python run_gnn_training.py -n train0 -g 0 > log_train0.log 2>&1 &  
setsid python run_gnn_training.py -n train1 -g 0 > log_train1.log 2>&1 &  
setsid python run_differentiable_search.py -n ds0 -g 0 > log_ds0.log 2>&1 &  
setsid python run_differentiable_search.py -n ds1 -g 0 > log_ds1.log 2>&1 &  
```

## Results

### Results on Maxcut

| Graph | DSDP | SA | Tabu | PI-GNN | RUN-CSP | CIM | **Evo-GNAS** |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| G14 | 2922 | <u>3037</u> | 3014 | 3026 | 2943 | 3020 | **3039** |
| G15 | 2938 | 2988 | 2972 | <u>2990</u> | 2928 | 2976 | **3004** |
| G22 | 12960 | <u>13186</u> | 13021 | 13181 | 13028 | **13316** | <u>13186</u> |
| G49 | **6000** | **6000** | **6000** | <u>5918</u> | **6000** | **6000** | **6000** |
| G50 | **5880** | **5880** | 5740 | 5820 | **5880** | **5880** | <u>5870</u> |
| G55 | 9960 | 9979 | 9906 | <u>10138</u> | 10116 | 10135 | **10162** |
| G70 | <u>9456</u> | 9346 | 8268 | 9421 | 9319 | 9363 | **9514** |

> **Note**: Bold numbers denote the best performance.

### Results on MIS

| Model | $n=500$ | $n=800$ | $n=1000$ | $n=2000$ |
|:---:|:---:|:---:|:---:|:---:|
| BH | 40.7 | 65.6 | 81.8 | 161.5 |
| SA | 44.9 | 70.0 | 87.0 | 165.1 |
| Tabu | **49.2** | 70.7 | 84.5 | 162.8 |
| PI-GNN | <u>49.0</u> | <u>76.7</u> | <u>96.7</u> | <u>189.0</u> |
| **Evo-GNAS** | **49.2** | **77.2** | **97.1** | **190.3** |

> **Note**: $n$ denotes the number of nodes in the graph. Bold numbers denote the best performance.



