import numpy as np
from pyqubo import Array
import torch
import torch.nn.functional as F
import networkx as nx
import jax.numpy as jnp


# Build the Max-Cut Hamiltonian and compile it into a QUBO model.
def create_max_cut_model(graph):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = 0
    for u, v in graph.edges:
        hamiltonian -= (X[u] - X[v]) ** 2

    return hamiltonian.compile()


# Build the MIS Hamiltonian with penalties and compile it into a QUBO model.
def create_mis_model(graph, penalty=2):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = -sum(X)
    for u, v in graph.edges:
        hamiltonian += penalty * (X[u] * X[v])

    return hamiltonian.compile()


# Build the Max-Cut Hamiltonian and compile it into a QUBO model.
def create_Q_matrix(graph, is_max_cut=True):
    if is_max_cut:
        model = create_max_cut_model(graph)
    else:
        model = create_mis_model(graph)
    N = graph.number_of_nodes()
    extract_val = lambda x: int(x[2:-1])
    Q_matrix = np.zeros((N, N))
    qubo_dict, _ = model.to_qubo()
    for (a, b), quv in qubo_dict.items():
        u = min(extract_val(a), extract_val(b))
        v = max(extract_val(a), extract_val(b))
        Q_matrix[u, v] = quv
    Q_matrix = torch.tensor(Q_matrix, dtype=torch.float32)

    return Q_matrix


# Compute the QUBO loss from GNN probabilities and the Q matrix.
def loss_func(probs, Q_matrix):
    probs_ = torch.unsqueeze(probs, 1)
    cost = (probs_.T @ Q_matrix @ probs_).squeeze()
    return cost
