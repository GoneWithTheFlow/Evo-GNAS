import numpy as np
from pyqubo import Array
import torch
import torch.nn.functional as F
import networkx as nx
import jax.numpy as jnp


# Build the Max-Cut Hamiltonian model.
def create_max_cut_model(graph):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = 0
    for u, v in graph.edges:
        hamiltonian -= (X[u] - X[v]) ** 2

    return hamiltonian.compile()


# Build the MIS Hamiltonian model with penalties.
def create_mis_model(graph, penalty=2):
    N = graph.number_of_nodes()
    X = Array.create("X", shape=(N,), vartype="BINARY")

    hamiltonian = -sum(X)
    for u, v in graph.edges:
        hamiltonian += penalty * (X[u] * X[v])

    return hamiltonian.compile()


# Create the QUBO matrix for the selected graph problem.
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


# Compute the QUBO objective used as the training loss.
def loss_func(probs, Q_matrix):
    probs_ = torch.unsqueeze(probs, 1)
    cost = (probs_.T @ Q_matrix @ probs_).squeeze()
    return cost


# Count selected edges that violate the independent-set constraint.
def count_selected_edge_conflicts(bitstring, graph):
    """Count conflicting edges whose endpoints are both selected."""
    bs = bitstring.detach().view(-1)
    selected = set(torch.nonzero(bs > 0.5, as_tuple=False).view(-1).cpu().numpy().tolist())
    conflicts = 0
    for u, v in graph.edges():
        if u in selected and v in selected:
            conflicts += 1
    return conflicts


# Extract unique undirected edge pairs from edge_index.
def get_undirected_edge_pairs(edge_index):
    """Extract unique undirected edge pairs (u < v) from edge_index."""
    u = edge_index[0].long()
    v = edge_index[1].long()
    mask = u < v
    return u[mask], v[mask]


# Count conflicts from tensor edge pairs.
def count_selected_edge_conflicts_tensor(bitstring, edge_u, edge_v):
    """Count conflicting edges from tensor edge indices."""
    if edge_u.numel() == 0:
        return 0
    selected = bitstring.detach().view(-1) > 0.5
    conflicts = (selected[edge_u] & selected[edge_v]).sum().item()
    return int(conflicts)


# Repair a tensor bitstring into a valid independent set.
def repair_to_independent_set_tensor(bitstring, edge_u, edge_v, scores=None):
    """
    Repair a candidate solution into an independent set with tensorized conflict counts.
    The rule matches the original implementation: remove the highest-conflict node,
    """
    original_device = bitstring.device
    original_dtype = bitstring.dtype

    bs = bitstring.detach().view(-1)
    n_nodes = bs.numel()
    selected = bs > 0.5

    if not torch.any(selected) or edge_u.numel() == 0:
        return bitstring.detach().clone()

    score_vec = None
    if scores is not None:
        score_vec = scores.detach().view(-1)

    conflict_deg = torch.zeros(n_nodes, dtype=torch.long, device=selected.device)
    edge_ones = torch.ones(edge_u.numel(), dtype=torch.long, device=selected.device)
    score_inf = None
    if score_vec is not None:
        score_inf = torch.full_like(score_vec, float('inf'))

    while True:
        conflict_mask = selected[edge_u] & selected[edge_v]
        if not torch.any(conflict_mask):
            break

        cu = edge_u[conflict_mask]
        cv = edge_v[conflict_mask]
        conflict_deg.zero_()
        ones = edge_ones[conflict_mask]
        conflict_deg.scatter_add_(0, cu, ones)
        conflict_deg.scatter_add_(0, cv, ones)

        max_deg = conflict_deg.max()
        candidate_mask = (conflict_deg == max_deg)

        if score_vec is None:
            remove_node = n_nodes - 1 - torch.argmax(torch.flip(candidate_mask.to(torch.int8), dims=[0]))
        else:
            masked_scores = torch.where(candidate_mask, score_vec, score_inf)
            remove_node = torch.argmin(masked_scores)

        selected[remove_node] = False

    repaired = torch.zeros(n_nodes, dtype=original_dtype, device=original_device)
    repaired[selected] = 1.0
    return repaired


# Repair a bitstring into a valid independent set.
def repair_to_independent_set(bitstring, graph, scores=None):
    """
    Repair a candidate binary solution into a valid independent set:
    repeatedly remove the highest-conflict node, breaking ties by lower score.
    """
    original_device = bitstring.device
    original_dtype = bitstring.dtype

    bs = bitstring.detach().view(-1)
    n_nodes = bs.numel()
    selected = set(torch.nonzero(bs > 0.5, as_tuple=False).view(-1).cpu().numpy().tolist())

    if not selected:
        return bitstring.detach().clone()

    score_arr = None
    if scores is not None:
        score_arr = scores.detach().view(-1).cpu().numpy()

    while True:
        conflict_deg = {}
        for u, v in graph.edges():
            if u in selected and v in selected:
                conflict_deg[u] = conflict_deg.get(u, 0) + 1
                conflict_deg[v] = conflict_deg.get(v, 0) + 1

        if not conflict_deg:
            break

        max_deg = max(conflict_deg.values())
        candidates = [node for node, deg in conflict_deg.items() if deg == max_deg]

        if score_arr is None:
            remove_node = max(candidates)
        else:
            remove_node = min(candidates, key=lambda node: (score_arr[node], node))

        selected.remove(remove_node)

    repaired = torch.zeros(n_nodes, dtype=original_dtype, device=original_device)
    if selected:
        idx = torch.tensor(sorted(selected), dtype=torch.long, device=original_device)
        repaired[idx] = 1.0
    return repaired
