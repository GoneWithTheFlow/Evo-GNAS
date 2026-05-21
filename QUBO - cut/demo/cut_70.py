import time
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, GATConv,GATv2Conv,GINConv, SAGEConv, ChebConv, ARMAConv, GraphConv,GraphNorm
import networkx as nx
import jax.numpy as jnp
from networkx import convert_node_labels_to_integers
import dgl
import numpy as np
import scipy.sparse
import sys
from pathlib import Path
from itertools import chain


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qubo_utils import loss_func
from qubo_utils import create_Q_matrix


# Runtime and database configuration.
class Config:
    epochs = 30000
    lr = 0.005
    epochs_ds = 20000
    lr_ds = 0.005
    out = 100
    IterNum = 1
    patience = 2000
    weight_decay = 5e-4
    close_frac = 0.95
    best_known = {
        "G14": 3064,
        "G15": 3050,
        "G22": 13359,
        "G49": 6000,
        "G50": 5880,
        "G55": 10294,
        "G70": 9541,
    }
    current_dataset = "G70"


# Dataset shape configuration.
class DATASET:
    in_channels = 64
    out_channels  = 1

dtype = torch.float32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
data_path = r"C:\Users\XinLiu\Desktop\QUBO\data\G70.txt"


# Generated GNN architecture.
class GEN_GNN_MODEL(torch.nn.Module):
    def __init__(self):
        super(GEN_GNN_MODEL, self).__init__()
        self.gnn_1 = SAGEConv(in_channels=64, out_channels=1)
        self.gnn_2 = GCNConv(in_channels=1, out_channels=64)
        self.gnn_3 = GCNConv(in_channels=64, out_channels=1)
        self.gnn_5 = GCNConv(in_channels=2, out_channels=1)
        self.gnn_6 = ChebConv(in_channels=1, out_channels=29, K=4)
        self.gnn_7 = ChebConv(in_channels=29, out_channels=29, K=4)
        self.gnn_8 = ChebConv(in_channels=29, out_channels=1, K=4)

    def forward(self, x_0, edge_index):
        x_1 = self.gnn_1(x_0, edge_index)
        x_1 = F.dropout(x_1, p=0.0, training=self.training)
        x_2 = F.relu(self.gnn_2(x_1, edge_index))
        x_2 = F.dropout(x_2, p=0.4, training=self.training)
        x_3 = F.relu(self.gnn_3(x_2, edge_index))
        x_3 = F.dropout(x_3, p=0.4, training=self.training)
        x_4 = torch.cat([x_3, x_1], dim=1)
        x_5 = F.elu(self.gnn_5(x_4, edge_index))
        x_5 = F.dropout(x_5, p=0.4, training=self.training)
        x_6 = F.elu(self.gnn_6(x_5, edge_index))
        x_6 = F.dropout(x_6, p=0.4, training=self.training)
        x_7 = F.elu(self.gnn_7(x_6, edge_index))
        x_7 = F.dropout(x_7, p=0.5, training=self.training)
        x_8 = F.elu(self.gnn_8(x_7, edge_index))
        return torch.sigmoid(x_8)


# Prepare GNN, node embeddings, optimizer, and device placement before training.
def get_edge_index(allRows):
    G = nx.from_edgelist(allRows)
    G = convert_node_labels_to_integers(G, first_label=0, ordering="default", label_attribute=None)

    H = nx.Graph()
    H.add_nodes_from(sorted(G.nodes(data=True)))
    H.add_edges_from(G.edges(data=True))

    graph_dgl = dgl.from_networkx(nx_graph=H)
    graph_dgl = graph_dgl.to(device)

    n_nodes = len(G.nodes())
    adj_matrix = nx.adjacency_matrix(G)
    coo_matrix = scipy.sparse.coo_matrix(adj_matrix)

    indices = np.vstack((coo_matrix.row, coo_matrix.col))
    edge_index_A = torch.LongTensor(indices)
    edge_index_A = edge_index_A.to(device)

    return edge_index_A, graph_dgl, G, n_nodes


# Initialize GNN parameters: embedding, network, and optimizer.
# Prepare GNN, node embeddings, optimizer, and device placement before training.
def get_gnn_params(in_features, n_nodes, MyGraphNetwork):
    net = MyGraphNetwork()
    dim_embedding = in_features
    net = net.type(dtype).to(device)
    embed = nn.Embedding(n_nodes, dim_embedding)
    embed = embed.type(dtype).to(device)

    params = chain(net.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(params, lr=Config.lr, weight_decay=5e-4)

    return net, embed, optimizer


# Load graph data and convert edge lists into graph formats.
def run_gnn_training_qubo(
        embed,
        dgl_graph,
        q_torch,
        net,
        optimizer,
        edge_index,
        prob_threshold_scan_step=0.01,
        scan_frequency=100,
        prob_threshold=0.5
):
    inputs = embed.weight
    data = torch_geometric.data.Data(x=inputs, edge_index=edge_index)
    prev_loss = 1.0
    count = 0
    patience = Config.patience
    tol = 1e-6
    number_epochs = Config.epochs
    out = Config.out

    losses = []
    epochs = []

    best_bitstring = torch.zeros((dgl_graph.number_of_nodes(),), dtype=dtype).to(device)
    best_loss = loss_func(best_bitstring.float(), q_torch)
    best_cut = -((best_bitstring.unsqueeze(0) @ q_torch @ best_bitstring.unsqueeze(0).T).item())
    last_best_threshold = prob_threshold

    best_known_cut = Config.best_known[Config.current_dataset]
    close_threshold = Config.close_frac * best_known_cut
    early_stopping_enabled = True

    def scan_best_threshold(probs):
        """
        Scan candidate thresholds and find the best bitstring, cut value, and threshold.
        probs: node assignment probabilities from the GNN, shape [num_nodes].
        Return: best bitstring, best cut value, and best threshold.
        """
        probs_det = probs.detach()

        thresholds = torch.linspace(
            start=0.0,
            end=1.0,
            steps=int(1.0 / prob_threshold_scan_step) + 1,
            device=probs.device
        )

        threshold_bitstrings = (probs_det.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
        threshold_cuts = -((threshold_bitstrings @ q_torch) * threshold_bitstrings).sum(dim=1)
        best_idx = int(torch.argmax(threshold_cuts).item())
        best_cut = float(threshold_cuts[best_idx].item())
        best_th = float(thresholds[best_idx].item())
        best_bs = threshold_bitstrings[best_idx]

        num_samples = 2048

        samples = torch.bernoulli(probs_det.expand(num_samples, -1)).float()

        sample_cuts = -((samples @ q_torch) * samples).sum(dim=1)
        sample_best_idx = int(torch.argmax(sample_cuts).item())
        sample_best_cut = float(sample_cuts[sample_best_idx].item())

        if sample_best_cut > best_cut:
            best_cut = sample_best_cut
            best_bs = samples[sample_best_idx]

        return best_bs, best_cut, best_th

    t_gnn_start = time.time()
    for epoch in range(number_epochs):
        net.train()
        optimizer.zero_grad()

        probs = net(data.x, data.edge_index)[:, 0]

        if (epoch % scan_frequency == 0) or (epoch == 0):
            current_bitstring, current_cut, last_best_threshold = scan_best_threshold(probs)
            current_loss = -current_cut
        else:
            current_bitstring = (probs.detach() >= last_best_threshold).float()

            bitstring_row = current_bitstring.unsqueeze(0)
            current_cut = -(bitstring_row @ q_torch @ bitstring_row.T).item()
            current_loss = -current_cut

        loss = loss_func(probs, q_torch)
        loss_val = loss.item()

        loss.backward()
        optimizer.step()

        if current_loss < best_loss:
            best_loss = current_loss
            best_cut = current_cut
            best_bitstring = current_bitstring
            count = 0

            if best_cut >= close_threshold and early_stopping_enabled:
                early_stopping_enabled = False
                print(
                    f"Reached {Config.close_frac * 100:.1f}% of SOTA ({best_cut:.0f}/{best_known_cut}), disabling early stopping")
        else:
            count += 1

        if early_stopping_enabled and count >= patience:
            print(f'Early stopping at epoch {epoch} (no best-solution improvement for {count} consecutive epochs)')
            break

        if epoch % out == 0:
            print(f'Epoch: {epoch}, Loss: {loss_val:.6f}, best loss: {best_loss:.6f}')
            losses.append(loss_val)
            epochs.append(epoch)

    t_gnn = time.time() - t_gnn_start

    final_bitstring, final_cut, _ = scan_best_threshold(probs)
    if final_cut > best_cut:
        best_cut = final_cut
        best_bitstring = final_bitstring
        best_loss = -best_cut

    final_bitstring = best_bitstring.clone()
    print(f'Training took {t_gnn:.3f}s')
    print(f'Best loss: {best_loss:.6f}')

    return net, epoch, final_bitstring, best_bitstring, losses, epochs, t_gnn


# Load graph data and convert edge lists into graph formats.
def evaluate_gen_model(model_class=None):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        allRows = [list(map(int, row[0].split())) for row in reader]
    allRows = allRows[1:]
    for i in range(len(allRows)):
        del allRows[i][2]
    allRows = list(map(tuple, allRows))

    edge_index, graph_dgl, G, n_nodes = get_edge_index(allRows)
    all_edges = G.number_of_edges()

    Q = create_Q_matrix(G, is_max_cut=True).to(device)

    dim_embedding = DATASET.in_channels
    in_features = dim_embedding
    IterNum = Config.IterNum

    net, embed, optimizer = get_gnn_params(
        in_features=in_features,
        n_nodes=n_nodes,
        MyGraphNetwork=model_class
    )

    cut_vals = []
    for i in range(IterNum):
        print(f'Running iteration {i + 1}/{IterNum}')

        net, embed, optimizer = get_gnn_params(
            in_features=in_features,
            n_nodes=n_nodes,
            MyGraphNetwork=model_class
        )

        net, epoch, final_bitstring, best_bitstring, losses, epochs,runtime = run_gnn_training_qubo(
            embed=embed,
            dgl_graph=graph_dgl,
            q_torch=Q,
            net=net,
            optimizer=optimizer,
            edge_index=edge_index,
        )

        best_bitstring = best_bitstring.to(device)

        best_bitstring_row = best_bitstring.unsqueeze(0)
        cut_value = -(best_bitstring_row @ Q @ best_bitstring_row.T).item()
        cut_vals.append(cut_value)

    best_cut = max(cut_vals)
    acc = best_cut / all_edges
    print(f'Best cut value: {best_cut}')
    print(f'Accuracy: {acc:.4f}')

    with open("experiment.txt", "a") as f:
        f.write(f"GEN_GNN_MODEL     {best_cut}     {acc:.4f}\n")

    return best_cut, acc


# Load graph data and convert edge lists into graph formats.
def load_model_and_evaluate(model_path, model_class):
    with open(data_path, 'r') as f:
        reader = csv.reader(f)
        allRows = [list(map(int, row[0].split())) for row in reader]
    allRows = allRows[1:]
    for i in range(len(allRows)):
        del allRows[i][2]
    allRows = list(map(tuple, allRows))

    edge_index, graph_dgl, G, n_nodes = get_edge_index(allRows)
    all_edges = G.number_of_edges()

    Q = create_Q_matrix(G, is_max_cut=True).to(device)

    dim_embedding = DATASET.in_channels
    in_features = dim_embedding
    net, embed, _ = get_gnn_params(
        in_features=in_features,
        n_nodes=n_nodes,
        MyGraphNetwork=model_class
    )

    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location=device)
    net.load_state_dict(checkpoint['net'])
    embed.load_state_dict(checkpoint['embed'])

    net.eval()
    with torch.no_grad():
        inputs = embed.weight
        data = torch_geometric.data.Data(x=inputs, edge_index=edge_index)
        probs = net(data.x, data.edge_index)[:, 0]

        prob_threshold = torch.mean(probs.detach())
        best_bitstring = (probs >= prob_threshold).float()

        best_bitstring_row = best_bitstring.unsqueeze(0)
        cut_value = -(best_bitstring_row @ Q @ best_bitstring_row.T).item()

    print(f"Best cut value: {cut_value}")
    acc = cut_value / all_edges
    print(f"Accuracy: {acc:.4f}")

    return cut_value, acc

if __name__ == "__main__":
    evaluate_gen_model(model_class=GEN_GNN_MODEL)
