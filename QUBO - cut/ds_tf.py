import hashlib
import os
import random
from importlib import import_module
from typing import List
from train_gnn import DATASET,Config


from sqlalchemy.orm import Session
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, ChebConv, GINConv, ARMAConv


from ea_code_tf import gene_to_code
from graph_seq import gene_graph_seq_with_info
from init_ea import Code, CodeFile, GeneExpressed, Hash, Length, Main, ResFile, Runtime, create_db_engine, Score, Acc,Generation
from train_gnn import train
from train_gnn import DATASET


import time
import csv
import torch_geometric
import networkx as nx
import dgl
import numpy as np
import scipy.sparse
from itertools import chain
from co_corefunc import loss_func as qubo_loss_func
from co_corefunc import create_Q_matrix
from train_gnn import get_edge_index


# Set random seeds for reproducibility.
def set_seed(seed=666):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


class GNNCell(nn.Module):
    @staticmethod
    def get_gnn_params():
        gnn_types = [1, 2, 3, 4, 5, 6, 7]
        activation_list = [None, "relu", "elu", "leaky_relu"]
        return gnn_types, activation_list

    def __init__(self, in_channels: int, hidden_dim: int, heads: int, dropout_prob: float, init_weight: List[float]):
        super(GNNCell, self).__init__()

        self.weight = nn.Parameter(torch.tensor(init_weight, dtype=torch.float32), requires_grad=True)
        self.gnn_types, self.activation_list = GNNCell.get_gnn_params()

        self.gnns = self._build_gnn_layers(in_channels, hidden_dim, heads, dropout_prob)

    def _build_gnn_layers(self, in_channels, hidden_dim, heads, dropout_prob):
        """Build all candidate GNN layers for one differentiable-search cell."""
        gnns = []
        for gnn_type in self.gnn_types:
            if gnn_type == 1:
                gnn = GCNConv(in_channels, hidden_dim)
            elif gnn_type == 2:
                gnn = GATConv(in_channels, hidden_dim, heads=heads, concat=False,dropout=dropout_prob)
            elif gnn_type == 3:
                gnn = SAGEConv(in_channels, hidden_dim)
            elif gnn_type == 4:
                gnn = GATv2Conv(in_channels, hidden_dim, heads=heads, concat=False, dropout=dropout_prob)
            elif gnn_type == 5:
                K = 2 + (heads % 3)
                gnn = ChebConv(in_channels, hidden_dim, K=K)
            elif gnn_type == 6:
                mlp_mul = 1 + (heads % 3)
                mlp_hidden = hidden_dim * mlp_mul
                mlp = nn.Sequential(
                    nn.Linear(in_channels, mlp_hidden),
                    nn.ReLU(),
                    nn.Linear(mlp_hidden, hidden_dim),
                )
                gnn = GINConv(mlp)
            elif gnn_type == 7:
                arma_stacks = 1 + (heads % 3)
                arma_layers = 2
                gnn = ARMAConv(in_channels, hidden_dim, num_stacks=arma_stacks, num_layers=arma_layers, dropout=dropout_prob)
            else:
                raise ValueError(f"Unsupported GNN type: {gnn_type}")
            gnns.append(gnn)
        return nn.ModuleList(gnns)

    def forward(self, x, edge_index):
        """Apply the weighted mixture of candidate GNN layers."""
        weight = F.softmax(self.weight, dim=0)
        y = self.gnns[0](x, edge_index) * weight[0]

        for i in range(1, len(self.gnns)):
            y += self.gnns[i](x, edge_index) * weight[i]
        return y

    def get_weight_list(self):
        """Return the normalized operation weights as a Python list."""
        weight = F.softmax(self.weight, dim=0)
        return weight.detach().cpu().numpy().tolist()

    def get_weight_penalty(self):
        """Compute a concentration penalty for operation weights."""
        weight = F.softmax(self.weight, dim=0)

        return torch.sum(weight ** 2)


# Differentiable-search model built from a gene-space description.
class DSModel(nn.Module):
    def __init__(self, ds_describe: str, init_gene: str, weight_a: float = 0.445, num_classes: int = DATASET.out_channels):
        super(DSModel, self).__init__()
        self.ds_describe_items = [[int(i) for i in item.split(",")] for item in ds_describe.split("-")[:-1]]
        self.init_gene_params = [[int(i) for i in item.split(",")] for item in init_gene.split("-")[:-1]]
        self.cells = nn.ModuleDict()
        self.num_classes = num_classes

        self.out_layer = None
        self.weight_a = weight_a

        self.gnn_types, self.activation_list = GNNCell.get_gnn_params()
        self.l_gnn = len(self.gnn_types)
        self.l_act = len(self.activation_list)

        self.gene_list = [init_gene]
        for i in range(1, len(self.init_gene_params)):
            param = self.init_gene_params[i]
            if param[0] == 1:
                pre = "-".join([",".join([str(i) for i in p]) for p in self.init_gene_params[:i]])
                las = "-".join([",".join([str(i) for i in p]) for p in self.init_gene_params[i + 1:]] + ["255"])
                mod = [i for i in param]

                for gnn_type in range(self.l_gnn):
                    for act_idx in range(self.l_act):
                        mod[2] = self.gnn_types[gnn_type]
                        mod[6] = act_idx
                        mods = ",".join([str(i) for i in mod])
                        gene = "-".join([pre, mods, las])
                        self.gene_list.append(gene)

        self._build_cells()

    def _build_cells(self):
        """Create differentiable-search cells and track channel dimensions."""
        channel_record = []
        channel_record.append((0, DATASET.in_channels))
        for i in range(1, len(self.ds_describe_items)):
            item = self.ds_describe_items[i]
            param = self.init_gene_params[i]
            if item[0] == 1:
                gnn_type = param[2]

                input_offset = param[1]
                input_pos = max(0, i - input_offset)

                in_channels = next(ch for pos, ch in channel_record if pos == input_pos)

                if gnn_type in [8, 9]:
                    param[3] = in_channels
                    channel_record.append((i, in_channels))
                    continue

                gnn_idx = self.gnn_types.index(gnn_type)
                p_index = gnn_idx
                weight_a = max(0, min(1, self.weight_a))
                init_weight = [(1 - weight_a) / self.l_gnn for _ in range(self.l_gnn)]

                init_weight[p_index] = 1 / self.l_gnn + (self.l_gnn - 1) / self.l_gnn * weight_a

                hidden_dim = param[3]

                heads = param[4]
                dropout_prob = param[5] / 100.0
                channel_record.append((i, hidden_dim))

                self.cells[str(i)] = GNNCell(
                    in_channels = in_channels,
                    hidden_dim = hidden_dim,
                    heads = heads,
                    dropout_prob=dropout_prob,
                    init_weight=init_weight
                )
            else:
                input_offset_a = param[1]
                input_offset_b = param[2]
                input_pos_a = max(0, i - input_offset_a)
                input_pos_b = max(0, i - input_offset_b)
                in_channels_a = next(ch for pos, ch in channel_record if pos == input_pos_a)
                in_channels_b = next(ch for pos, ch in channel_record if pos == input_pos_b)

                in_channels = in_channels_a if in_channels_a == in_channels_b else in_channels_a
                if param[3] == 2:
                    channel_record.append((i, in_channels))
                elif param[3] == 3:
                    channel_record.append((i, in_channels_a + in_channels_b))

    def forward(self, x, edge_index):
        """Run the differentiable-search architecture on graph features."""
        mid = {0: x}
        for i in range(1, len(self.ds_describe_items)):
            item = self.ds_describe_items[i]
            if item[0] == 1:
                offset = item[1]
                in_pos = max(i - offset, 0)
                param = self.init_gene_params[i]
                gnn_type = param[2]
                act_idx = param[6]
                if gnn_type in [8, 9]:
                    mid[i] = mid[in_pos]
                else:
                    cell_output = self.cells[str(i)](mid[in_pos], edge_index)

                    if act_idx == 1:
                        mid[i] = F.relu(cell_output)
                    elif act_idx == 2:
                        mid[i] = F.elu(cell_output)
                    elif act_idx == 3:
                        mid[i] = F.leaky_relu(cell_output)
                    else:
                        mid[i] = cell_output
                    dropout_prob = param[5] / 100.0

                    if i != len(self.ds_describe_items) - 1:
                        if dropout_prob > 0:
                            mid[i] = F.dropout(mid[i], p=dropout_prob, training=self.training)
            else:
                offset = (item[1], item[2])
                in_pos = [max(i - off, 0) for off in offset]
                if item[3] == 2:
                    mid[i] = mid[in_pos[0]] + mid[in_pos[1]]
                elif item[3] == 3:
                    mid[i] = torch.cat([mid[in_pos[0]], mid[in_pos[1]]], dim=1)
                else:
                    raise ValueError(f"Unknown merge op m_k={item[3]} at node {i}")

        final_feat = mid[len(self.ds_describe_items) - 1]
        if self.out_layer is None:
            self.out_layer = nn.Linear(final_feat.shape[1], self.num_classes).to(final_feat.device)

        return torch.sigmoid(self.out_layer(final_feat))

    def get_result_gene(self):
        """Export the highest-weight architecture choices as a gene string."""
        params = ["0"]
        for i in range(1, len(self.init_gene_params)):
            param = self.init_gene_params[i]
            mod = [i for i in param]
            if param[0] == 1:
                gnn_type = param[2]
                if gnn_type in [8, 9]:
                    params.append(",".join(map(str, mod)))
                    continue

                cell = self.cells[str(i)]
                weight = cell.get_weight_list()
                wi = weight.index(max(weight))

                gnn_idx = wi
                mod[2] = self.gnn_types[gnn_idx]

                params.append(",".join(map(str, mod)))
            else:
                params.append(",".join(map(str, mod)))
        params.append("255")
        return "-".join(params)

    def get_cell_penalty(self):
        """Average the operation-weight penalties across all searchable cells."""
        if not self.cells:
            return 0.0
        return torch.mean(torch.stack([cell.get_weight_penalty() for cell in self.cells.values()]))

    def print_weight_dist(self):
        """Print operation-weight distributions for each searchable cell."""
        for k, cell in self.cells.items():
            print(f"Cell {k} weights: {cell.get_weight_list()}")

    def get_weight_slight(self):
        """Return compact indices of the strongest operation weights."""
        res = []
        for cell in self.cells.values():
            w = cell.get_weight_list()
            res.append(str(w.index(max(w))))
        return ",".join(res)


# Train a differentiable-search model and evaluate the resulting gene.
def search(gpu_idx, iteration, ds_describe, init_id, init_gene, init_score, ckpt_dir, save_dir,log_dir, epochs_ds):
    if gpu_idx >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{gpu_idx}")
        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    ds_ckpt_dir = ckpt_dir + '/ds'
    ds_save_dir = save_dir + '/ds'
    ds_log_dir = log_dir + '/ds'
    os.makedirs(ds_ckpt_dir, exist_ok=True)
    os.makedirs(ds_save_dir, exist_ok=True)
    os.makedirs(ds_log_dir, exist_ok=True)

    final_gene = ds_train(ds_describe, init_gene, ds_ckpt_dir, ds_save_dir, ds_log_dir, epochs_ds)

    if final_gene == init_gene:
        return -1, -1

    re_save_dir = save_dir + '/re'
    re_log_dir = log_dir + '/re'
    os.makedirs(re_save_dir, exist_ok=True)
    os.makedirs(re_log_dir, exist_ok=True)

    gene_units = [item.split(",") for item in final_gene.split("-")]
    code, gene_length = gene_to_code(gene_units)
    file_head = '''import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, GATv2Conv, ChebConv, GINConv, ARMAConv'''
    file_content = file_head + "\n\n\n" + code

    os.makedirs("./ds_model", exist_ok=True)
    with open("./ds_model/model_{}.py".format(init_id), "w") as f:
        f.write(file_content)

    model_class = getattr(import_module(f'ds_model.model_{init_id}'), "GEN_GNN_MODEL")

    final_model_dirname,best_cut, best_acc,runtime = train(model_class=model_class,save_dir=re_save_dir)

    new_score = best_acc
    print(f"best_cut: {best_cut},best_acc: {best_acc}")

    if new_score <= init_score:
        return -1, -3

    engine = create_db_engine()
    with Session(engine) as session:
        row = session.execute("select * from main order by id desc limit 1").first()

        if row is None:
            new_id = 1
        else:
            new_id = row[0] + 1

        row = Main(id=new_id, gene=final_gene)
        session.merge(row)
        row = Acc(id=new_id, accuracy=best_acc,cut=best_cut)
        session.merge(row)
        row = Runtime(id=new_id, runtime=runtime)
        session.merge(row)
        row = Score(id=new_id, score=new_score)
        session.merge(row)
        row = Generation(id=new_id, father=init_id, mother=init_id, iteration=iteration)
        session.merge(row)
        row = Length(id=new_id, length=gene_length, gene_code_length=gene_length)
        session.merge(row)
        row = Code(id=new_id, code=code)
        session.merge(row)
        row = CodeFile(id=new_id, file=f"./ds_model/model_{init_id}.py", ckpt_dir=ckpt_dir, save_dir=save_dir,
                       log_dir=log_dir)
        session.merge(row)
        row = ResFile(id=new_id, save_name=final_model_dirname, save_dir=re_save_dir)
        session.merge(row)
        row = GeneExpressed(id=new_id, gene_expressed=final_gene)
        session.merge(row)

        gene_units = final_gene.split("-")[1:-1]
        seq, info = gene_graph_seq_with_info(gene_units)
        info_s = '-'.join(info)
        hash_val = hashlib.sha1((seq + '|' + info_s).encode(encoding='utf-8')).hexdigest()
        row = Hash(id=new_id, hash=hash_val, seq_info=seq + '|' + info_s)
        session.merge(row)
        session.commit()
    engine.dispose()
    return new_id, new_score


# Train DSModel, optimize architecture weights, and return the best gene sequence.
def ds_train(ds_describe, init_gene, ckpt_dir, save_dir, log_dir, epochs, prob_threshold=0.5,device=None):
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    best_cut = -np.inf
    best_gene = init_gene

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data_path = "./data/G14.txt"
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
    embed = nn.Embedding(n_nodes, dim_embedding).type(torch.float32).to(device)

    model = DSModel(
        ds_describe=ds_describe,
        init_gene=init_gene
    ).to(device)

    params = chain(model.parameters(), embed.parameters())
    optimizer = torch.optim.Adam(
        params,
        lr=Config.lr_ds,
        weight_decay=Config.weight_decay
    )

    def scan_best_threshold(probs):
        probs_det = probs.detach()

        thresholds = torch.linspace(
            start=0.0,
            end=1.0,
            steps=int(1.0 / 0.01) + 1,
            device=probs.device
        )
        threshold_bitstrings = (probs_det.unsqueeze(0) >= thresholds.unsqueeze(1)).float()
        threshold_cuts = -((threshold_bitstrings @ Q) * threshold_bitstrings).sum(dim=1)
        best_idx = int(torch.argmax(threshold_cuts).item())
        best_cut = float(threshold_cuts[best_idx].item())
        best_th = float(thresholds[best_idx].item())
        best_bs = threshold_bitstrings[best_idx]

        num_samples = 256
        samples = torch.bernoulli(probs_det.expand(num_samples, -1)).float()
        sample_cuts = -((samples @ Q) * samples).sum(dim=1)
        sample_best_idx = int(torch.argmax(sample_cuts).item())
        sample_best_cut = float(sample_cuts[sample_best_idx].item())
        if sample_best_cut > best_cut:
            best_cut = sample_best_cut
            best_bs = samples[sample_best_idx]

        return best_bs, best_cut, best_th

    inputs = embed.weight
    data = torch_geometric.data.Data(x=inputs, edge_index=edge_index)
    prev_loss = np.inf
    count = 0
    patience = Config.patience

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()

        probs = model(data.x, data.edge_index)[:, 0]

        loss = qubo_loss_func(probs, Q)

        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            if (epoch % 100 == 0) or (epoch == 0):
                current_bitstring, current_cut, last_best_threshold = scan_best_threshold(probs)
            else:
                current_bitstring = (probs.detach() >= last_best_threshold).float()
                bitstring_row = current_bitstring.unsqueeze(0)
                current_cut = -(bitstring_row @ Q @ bitstring_row.T).item()

        if current_cut > best_cut:
            best_cut = current_cut
            best_gene = model.get_result_gene()

            torch.save({
                'model_state_dict': model.state_dict(),
                'embed_state_dict': embed.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_cut': best_cut
            }, os.path.join(ckpt_dir, "best_ds_model.pt"))
            print(f"Updated best gene (cut: {best_cut:.4f}) | Gene: {best_gene}")

        if loss.item() >= prev_loss - Config.tol:
            count += 1
        else:
            count = 0
        if count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
        prev_loss = loss.item()

        if epoch % 100 == 0:
            acc = best_cut / all_edges
            print(f"Epoch {epoch} | Loss: {loss.item():.6f} | Best Cut: {best_cut:.4f} | Acc: {acc:.4f}")

    acc = best_cut / all_edges
    print(f"Training finished | Best Cut: {best_cut:.4f} | Accuracy: {acc:.4f}")

    with open(os.path.join(log_dir, "experiment.txt"), "a") as f:
        f.write(f"DSModel     {best_cut}     {acc:.4f}\n")

    return best_gene
