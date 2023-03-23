import os.path as osp
from collections import OrderedDict
import numpy as np
import torch
from torch import Tensor
import torch_geometric.transforms as transforms
from torch_geometric.datasets import TUDataset
from torch_geometric.nn import Sequential, Linear
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_dense_batch, to_dense_adj
from sklearn.model_selection import StratifiedKFold, train_test_split
from GTVConv import GTVConv
from AsymCheegerCutPool import AsymCheegerCutPool


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
# CONFIG
################################
mp_layers = 1
mp_channels = 32
mp_activation = "relu"
delta_coeff = 2.0

mlp_hidden_layers = 1
mlp_hidden_channels = 32
mlp_activation = "relu"
totvar_coeff = 0.5
balance_coeff = 0.5

epochs = 100
batch_size = 16
learning_rate = 5e-4
l2_reg_val = 0
patience = 10
 
results = {"acc_scores": []}

################################
# LOAD DATASET
################################
dataset_id = "PROTEINS"

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_id)
dataset = TUDataset(path, "PROTEINS", use_node_attr=True, cleaned=True)

# Parameters
N = max(graph.num_nodes for graph in dataset)
n_out = dataset.num_classes # Dimension of target

# Train/test split
idxs = np.random.permutation(len(dataset))
split_va, split_te = int(0.8 * len(dataset)), int(0.9 * len(dataset))
idx_tr, idx_va, idx_te = np.split(idxs, [split_va, split_te])
dataset_tr = dataset[torch.tensor(idx_tr).long()]
dataset_va = dataset[torch.tensor(idx_va).long()]
dataset_te = dataset[torch.tensor(idx_te).long()]
loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)
loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)

################################
# MODEL
################################
class ClassificationModel(torch.nn.Module):
    def __init__(self, n_out, mp1, pool1, mp2, pool2, mp3):
        super().__init__()

        self.mp1 = mp1
        self.pool1 = pool1
        self.mp2 = mp2
        self.pool2 = pool2
        self.mp3 = mp3
        self.output_layer = Linear(mp_channels, n_out)


    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor):
        
        # 1st block
        x = self.mp1(x, edge_index, edge_weight)
        x, mask = to_dense_batch(x, batch)
        adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)
        x, adj, tv1, bal1 = self.pool1(x, adj, mask=mask)

        # 2nd block
        x = self.mp2(x, edge_index=adj, edge_weight=None)
        x, adj, tv2, bal2 = self.pool2(x, adj)

        # 3rd block
        x = self.mp3(x, edge_index=adj, edge_weight=None)
        x = x.mean(dim=1) # global mean pooling
        x = self.output_layer(x)

        return x, tv1 + tv2, bal1 + bal2


MP1 = [
    (GTVConv(dataset.num_features if i==0 else mp_channels,
            mp_channels,
            act=mp_activation,
            delta_coeff=delta_coeff),
        'x, edge_index, edge_weight -> x')
    for i in range(mp_layers)]

MP1 = Sequential('x, edge_index, edge_weight', MP1)


Pool1 = AsymCheegerCutPool(int(N//2),
                           mlp_channels=[mp_channels] + 
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)


MP2 = [
    (GTVConv(mp_channels,
            mp_channels,
            act=mp_activation,
            delta_coeff=delta_coeff),
        'x, edge_index, edge_weight -> x')
    for _ in range(mp_layers)]

MP2 = Sequential('x, edge_index, edge_weight', MP2)


Pool2 = AsymCheegerCutPool(int(N//4),
                           mlp_channels=[mp_channels] + 
                                [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                           mlp_activation=mlp_activation,
                           totvar_coeff=totvar_coeff,
                           balance_coeff=balance_coeff,
                           return_selection=False,
                           return_pooled_graph=True)


MP3 = [
    (GTVConv(mp_channels,
            mp_channels,
            act=mp_activation,
            delta_coeff=delta_coeff),
        'x, edge_index, edge_weight -> x')
    for _ in range(mp_layers)]

MP3 = Sequential('x, edge_index, edge_weight', MP3)


# Setup model
model = ClassificationModel(n_out,
                            mp1=MP1,
                            pool1=Pool1,
                            mp2=MP2,
                            pool2=Pool2,
                            mp3=MP3).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()

        
################################
# TRAIN AND TEST
################################

def train():
    model.train()

    for data in loader_tr:
        data.to(device)
        out, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = tv_loss + bal_loss
        loss += loss_fn(out, data.y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

@torch.no_grad()       
def test(loader):
    model.eval()

    correct = 0
    for data in loader:
        data.to(device)
        out, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight, data.batch)
        loss = tv_loss + bal_loss + loss_fn(out, data.y)
        pred = out.argmax(dim=1)
        correct += int((pred == data.y).sum())
    return loss, correct / len(loader.dataset)


best_val_acc = 0
patience_count = patience
for epoch in range(1, epochs + 1):
    train()
    train_loss, train_acc = test(loader_tr)
    val_loss, val_acc = test(loader_va)
    test_loss, test_acc = test(loader_te)

    print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc: .4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        test_loss_at_best_val = test_loss
        test_acc_at_best_val = test_acc
        patience_count = patience
    else:
        patience_count -= 1
    if patience_count == 0:
        break

print("Test loss: {}. Test acc: {}".format())
            
