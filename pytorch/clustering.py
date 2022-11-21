import sys
sys.path.insert(1, '../utils')

import os.path as osp
import torch
from torch import Tensor
import torch_geometric.transforms as transforms
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric import utils
from torch_geometric.nn import Sequential
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from metrics import cluster_acc
from GTVConv import GTVConv
from AsymCheegerCutPool import AsymCheegerCutPool

torch.manual_seed(1) 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

################################
# CONFIG
################################
dataset_id="Cora"
mp_channels=512
mp_layers=2
mp_activation="elu"
delta_coeff=0.311
mlp_hidden_channels=256
mlp_hidden_layers=1
mlp_activation="relu"
totvar_coeff=0.785
balance_coeff=0.514
learning_rate=1e-3
epochs=500

################################
# LOAD DATASET
################################
path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_id)
if dataset_id in ["Cora", "CiteSeer", "PubMed"]:
    dataset = Planetoid(path, dataset_id, transform=transforms.NormalizeFeatures())
elif dataset_id == "DBLP":
    dataset = CitationFull(path, dataset_id, transform=transforms.NormalizeFeatures())

data = dataset[0]
data = data.to(device)

############################################################################
# MODEL
############################################################################
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()

        # Message passing layers
        mp = [
            (GTVConv(dataset.num_features if i==0 else mp_channels,
                     mp_channels,
                     act=mp_activation,
                     delta_coeff=delta_coeff),
             'x, edge_index, edge_weight -> x')
        for i in range(mp_layers)]
        
        self.mp = Sequential('x, edge_index, edge_weight', mp)
        
        # Pooling layer
        self.pool = AsymCheegerCutPool(
            dataset.num_classes,
            mlp_channels=[mp_channels] + [mlp_hidden_channels for _ in range(mlp_hidden_layers-1)],
            mlp_activation=mlp_activation,
            totvar_coeff=totvar_coeff,
            balance_coeff=balance_coeff,
            return_selection=True,
            return_pooled_graph=False)

    def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor):

        # Propagate node features
        x = self.mp(x, edge_index, edge_weight)
        
        # Compute cluster assignment and obtain auxiliary losses
        adj = utils.to_dense_adj(edge_index, edge_attr=edge_weight)
        s, tv_loss, bal_loss = self.pool(x, adj)

        return s.squeeze(0), tv_loss, bal_loss

model = Net().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

############################################################################
# TRAINING
############################################################################
def train():
    model.train()
    optimizer.zero_grad()
    _, tv_loss, bal_loss = model(data.x, data.edge_index, data.edge_weight)
    loss = tv_loss + bal_loss
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test():
    model.eval()
    clust, _, _ = model(data.x, data.edge_index, data.edge_weight)
    return NMI(data.y.cpu(), clust.max(1)[1].cpu()), cluster_acc(data.y.cpu().numpy(), clust.max(1)[1].cpu().numpy())[0]

patience = 50
best_loss = 1
nmi_at_best_loss = 0
acc_at_best_loss = 0
for epoch in range(1, epochs+1):
    train_loss = train()
    nmi, acc = test()
    print(f"Epoch: {epoch:03d}, Loss: {train_loss:.4f}, NMI: {nmi:.3f}, ACC: {acc*100: .3f}")
    if train_loss < best_loss:
        best_loss = train_loss
        nmi_at_best_loss = nmi
        acc_at_best_loss = acc
        patience = 50
    else:
        patience -= 1
    if patience == 0:
        break

print(f"NMI: {nmi_at_best_loss:.3f}, ACC: {acc_at_best_loss*100:.1f}")
