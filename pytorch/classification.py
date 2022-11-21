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

############################################################################
# CONFIG
############################################################################

dataset_id = "PROTEINS"

mp_layers = 1
mp_channels = 32
mp_activation = "relu"
delta_coeff = 2.0

mlp_hidden_layers = 1
mlp_hidden_channels = 32
mlp_activation = "relu"
totvar_coeff = 0.5
balance_coeff = 0.5

n_folds = 5
n_runs = 3
epochs = 100
batch_size = 16
learning_rate = 5e-4
l2_reg_val = 0
es_patience = 10
 
results = {"acc_scores": []}

############################################################################
# LOAD DATASET
############################################################################
print("Processing dataset: {}".format(dataset_id))

max_degrees = {"REDDIT-BINARY": 3062, "COLLAB": 491}
if dataset_id in list(max_degrees.keys()):
    pre_transform = transforms.OneHotDegree(max_degrees[dataset_id])
else:
    pre_transform = None

path = osp.join(osp.dirname(osp.realpath(__file__)), '.', 'data', dataset_id)
dataset = TUDataset(path, dataset_id, use_node_attr=True, cleaned=True, pre_transform=pre_transform)

# Graph parameters
N = max(graph.num_nodes for graph in dataset)
n_out = dataset.num_classes

# Apply statified K-folding
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=123)
full_label_set = np.vstack([dataset[i].y.numpy() for i in range(len(dataset))])

for train_index, test_index in skf.split(X=np.zeros(len(dataset)), y=full_label_set):
   
    # Split validation set from training set
    train_index, val_index = train_test_split(train_index, test_size=0.1,
                                              stratify=full_label_set[train_index],
                                              shuffle=True, random_state=123)
    
    dataset_tr = dataset[torch.tensor(train_index).long()]
    dataset_va = dataset[torch.tensor(val_index).long()]
    dataset_te = dataset[torch.tensor(test_index).long()]

    for run in range(n_runs):
        loader_tr = DataLoader(dataset_tr, batch_size=batch_size, shuffle=True)
        loader_va = DataLoader(dataset_va, batch_size=batch_size, shuffle=False)
        loader_te = DataLoader(dataset_te, batch_size=batch_size, shuffle=False)


        ############################################################################
        # MODEL
        ############################################################################

        class Net(torch.nn.Module):

            def __init__(self):
                super().__init__()

                # First message passing block
                mp1 = [
                    (GTVConv(dataset.num_features if i==0 else mp_channels,
                            mp_channels,
                            act=mp_activation,
                            delta_coeff=delta_coeff),
                        'x, edge_index, edge_weight -> x')
                for i in range(mp_layers)]
                
                self.mp1 = Sequential('x, edge_index, edge_weight', mp1)
                
                # First pooling layer
                self.pool1 = AsymCheegerCutPool(int(N//2),
                                                mlp_channels=[mp_channels] + [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                                                mlp_activation=mlp_activation,
                                                totvar_coeff=totvar_coeff,
                                                balance_coeff=balance_coeff,
                                                return_selection=False,
                                                return_pooled_graph=True)
                
                # Second message passing block
                mp2 = [
                    (GTVConv(mp_channels,
                            mp_channels,
                            act=mp_activation,
                            delta_coeff=delta_coeff),
                        'x, edge_index, edge_weight -> x')
                for _ in range(mp_layers)]

                self.mp2 = Sequential('x, edge_index, edge_weight', mp2)

                # Second pooling layer
                self.pool2 = AsymCheegerCutPool(int(N//4),
                                                mlp_channels=[mp_channels] + [mlp_hidden_channels for _ in range(mlp_hidden_layers)],
                                                mlp_activation=mlp_activation,
                                                totvar_coeff=totvar_coeff,
                                                balance_coeff=balance_coeff,
                                                return_selection=False,
                                                return_pooled_graph=True)

                # Third message passing block
                mp3 = [
                    (GTVConv(mp_channels,
                            mp_channels,
                            act=mp_activation,
                            delta_coeff=delta_coeff),
                        'x, edge_index, edge_weight -> x')
                for _ in range(mp_layers)]

                self.mp3 = Sequential('x, edge_index, edge_weight', mp3)

                self.output_layer = Linear(mp_channels, n_out)


            def forward(self, x: Tensor, edge_index: Tensor, edge_weight: Tensor, batch: Tensor):
                
                x = self.mp1(x, edge_index, edge_weight)

                x, mask = to_dense_batch(x, batch)
                adj = to_dense_adj(edge_index, edge_attr=edge_weight, batch=batch)
                
                x, adj, tv1, bal1 = self.pool1(x, adj, mask=mask)

                x = self.mp2(x, edge_index=adj, edge_weight=None)

                x, adj, tv2, bal2 = self.pool2(x, adj)

                x = self.mp3(x, edge_index=adj, edge_weight=None)

                x = x.mean(dim=1)

                x = self.output_layer(x)

                x = torch.softmax(x, -1)

                return x, tv1 + tv2, bal1 + bal2
        
        model = Net().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        
        ############################################################################
        # TRAINING
        ############################################################################
        loss_fn = torch.nn.CrossEntropyLoss()

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
        
        def test(loader):
            model.eval()

            correct = 0
            for data in loader:
                data.to(device)
                out, _, _ = model(data.x, data.edge_index, data.edge_weight, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
            return correct / len(loader.dataset)


        best_val_acc = 0
        patience = es_patience
        for epoch in range(1, epochs + 1):
            train()
            train_acc = test(loader_tr)
            val_acc = test(loader_va)
            test_acc = test(loader_te)

            print(f"Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}, Test Acc: {test_acc: .4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                test_acc_at_best_val = test_acc
                patience = es_patience
            else:
                patience -= 1
            if patience == 0:
                break
        
        ############################################################################
        # RESULTS
        ############################################################################
        results["acc_scores"].append(test_acc_at_best_val)

    print(f"Finished run {run+1}.")

print(f"ACC: {np.mean(results['acc_scores'])*100:.4f} +/- {np.std(results['acc_scores'])*100:.4f}")
            