from typing import List, Optional, Tuple, Union
import math
import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import Linear
from torch_geometric.nn.resolver import activation_resolver


class AsymCheegerCutPool(torch.nn.Module):
    r"""
    The asymmetric cheeger cut pooling layer from the `"Clustering with Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ paper

    Args:
        k (int): Number of clusters or output nodes
        mlp_channels (int, list of int): Number of hidden units for each hidden layer in
            the MLP used to compute cluster assignments. First integer must match the number of input channels 
            of input channels. 
        mlp_activation (any): Activation function between hidden layers of the MLP. Must be compatible
            with `torch_geometric.nn.resolver`.
        return_selection (bool): Whether to return selection matrix. Cannot not 
            be False if `return_pooled_graph` is False. (default: :obj:`False`)
        return_pooled_graph (bool): Whether to return pooled node features and 
            adjacency. Cannot be False if `return_selection` is False. (default: :obj:`True`)
        bias (bool): whether to add a bias term to the MLP layers. (default: :obj:`True`)
        totvar_coeff (float): Coefficient for graph total variation loss component. (default: :obj:`1.0`)
        balance_coeff (float): Coefficient for asymmetric norm loss component. (default: :obj:`1.0`)
    """

    def __init__(self, 
                 k: int, 
                 mlp_channels: Union[int, List[int]], 
                 mlp_activation="relu",
                 return_selection: bool = False,
                 return_pooled_graph: bool = True,
                 bias: bool = True,
                 totvar_coeff: float = 1.0,
                 balance_coeff: float = 1.0,
                 ):
        super().__init__()

        if not return_selection and not return_pooled_graph:
            raise ValueError("return_selection and return_pooled_graph can not both be False")

        if isinstance(mlp_channels, int):
            mlp_channels = [mlp_channels]

        act = activation_resolver(mlp_activation)
        in_channels = mlp_channels[0]
        self.mlp = torch.nn.Sequential()
        for channels in mlp_channels[1:]:
            self.mlp.append(Linear(in_channels, channels, bias=bias))
            in_channels = channels
            self.mlp.append(act)


        self.mlp.append(Linear(in_channels, k))
        self.k = k
        self.return_selection = return_selection
        self.return_pooled_graph = return_pooled_graph
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.mlp:
            if isinstance(layer, Linear):
                torch.nn.init.xavier_uniform(layer.weight)
                torch.nn.init.zeros_(layer.bias)

    def forward(
        self,
        x: Tensor,
        adj: Tensor,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        r"""
        Args:
            x (Tensor): Node feature tensor :math:`\mathbf{X} \in
                \mathbb{R}^{B \times N \times F}` with batch-size
                :math:`B`, (maximum) number of nodes :math:`N` for each graph,
                and feature dimension :math:`F`.
                Note that the cluster assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times C}` is
                being created within this method.
            adj (Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (BoolTensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        :rtype: (:class:`Tensor`, :class:`Tensor`, :class:`Tensor`,
            :class:`Tensor`, :class:`Tensor`, :class:`Tensor`)
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x
        adj = adj.unsqueeze(0) if adj.dim() == 2 else adj

        s = self.mlp(x)
        s = torch.softmax(s, dim=-1)

        batch_size, n_nodes, _ = x.size()

        if mask is not None:
            mask = mask.view(batch_size, n_nodes, 1).to(x.dtype)
            x, s = x * mask, s * mask

        # Pooled features and adjacency
        if self.return_pooled_graph:
            x_pool = torch.matmul(s.transpose(1, 2), x)
            adj_pool = torch.matmul(torch.matmul(s.transpose(1, 2), adj), s) 

        # Total variation loss
        tv_loss = self.totvar_coeff*torch.mean(self.totvar_loss(adj, s))
        
        # Balance loss
        bal_loss = self.balance_coeff*torch.mean(self.balance_loss(s))

        if self.return_selection and self.return_pooled_graph:
            return s, x_pool, adj_pool, tv_loss, bal_loss
        elif self.return_selection and not self.return_pooled_graph:
            return s, tv_loss, bal_loss
        else:
            return x_pool, adj_pool, tv_loss, bal_loss

    def totvar_loss(self, adj, s):

        batch_idx, node_i, node_j = torch.nonzero(adj, as_tuple=True)
        l1_norm = torch.sum(torch.abs(s[batch_idx, node_i, :] - s[batch_idx, node_j, :]), dim=(-1)) 
        loss = torch.sum(adj[batch_idx, node_i, node_j]*l1_norm)
        
        # Normalize the loss
        n_edges = len(node_i)
        loss *= 1 / (2 * n_edges)

        return loss

    def balance_loss(self, s):
        n_nodes = s.size()[-2]

        # k-quantile
        idx = int(math.floor(n_nodes / self.k))
        quant = torch.sort(s, dim=-2, descending=True)[0][:, idx, :] # shape [B, K]

        # Asymmetric l1-norm
        loss = s - torch.unsqueeze(quant, dim=1)
        loss = (loss >= 0) * (self.k - 1) * loss + (loss < 0) * loss * -1 
        loss = torch.sum(loss, axis=(-1, -2)) # shape [B]
        loss = 1 / (n_nodes * (self.k - 1)) * (n_nodes * (self.k - 1) - loss)

        return loss
