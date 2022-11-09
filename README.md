**TODO**

- [ ] Docstring in GTVConv and AsymCheegerCut files Spektral
- [ ] Docstring in GTVConv and AsymCheegerCut files PyG
- [ ] conda env for pytorch
- [ ] graph classification in pytorch
- [ ] report new hyperparameters on the paper
- [ ] put the paper on arxiv
- [ ] fix references to the paper
- [ ] write down citation

# Introduction 
Implementation in Tensorflow and Pytorch of the Total Variation Graph Neural Network (TVGNN) as presented in the [original paper](https://arxiv.org/).

The TVGNN model can be used to **cluster** the vertices of an annotated graph, by accounting both for the graph topology and the vertex features. Compared to other GNNs for clustering, TVGNN creates *sharp* cluster assignments that better approximate the optimal (in the minimum cut sense) partition.

<img align="center" width="273" height="143" src="img/sharp.png" alt="smooth and sharp clustering assignments">

The TVGNN model can also be used to implement **graph pooling** in a deep GNN architecture for tasks such as graph classification.

# Model description 
TVGNN consists of the GTVConv layer and the AsymmetricCheegerCut layer.

### GTVConv
The GTVConv layer is a *message-passing* layer that minimizes the $L_1$-norm of the features in adjacent nodes in the graphs. The $l$-th GTVConv layer updates the node features as
$$\mathbf{X}^{(l+1)}  = \sigma\left[ \left( \mathbf{I} - \delta \mathbf{L}_\Gamma^{(l)}  \right) \mathbf{X}^{(l)}\mathbf{\Theta}  \right] $$ 
where $\sigma$ is a non-lineary, $\mathbf{\Theta}$ are the trainable weights of the layer, and $\delta$ is an hyperparameter. The Laplacian matrix is defined as $\mathbf{L}_\Gamma^{(l)} = \mathbf{D}_\Gamma - \mathbf{\Gamma}$, where $\mathbf{D}_\Gamma = \text{diag}(\Gamma \boldsymbol{1})$ and 
$$ [\mathbf{\Gamma}]_{ij} = \frac{a_{ij}}{\texttt{max}\{ \| \boldsymbol{x}_i^{(l)} - \boldsymbol{x}_j^{(l)}  \|_1, \epsilon \}}$$

where $a_{ij}$ is the $ij$-th entry of the adjacency matrix, $\boldsymbol{x}_i^{(l)}$ is the feature of vertex $i$ at layer $l$ and $\epsilon$ is a small constant that avoids zero-division.

### AsymCheegerCut
The AsymCheegerCut is a *graph pooling* layer that internally contains an $\texttt{MLP}$ parametrized by $\mathbf{\Theta}_\text{MLP}$ and that computes:
- a cluster assignment matrix $\mathbf{S} = \texttt{Softmax}(\texttt{MLP}(\mathbf{X}; \mathbf{\Theta}_\text{MLP})) \in \mathbb{R}^{N\times K}$, which maps the $N$ vertices in $K$ clusters,
- an unsupervised loss $\mathcal{L} = \alpha_1\mathcal{L}_\text{GTV} + \alpha_2\mathcal{L}_\text{AN}$, where $\alpha_1$ and $\alpha_2$ are two hyperparameters,
- the adjacency matrix and the vertex features of a coarsened graph
$$
    \mathbf{A}^\text{pool} = \mathbf{S}^T \tilde{\mathbf{A}} \mathbf{S} \in\mathbb{R}^{K\times K}; \, \mathbf{X}^\text{pool}=\mathbf{S}^T\mathbf{X} \in\mathbb{R}^{K\times F}.
$$

The term $\mathcal{L}_\text{GTV}$ in the loss minimizes the graph total variation of the cluster assignments $\mathbf{S}$ and is defined as
$$\mathcal{L}_\text{GTV} = \frac{\mathcal{L}_\text{GTV}^*}{2E} \in [0, 1],$$
where $\mathcal{L}_\text{GTV}^* = \sum_{k=1}^K \sum_{i=1}^N \sum_{j=i}^N a_{i,j} |s_{i,k} - s_{j,k}|$, $s_{i,k}$ is the assignment of vertex $i$ to cluster $k$ and $E$ is the number of edges.

The term $\mathcal{L}_\text{AN}$ encourages the partition to be balanced and is defined as
$$\mathcal{L}_\text{AN} = \frac{\beta -  \mathcal{L}_\text{AN}^*}{\beta} \in [0, 1],$$
where $\mathcal{L}_\text{AN}^* = \sum_{k=1}^K ||\boldsymbol{s}_{:,k} - \textrm{quant}_\rho (\boldsymbol{s}_{:,k})||_{1, \rho}$.
When $\rho = K-1$, $\beta = N\rho$.
When $\rho$ takes different values, $\beta = N\rho\min(1, K/(\rho+1))$. 

$\text{quant}_\rho(\boldsymbol{s}_k)$ denotes the $\rho$-quantile of $\boldsymbol{s}_k$ and $||\cdot||_{1,\rho}$ denotes an asymmetric $\ell_1$ norm, which for a vector $\boldsymbol{x}\in\mathbb{R}^{N\times 1}$ is

$$||\boldsymbol{x}||_{1,\rho} = \sum_{i=1}^N |x_{i}|_\rho, \,\textrm{where}\, |x_i|_\rho = \begin{cases}\rho x_i, & x_i\geq 0\\ -x_i, & x_i < 0 \end{cases}.$$ 

# Downstream tasks
We use TVGNN to perform vertex clustering and graph classification. Other tasks such as graph regression could be considered as well.

### Vertex clustering
This is an unsupervised task, where the goal is to generate a partition of the vertices based on the similarity of their vertex features and the graph topology. The GNN model is trained only by minimizing the unsupervised loss $\mathcal{L}$.

<img align="center" width="215" height="143" src="img/clustering.png" alt="clustering architecture">

### Graph classification
This is a supervised with goal of predicting the class of each graph. The GNN rchitectures for graph classification alternates GTVConv layers with a graph pooling layer, which gradually distill the global label information from the vertex representations. The GNN is trained by minimizing the unsupervised loss $\mathcal{L}$ for each pooling layer and a supervised cross-entropy loss $\mathcal{L}_\text{cross-entr}$ between the true and predicted class label.

<img align="center" width="710" height="185" src="img/classification.png" alt="classification architecture">

# Implementation

<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/2/2d/Tensorflow_logo.svg" alt="Tensorflow icon">

## Tensorflow
This implementation is based on the [Spektral](https://graphneural.network/) library and follows the [Select-Reduce-Connect](https://graphneural.network/layers/pooling/#srcpool) API.
To execute the code, first install the conda environment from [tf_environment.yml](tensorflow/tf_environment.yml) as

    conda env create -f tf_environment.yml

The ``tensorflow/`` folder includes:

- The implementation of the [GTVConv](/tensorflow/GTVConv.py) layer
- The implementation of the [AsymmetricCheegerCutPool](/tensorflow/AsymCheegerCutPool.py) layer
- An example script to perform the [clustering](/tensorflow/clustering.py) task
- An example script to perform the  [classification](/tensorflow/classification.py) task

<img align="left" width="30" height="30" src="https://upload.wikimedia.org/wikipedia/commons/1/10/PyTorch_logo_icon.svg" alt="Pytorch icon">

## Pytorch
This implementation is based on the [Pytorch Geometric]() library.

- [GTVConv]() layer
- [AsymmetricCheegerCutPool]() layer
- [Vertex clustering]() example
- [Graph classification]() example

# Citation
If you use TVGNN in your research, please consider citing our work as
...