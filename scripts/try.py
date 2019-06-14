import torch
import torch.nn as nn
import numpy as np
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv
from torch_scatter import scatter_mean


class Net(torch.nn.Module):
    def __init__(self, input_ch, output_ch):
        super(Net, self).__init__()

        self.layer1 = GCNConv(input_ch, output_ch)

    def forward(self, data):
        print("=====================================================")
        x, edge_index, batch = data.x, data.edge_index, data.batch

        return x

def construct_batch_edge_index(x, edge_index):
    """

    Parameters
    ----------
    x : torch.tensor
        Shape = (m, num_nodes, node_fea)
    edge_index : torch.tensor
        Shape = (2, num_edges), data type = long
    Returns
    -------
    x_flat : torch.tensor
        Shape = (m * num_nodes, node_fea)
    all_edge_index : torch.tensor
        Shape = (2, num_edges * m), data type = long
    """

    single_num_edges = edge_index.shape[1]
    num_samples = x.shape[0]

    # Flatten x
    x_flat = x.view(-1, x.shape[2])

    # Repeat Edge index
    all_edge_index = torch.zeros((2, num_samples * single_num_edges))
    all_edge_index[:, 0:single_num_edges] = edge_index.clone()

    for i in range(num_samples):
        new_edge_index = (edge_index + 25 * i)
        all_edge_index[:, i * single_num_edges: (i+1) * single_num_edges ] = new_edge_index
    return x_flat, all_edge_index.long()



batch_size = 10
num_nodes = 25

np.random.seed(50)
x = np.random.randint(1, 10, (num_nodes, 2))
x = np.repeat(x[np.newaxis], batch_size, axis=0)

x_tensor = torch.from_numpy(x)
edge_index = torch.tensor(([1, 2, 3, 4], [3, 4, 1, 10])).long()
data_list = [Data(x = x_tensor_each, edge_index=edge_index) for x_tensor_each in x_tensor]
loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

x_tensor_recon, batch_edge_index_recon = construct_batch_edge_index(x_tensor, edge_index)

for data_batch in loader:
    output = data_batch.x


    # Convert to np
    output_np = output.numpy()
    x_tensor_recon = x_tensor_recon.numpy()
    batch_edge_index = data_batch.edge_index.numpy()
    batch_edge_index_recon = batch_edge_index_recon.numpy()


    # Compare
    arr_flag = np.array_equal(x_tensor_recon, output_np)
    edge_flag = np.array_equal(batch_edge_index, batch_edge_index_recon)
    print("Same input arr = ", arr_flag)
    print("Same edge_index = ", edge_flag)
    print("edge_index_ori = \n", edge_index)
    print("edge_index_collated = \n", batch_edge_index)
    print("edge_index_recon = \n", batch_edge_index_recon)

