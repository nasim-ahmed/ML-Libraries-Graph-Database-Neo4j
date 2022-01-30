from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
import torch 
import torch.nn.functional as F
import time 
from embedding_visualize import TSNE_VISUALIZE


class GAT(torch.nn.Module):
    def __init__(self, dataset):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1
        
        
        self.conv1 = GATConv(dataset.num_features, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid*self.in_head, dataset.num_classes, concat=False,
                             heads=self.out_head, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
                
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)


def train_gnn(train_data):
    epochs = 200
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = "cpu"

    model = GAT(train_data).to(device)

    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

    data = train_data.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    st_train_time = time.time()

    model.train()

    losses = []
    images = []
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        losses.append(loss)
        
        if epoch%100 == 0:
            tsne = TSNE_VISUALIZE()

            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            images.append(tsne.visualize(out=out, color=data.y, epoch=epoch))
        
        loss.backward()
        optimizer.step()

    print("Time to train GAT %s seconds" % (time.time() - st_train_time))
    print("TSNE Visualization finished.")
    
    #Model evaluation 
    model.eval()
    _, pred = model(data).max(dim=1)
    correct = float(pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
    acc = correct / data.test_mask.sum().item()
    print('Accuracy: {:.4f}'.format(acc))

    return losses, images 