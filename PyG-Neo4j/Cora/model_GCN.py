import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import time
from embedding_visualize import TSNE_VISUALIZE


class GCN(torch.nn.Module):
    def __init__(self, dataset):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_node_features, 8)
        self.conv2 = GCNConv(8, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)

        return F.log_softmax(x, dim=1)


def train_gcn(train_data):
    epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GCN(train_data).to(device)

    # Reset the previously trained model weights
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()
    
    data = train_data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

    
    st_train_time = time.time()

    model.train()
    
    #Custom traininig loop
    losses = []
    images = []

    
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
        losses.append(loss)
        tsne = TSNE_VISUALIZE()

        if epoch%100 == 0:
            print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
            images.append(tsne.visualize(out=out, color=data.y, epoch=epoch))
        
        loss.backward()
        optimizer.step()

    print("Time to train GCN %s seconds" % (time.time() - st_train_time))
    print("TSNE Visualization finished.")

    # Model evaluation
    model.eval()
    pred = model(data).argmax(dim=1)
    correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    acc = int(correct) / int(data.test_mask.sum())
    print(f'Accuracy: {acc:.4f}')

    return losses, images


   
