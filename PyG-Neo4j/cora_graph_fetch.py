from json.tool import main
from neo4j_connections import Neo4jConnection
import yaml
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch 
from torch_geometric.data import Data
import torch_geometric.transforms as T
import model_GAT
import model_GCN
import time
import seaborn as sns
import matplotlib.pyplot as plt
from moviepy.editor import ImageSequenceClip
import matplotlib; matplotlib.use('agg')

def load_config(configuration): 
    global config
    with open(configuration) as config_file: 
        config = yaml.load(config_file, Loader = yaml.FullLoader)


def fetch_node_features():
    query = """
        MATCH (n) RETURN n.features AS feature, n.subject AS label
        """  

    connection = Neo4jConnection(config)
    result = connection.query(query)

    label_df = result['label']
    flattened_df_y = label_df.to_numpy().flatten()
    le = LabelEncoder()
    y = le.fit_transform(flattened_df_y)
     
    new_df = result.drop(['label'], axis=1)
    pre_data_x = pd.DataFrame(new_df['feature'].to_list())

    return pre_data_x,y 


def create_train_test_mask(y):
    x_train, x_test = train_test_split(pd.Series(y), test_size=0.3, random_state=42)
    train_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    test_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    train_mask[x_train.index] = True 
    test_mask[x_test.index] = True
    return train_mask, test_mask 



def fetch_edge_list():
    e_query = """
     MATCH (n:paper)-[:cites]-(m) RETURN n.edge_index, m.edge_index
    """
    connection = Neo4jConnection(config)
    result = connection.query(e_query)

    return result.transpose()

#Visualising in Loss
def plot_train_loss(gat_loss, gcn_loss):
    gat_losses_float = [float(loss.cpu().detach().numpy()) for loss in gat_loss]
    gat_losses_indices = [i for i, l in enumerate(gat_losses_float)]

    gcn_losses_float = [float(loss.cpu().detach().numpy()) for loss in gcn_loss]
    gcn_losses_indices = [i for i, l in enumerate(gcn_losses_float)]

    sns.lineplot(gat_losses_indices, gat_losses_float, x="epochs",y="loss")
    sns.lineplot(gcn_losses_indices,gcn_losses_float, x="epochs",y="loss")
    plt.legend(labels=["GAT","GCN"])

    plt.savefig("figure/train_losses.png") 

def create_gif(images, name="GAT"):
    fps = 1
    filename = "figure/embeddings_{}.gif".format(name)
    clip = ImageSequenceClip(images, fps=fps)
    clip.write_gif(filename, fps=fps)


def main():
    configure_path = './config.yaml'
    load_config(configure_path)

    st_time_nodes = time.time()
    x, y = fetch_node_features()
    print("Time to fetch nodes %s seconds" % (time.time() - st_time_nodes))
    
    st_time_edges = time.time()
    edge_list = fetch_edge_list()
    print("Time to fetch edges %s seconds" % (time.time() - st_time_edges))
    

    train_mask, test_mask = create_train_test_mask(y)
    edge_index = torch.tensor(edge_list.values, dtype=torch.long)
    y = torch.tensor(y, dtype=torch.long)
    x_main = torch.tensor(x.values, dtype = torch.float)


    data = Data(x=x_main, edge_index=edge_index.contiguous(), train_mask = train_mask, 
              test_mask = test_mask, y = y, num_classes = len(y.unique()))
    data.transform = T.NormalizeFeatures()

    print("*"*20)
    print("GCN Training in progress")
    print("*"*20)

    #Train GCN
    gcn_train_loss, gcn_images = model_GCN.train_gcn(data)
    create_gif(gcn_images, name = "gcn")
 

    print("*"*20)
    print("GAT Training in progress")
    print("*"*20)

    #Train GAT
    gat_train_loss, gat_images = model_GAT.train_gnn(data)
    create_gif(gat_images, name = "gat")

    plot_train_loss(gat_train_loss, gcn_train_loss)

    



if __name__ == "__main__":
    main()