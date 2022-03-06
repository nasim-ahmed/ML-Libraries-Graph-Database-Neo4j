from json.tool import main
import yaml
import pandas as pd
import sys
from enum import Enum
import sys
from torch.nn import ReLU
sys.path.append('../')
from neo4j_connections import Neo4jConnection
import coloredlogs, logging
from sklearn.model_selection import train_test_split
import torch
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv, Linear
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear
import torch.nn.functional as F
from tqdm import tqdm
import time
import os.path as osp
import numpy as np

torch.cuda.empty_cache()

mylogs = logging.getLogger(__name__)

device = torch.device('cpu' if torch.cuda.is_available() else 'cpu')



class HGT(torch.nn.Module):
    def __init__(self, data, hidden_channels, out_channels, num_heads, num_layers):
        super().__init__()

        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            self.lin_dict[node_type] = Linear(-1, hidden_channels)

        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
        self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        for node_type, x in x_dict.items():
            x_dict[node_type] = self.lin_dict[node_type](x).relu_()

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        return self.lin(x_dict['paper'])




def load_config(configuration): 
    global config
    with open(configuration) as config_file: 
        config = yaml.load(config_file, Loader = yaml.FullLoader)


def fetch_node_features():
    paper_nodes_query = """
        MATCH (p:paper) RETURN p.feature AS feature, p.name AS name, p.year AS year, p.venue AS venue, p.label AS label
        """

    connection = Neo4jConnection(config)
    result = connection.query(paper_nodes_query)

    data_paper_y = result['label']

    new_df = result.drop(['name', 'year', 'venue', 'label'], axis=1)
    data_paper_x = pd.DataFrame(new_df['feature'].to_list())

    return data_paper_x, data_paper_y

def get_edges(query):
    connection = Neo4jConnection(config)
    result = connection.query(query)
    return result


'''
 Fetch edges for paper cites paper 
'''
def fetch_cites_edge_list():
    paper_cites_paper_query = """
     MATCH (n:paper)-[:cites]->(m: paper) RETURN n.id, m.id
    """

    result = get_edges(paper_cites_paper_query)

    return result


'''
 Fetch edges for paper has_topic field_of_study 
'''
def fetch_has_topic_edge_list():
    paper_has_topic_query = """
     MATCH (n:paper)-[:has_topic]->(m: field_of_study) RETURN n.id, m.id
    """

    result = get_edges(paper_has_topic_query)

    return result


'''
 Fetch edges for author writes paper 
'''
def fetch_author_writes_paper():
    author_writes_paper_query = """
     MATCH (n:author)-[:writes]->(m: paper) RETURN n.id, m.id
    """
    result = get_edges(author_writes_paper_query)
    return result


'''
 Fetch edges for author affiliated with institution
'''
def fetch_author_affiliated_with():
    author_affiliated_with_query = """
     MATCH (n:author)-[:affiliated_with]->(m: institution) RETURN n.id, m.id
    """
    result = get_edges(author_affiliated_with_query)
    return result
    


def create_train_test_mask(y):
    x_train, x_test = train_test_split(pd.Series(y), test_size=0.3, random_state=42)
    train_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    test_mask = torch.zeros(y.shape[0], dtype= torch.bool)
    train_mask[x_train.index] = True 
    test_mask[x_test.index] = True
    return train_mask, test_mask 


def create_train_val_test_mask(raw_dir, y):
    masks = []
    for f, v in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        path = osp.join(raw_dir, 'split', 'time', 'paper',
                        f'{f}.csv.gz')
        idx = pd.read_csv(path, compression='gzip', header=None,
                            dtype=np.int64).values.flatten()
        idx = torch.from_numpy(idx)
        mask = torch.zeros(y.shape[0], dtype=torch.bool)
        mask[idx] = True
        #data['paper'][f'{v}_mask'] = mask
        masks.append(mask)
    return masks



def map_edge_list(lst1, lst2):
    final_lst=[]
    set1= set(lst1+lst2)

    i=0
    lst1_new={}
    for val in set1:
        lst1_new[val]=i
        i=i+1

    for x in range(len(lst1)):
        #print(x)
        start= lst1_new[lst1[x]]
        end= lst1_new[lst2[x]]
        final_lst.append([start,end])
    
    return final_lst



def nodeFeaturesUsingMetapath2Vec(data):
    metapath = [('author', 'writes', 'paper'),('author', 'affiliated_with', 'institution'),('paper', 'has_topic', 'field_of_study')]
    model = MetaPath2Vec(data.edge_index_dict,embedding_dim=128,metapath=metapath,walk_length=5,walks_per_node=3,num_negative_samples=1).to(device)
    

    loader = model.load_state_dict(torch.load(pretrained_path))
    return loader
    
def reverse_edge_index(data):
    #print(data)
    cols = data.columns.tolist()
    #print(cols)
    rev_cols = reversed(cols)
    #print(rev_cols)
    df1=data.reindex(columns=rev_cols)
    #print(data.head(5))
    #print(df1.head(5))
    return df1.values.tolist()


def main():
    configure = 'config.yaml'
    load_config(configure)
    transform = T.ToUndirected(merge=True)
    dataset = OGB_MAG(root='./data', preprocess="metapath2vec", transform=transform)
    
    ogb_data = dataset[0]

    data_paper_x, data_paper_y = fetch_node_features()
    
    #coloredlogs.install(level=logging.DEx_node_features())
    
    # print("Time to fetch nodes %s seconds" % (time.time() - st_time_nodes))

    # # #mylogs.info("Printing paper cites paper")
    # st_time_edges = time.time()

    # # #Get paper edge features 
    paper_cites_paper_edges = fetch_cites_edge_list()
    
    # mylogs.info("Logging paper has topic")

    # #Get paper has topic features
    paper_has_topic_edges = fetch_has_topic_edge_list()
    
    # #mylogs.info("Logging author writes paper")

    # #Get author writes paper
    author_writes_paper_edges = fetch_author_writes_paper()

    # mylogs.info("author affiliated with institution")

    # #Get author affiliated with institution
    author_affiliated_with_edges = fetch_author_affiliated_with()

    # print("Time to fetch edges %s seconds" % (time.time() - st_time_edges))
    
    # train_mask, test_mask = create_train_test_mask(data_paper_y)
    raw_dir_path = "/home/nasim/Desktop/PyG-Neo4j/OGB-MAG/data/mag/raw"
    mask_list = create_train_val_test_mask(raw_dir_path, data_paper_y)
    

    data = HeteroData()

    st_time_nodes = time.time()



    # data['paper'].x = torch.tensor(data_paper_x.values, dtype = torch.float)
    # data['paper'].y = torch.tensor(data_paper_y, dtype = torch.long)
    # data['paper'].train_mask = train_mask
    # data['paper'].test_mask = test_mask
    # data['author'].x = ogb_data['author']['x']
    # data['institution'].x = ogb_data['institution']['x']
    # data['field_of_study'].x = ogb_data['field_of_study']['x']
    # data['paper', 'cites', 'paper'].edge_index = torch.tensor(paper_cites_paper_edges.values.tolist(), dtype=torch.long).t().contiguous()
    # data['author', 'writes', 'paper'].edge_index = torch.tensor(author_writes_paper_edges.values.tolist(), dtype=torch.long).t().contiguous()
    # data['author', 'affiliated_with', 'institution'].edge_index = torch.tensor(author_affiliated_with_edges.values.tolist(), dtype=torch.long).t().contiguous()
    #data['author', 'has_topic', 'institution'].edge_index = torch.tensor(paper_has_topic_edges.values.tolist(), dtype=torch.long).t().contiguous()
    
    data['paper'].x = torch.tensor(data_paper_x.values, dtype = torch.float)
    data['paper'].y =  torch.tensor(data_paper_y, dtype = torch.long)
    data['paper'].train_mask = mask_list[0]
    data['paper'].val_mask = mask_list[1]
    data['paper'].test_mask =  mask_list[2]
    #data['author'].x =  ogb_data['author']['x']
    #data['institution'].x = ogb_data['institution']['x']
    #data['field_of_study'].x = ogb_data['field_of_study']['x']
    data['paper', 'cites', 'paper'].edge_index = torch.tensor(paper_cites_paper_edges.values.tolist(), dtype=torch.long).t().contiguous() #ogb_data['paper', 'cites', 'paper']['edge_index']
    data['author', 'writes', 'paper'].edge_index = torch.tensor(author_writes_paper_edges.values.tolist(), dtype=torch.long).t().contiguous() #ogb_data['author', 'writes', 'paper']['edge_index']
    data['author', 'affiliated_with', 'institution'].edge_index = torch.tensor(author_affiliated_with_edges.values.tolist(), dtype=torch.long).t().contiguous() #ogb_data['author', 'affiliated_with', 'institution']['edge_index']
    data['paper', 'has_topic', 'field_of_study'].edge_index = torch.tensor(paper_has_topic_edges.values.tolist(), dtype=torch.long).t().contiguous() #ogb_data['paper', 'has_topic', 'field_of_study']['edge_index']
    data['institution', 'rev_affiliated_with', 'author'].edge_index = torch.tensor(reverse_edge_index(author_affiliated_with_edges), dtype=torch.long).t().contiguous()#ogb_data['institution', 'rev_affiliated_with', 'author']['edge_index']
    data['paper', 'rev_writes', 'author'].edge_index = torch.tensor(reverse_edge_index(author_writes_paper_edges), dtype=torch.long).t().contiguous() #ogb_data['paper', 'rev_writes', 'author']['edge_index']
    data['field_of_study', 'rev_has_topic', 'paper'].edge_index = torch.tensor(reverse_edge_index(paper_has_topic_edges), dtype=torch.long).t().contiguous() #ogb_data['field_of_study', 'rev_has_topic', 'paper']['edge_index']

    pretrained_path = "/home/nasim/Desktop/PyG-Neo4j/OGB-MAG/data/mag/raw/mag_metapath2vec_emb.pt"
    
    emb_dict = torch.load(pretrained_path)
    for key, value in emb_dict.items():
        if key != 'paper':
            #print(key, value)
            data[key].x = value

    #loaded_model = nodeFeaturesUsingMetapath2Vec(data,path='')
    #data['author'].x = loaded_model('author',batch=data.y_dict_index['author'] ).detach().numpy()
    #data = data.to(device, 'x', 'y')

    train_input_nodes = ('paper', data['paper'].train_mask)
    val_input_nodes = ('paper', data['paper'].test_mask)
    kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}


    train_loader = NeighborLoader(data, num_neighbors=[10] * 2, shuffle=True,
                                  input_nodes=train_input_nodes, **kwargs)
    val_loader = NeighborLoader(data, num_neighbors=[10] * 2,
                                input_nodes=val_input_nodes, **kwargs)

    # #data.transform = T.ToUndirected(merge=True)

    model = Sequential('x, edge_index', [
        (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (SAGEConv((-1, -1), 64), 'x, edge_index -> x'),
        ReLU(inplace=True),
        (Linear(-1, dataset.num_classes), 'x -> x'),
    ])
    model = to_hetero(model, data.metadata(), aggr='sum').to(device)


    @torch.no_grad()
    def init_params():
        # Initialize lazy parameters via forwarding a single batch to the model:
        batch = next(iter(train_loader))
        batch = batch.to(device, 'edge_index')
        model(batch.x_dict, batch.edge_index_dict)


    def train():
        model.train()

        total_examples = total_loss = 0
        for batch in tqdm(train_loader):
            optimizer.zero_grad()
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            loss = F.cross_entropy(out, batch['paper'].y[:batch_size])
            loss.backward()
            optimizer.step()

            total_examples += batch_size
            total_loss += float(loss) * batch_size

        return total_loss / total_examples


    @torch.no_grad()
    def test(loader):
        model.eval()

        total_examples = total_correct = 0
        for batch in tqdm(loader):
            batch = batch.to(device, 'edge_index')
            batch_size = batch['paper'].batch_size
            out = model(batch.x_dict, batch.edge_index_dict)['paper'][:batch_size]
            pred = out.argmax(dim=-1)

            total_examples += batch_size
            total_correct += int((pred == batch['paper'].y[:batch_size]).sum())

        return total_correct / total_examples


    init_params()  # Initialize parameters.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(1, 2):
        loss = train()
        val_acc = test(val_loader)
        print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, Val: {val_acc:.4f}')

    #Model evaluation
    # model.eval()
    # pred = model(data).argmax(dim=1)
    # correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
    # acc = int(correct) / int(data.test_mask.sum())
    # print(f'Accuracy: {acc:.4f}')
    model.eval()
    out = model(data.x_dict, data.edge_index_dict)['paper']
    pred = out.argmax(dim=-1)
    mask = data['paper']['test_mask']
    acc = (pred[mask] == data['paper'].y[mask]).sum() / mask.sum()
    print(f'Accuracy: {acc:.4f}')
       
    



if __name__ == "__main__":
    main()