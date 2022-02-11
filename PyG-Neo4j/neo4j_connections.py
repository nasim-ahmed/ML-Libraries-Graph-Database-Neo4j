from neo4j import GraphDatabase
import pandas as pd
import yaml
import sys
import torch
import torch_geometric.transforms as T
from torch_geometric.nn import SAGEConv, to_hetero
from torch_geometric.data import HeteroData
from torch_geometric.transforms import ToUndirected, RandomLinkSplit
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

config = dict()

class Neo4jConnection(object):
    
    def __init__(self, config):
        self.__uri = config['server_uri']
        self.__user = config['admin_user']
        self.__pwd = config['admin_pass']
        self.__driver = None
        try:
            self.__driver = GraphDatabase.driver(self.__uri, auth=(self.__user, self.__pwd))
        except Exception as e:
            print("Failed to create the driver:", e)
        
    def close(self):
        if self.__driver is not None:
            self.__driver.close()
        
    def query(self, query, params={},db=None):
        assert self.__driver is not None, "Driver not initialized!"
        session = None
        response = None
        try: 
            session = self.__driver.session(database=db) if db is not None else self.__driver.session() 
            #response = list(session.run(query))
            result = session.run(query, params)
            dataframe = pd.DataFrame([r.values() for r in result], columns=result.keys())

        except Exception as e:
            print("Query failed:", e)
        finally: 
            if session is not None:
                session.close()
        return dataframe


    
def load_node(conn, cypher, index_col, encoders=None, **kwargs):
    # Execute the cypher query and retrieve data from Neo4j
    df = conn.query(cypher)
    df.set_index(index_col, inplace=True)
    # Define node mapping
    mapping = {index: i for i, index in enumerate(df.index.unique())}
    # Define node features
    x = None
    if encoders is not None:
        xs = [encoder(df[col]) for col, encoder in encoders.items()]
        x = torch.cat(xs, dim=-1)

    return x, mapping


def load_edge(conn, cypher, src_index_col, src_mapping, dst_index_col, dst_mapping,
                  encoders=None, **kwargs):
    # Execute the cypher query and retrieve data from Neo4j
    df = conn.query(cypher)
    # Define edge index
    src = [src_mapping[index] for index in df[src_index_col]]
    dst = [dst_mapping[index] for index in df[dst_index_col]]
    edge_index = torch.tensor([src, dst])
    # Define edge features
    edge_attr = None
    if encoders is not None:
        edge_attrs = [encoder(df[col]) for col, encoder in encoders.items()]
        edge_attr = torch.cat(edge_attrs, dim=-1)

    return edge_index, edge_attr


def load_config(configuration): 
    global config
    with open(configuration) as config_file: 
        config = yaml.load(config_file, Loader = yaml.FullLoader)


def main(): 
    configure = './config.yaml'
    load_config(configure)
    connection = Neo4jConnection()
    # 1. Project the GDS in-memory graph
    #connection.query(config['graph_create_gds'])
    # 2. Store the embeddings back in the database and
    #connection.query(config['graph_node2vec_write'])
     
    #stroke mapping and features
    stroke_x, stroke_mapping = load_node(connection,config['get_stroke_query'], index_col='sId')
    
    #patient mapping and features
    patient_x, patient_mapping = load_node(connection,
    config['get_patients_query'], 
    index_col='patientId', encoders={
        'gender': SequenceEncoder(),
        'rts': CategoryEncoder(),
        'embedding': IdentityEncoder(is_list=True)
    })
    
    # Fetch information about stroke from edge 
    # between Stroke and Patient node

    edge_index, edge_label = load_edge(connection,
        config['get_stroke_boolean_query'],
        src_index_col='sId',
        src_mapping=stroke_mapping,
        dst_index_col='patientId',
        dst_mapping=patient_mapping,
        encoders={'strokeVal': IdentityEncoder(dtype=torch.long)}
    )

    data = HeteroData()
    # Add stroke node features for message passing:
    data['str'].x = torch.eye(len(stroke_mapping), device=device)
    # Add patient node features
    data['pat'].x = patient_x
    # Add strokeValue between strokes and patients
    data['str', 'hasStr', 'pat'].edge_index = edge_index
    data['str', 'hasStr', 'pat'].edge_label = edge_label
    data.to(device, non_blocking=True)
    
    

    data = ToUndirected()(data)
    
    del data['pat', 'rev_hasStr', 'str'].edge_label  # Remove "reverse" label.

    # 2. Perform a link-level split into training, validation, and test edges.
    transform = RandomLinkSplit(
        num_val=0.1,
        num_test=0.1,
        neg_sampling_ratio=0.0,
        edge_types=[('str', 'hasStr', 'pat')],
        rev_edge_types=[('pat', 'rev_hasStr', 'str')],
    )
    train_data, val_data, test_data = transform(data)
   


if __name__ == "__main__":
    main()
