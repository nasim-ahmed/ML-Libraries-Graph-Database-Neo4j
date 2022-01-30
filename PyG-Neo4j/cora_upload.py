from json.tool import main
from neo4j_connections import Neo4jConnection
import yaml
import pandas as pd


class Cora_Loader():
    def __init__(self, edge_path, node_path):
        self.edge_path = edge_path
        self.node_path = node_path


    def create_edge_list(self):
        edge_list = pd.read_csv(self.edge_path, sep="\t", header=None, names=["target", "source"],)
        edge_list["label"] = "cites"
        return edge_list

        
    def create_node_list(self):
        self.feature_names = ["w_{}".format(i) for i in range(1433)]
        column_names = self.feature_names + ["subject"]
        node_list = pd.read_csv(self.node_path, sep="\t", header=None, names=column_names)
        node_list.insert(0, 'edge_index', range(0, len(node_list)))
        return node_list

    
    def preprocess_node_list(self, n_list):
        n_list["feature"] = n_list[self.feature_names].values.tolist()
        n_list = n_list.drop(columns=self.feature_names)
        n_list["id"] = n_list.index
        return n_list
        




def load_config(configuration): 
    global config
    with open(configuration) as config_file: 
        config = yaml.load(config_file, Loader = yaml.FullLoader)


def create_nodes_neo4j(node_list):
    loading_node_query = """
    UNWIND $node_list as node
    CREATE( e: paper {
        ID: toInteger(node.id),
        edge_index: toInteger(node.edge_index),
        subject: node.subject,
        features: node.feature
    })
    """

    connection = Neo4jConnection(config)

    batch_len = 500

    for batch_start in range(0, len(node_list), batch_len):
        batch_end = batch_start + batch_len
        records = node_list.iloc[batch_start:batch_end].to_dict("records")
        connection.query(loading_node_query, {"node_list": records})


def create_edge_list(edge_list):
    loading_edge_query = """
    UNWIND $edge_list as edge

    MATCH(source: paper {ID: toInteger(edge.source)})
    MATCH(target: paper {ID: toInteger(edge.target)})

    MERGE (source)-[r:cites]->(target)
    """

    batch_len = 500

    connection = Neo4jConnection(config)

    for batch_start in range(0, len(edge_list), batch_len):
        batch_end = batch_start + batch_len
        # turn edge dataframe into a list of records
        records = edge_list.iloc[batch_start:batch_end].to_dict("records")
        connection.query(loading_edge_query, {"edge_list": records})


def create_constraint():
    node_id_constraint = """
        CREATE CONSTRAINT
        ON (n:paper)
        ASSERT n.ID IS UNIQUE
        """
    connection = Neo4jConnection(config)
    connection.query(node_id_constraint)

def main():
    configure = './config.yaml'
    load_config(configure)

    loader = Cora_Loader("data/cora.cites", "data/cora.content")
    edge_list = loader.create_edge_list()
    node_list = loader.create_node_list()
    node_list = loader.preprocess_node_list(node_list)
    
   
    create_nodes_neo4j(node_list)
    create_edge_list(edge_list)
    create_constraint()
    

if __name__ == "__main__":
    main()