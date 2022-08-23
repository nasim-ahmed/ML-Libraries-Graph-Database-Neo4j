from neo4j_connections import Neo4jConnection
import pandas as pd 
import yaml
import json

class EdgeLoader(object):
    def __init__(self, configuration, graph_dir):
        self.configuration = configuration 
        self.graph_dir = graph_dir

    def load_config(self):
        global config
        with open(self.configuration, 'r') as config_file:
            config = yaml.load(config_file, Loader = yaml.FullLoader)


    def create_edge_list(self, path):
        edge_list = pd.read_csv(path)
        edge_list = edge_list.drop('Unnamed: 0', axis=1)
        return edge_list


def create_edge_nodes(edge_list, connection):
    loading_edge_query = """
        UNWIND $edge_list as edge
        MATCH(source: patient {uniquepid: edge.source})
        MATCH(destination: patient {uniquepid: edge.destination})
        MERGE (source)-[r:IS_SIMILAR{weight: edge.weight}]->(destination)
    """

    batch_len = 1000

    for batch_start in range(0, len(edge_list), batch_len):
        batch_end = batch_start + batch_len
        # turn edge dataframe into a list of records
        records = edge_list.iloc[batch_start:batch_end].to_dict("records")
        connection.query(loading_edge_query, {"edge_list": records})

def main():
    # for debugging
    # eICU_path = "/media/nasim/31c299f0-f952-4032-9bd8-001b141183e0/ML-Libraries-Graph-Database-Neo4j/PyG-Neo4j/paths.json"

    with open('paths.json', 'r') as f:
        graph_dir = json.load(f)["graph_dir"]

    configure = 'config.yaml'
    edge_path = "/media/nasim/31c299f0-f952-4032-9bd8-001b141183e0/ML-Libraries-Graph-Database-Neo4j/PyG-Neo4j/app/graphs/nodes_edges_k=3_adjusted_ns.csv"
    
    edgeLoader = EdgeLoader(configure, graph_dir)
    edge_list = edgeLoader.create_edge_list(edge_path)
    
    edgeLoader.load_config()
    connection = Neo4jConnection(config)

    create_edge_nodes(edge_list, connection)

    
 
if __name__ == "__main__":
    main()