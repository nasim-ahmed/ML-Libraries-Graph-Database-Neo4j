from json.tool import main
import yaml
import pandas as pd
import sys
from enum import Enum
import sys
sys.path.append('../')
from neo4j_connections import Neo4jConnection
import logging

class NodeType(Enum):
    PAPER = "paper"
    AUTHOR = "author"
    FIELD_OF_STUDY = "field_of_study"
    INSTITUTION = "institution"


class Mag_Loader():
    def __init__(self, node_paths, paper_feature_paths, edge_paths):
        self.node_paths = node_paths
        self.paper_feature_paths = paper_feature_paths
        self.edge_paths = edge_paths


    def create_csv(self, path):
        df = pd.read_csv(path)
        return df 
    
    def create_node_features(self,df):
        year_df = pd.read_csv(self.paper_feature_paths[0], header=None)
        df["year"] = year_df
        feature_df = pd.read_csv(self.paper_feature_paths[1], header=None)
        df["feature"] = feature_df.values.tolist()
        return df

    def create_node_list(self):
        df_list = []
        for path in self.node_paths:
            if "paper" in path:
                df = self.create_csv(path)
                df_paper = self.create_node_features(df)
                df_paper.rename(columns={'ent idx': 'idx', 'ent name': 'name'}, inplace=True)
                df_list.append(df_paper)
            else: 
                other_df = self.create_csv(path)
                other_df.rename(columns={'ent idx': 'idx', 'ent name': 'name'}, inplace=True)
                df_list.append(other_df)
        return df_list 

    def create_edge_list(self):
        column_names = ["source", "target"]
        edge_list = []
        for path in self.edge_paths:
            df = pd.read_csv(path, header=None, names=column_names)
            edge_list.append(df)
        return edge_list
        

def load_config(configuration): 
    global config
    with open(configuration) as config_file: 
        config = yaml.load(config_file, Loader = yaml.FullLoader)

def upload_nodes(df_list):
    #Create papers nodes
    create_nodes_neo4j(df_list[0], NodeType.PAPER.value)
    logging.info('Papers has been loaded to Database')

     #Create author nodes
    create_nodes_neo4j(df_list[1], NodeType.AUTHOR.value)
    logging.info('Author has been loaded to Database')

    #Create institution nodes
    create_nodes_neo4j(df_list[2], NodeType.INSTITUTION.value)
    logging.info('Institution has been loaded to Database')

    #Create field of study nodes
    create_nodes_neo4j(df_list[3], NodeType.FIELD_OF_STUDY.value)
    logging.info('Field of study has been loaded to Database')


def upload_edges(edge_list):
    #Create papers cites paper edges
    create_edge_list(edge_list[0], NodeType.PAPER.value)
    logging.info('Papers cites paperhas been loaded to Database')

    #Create papers has topic field of study
    create_edge_list(edge_list[1], NodeType.FIELD_OF_STUDY.value)
    logging.info('Papers has topic field of study has been loaded to Database')

    #Create author writes paper
    create_edge_list(edge_list[2], NodeType.AUTHOR.value)
    logging.info('Author writes paper loaded to Database')

    #Create author affiliated with institution
    create_edge_list(edge_list[3], NodeType.INSTITUTION.value)
    logging.info('Author affiliated with institution been loaded to Database')


def main():
    configure = '../config.yaml'
    load_config(configure)
    
    paper_feature_paths = ['../dataset/mag/raw/node-feat/paper/node_year.csv',
        '../dataset/mag/raw/node-feat/paper/node-feat.csv'
    ]

    node_paths = ['../dataset/mag/mapping/paper_entidx2name.csv',
    '../dataset/mag/mapping/author_entidx2name.csv', 
    '../dataset/mag/mapping/institution_entidx2name.csv', 
    '../dataset/mag/mapping/field_of_study_entidx2name.csv']

    edge_paths = ['../dataset/mag/raw/relations/paper___cites___paper/edge.csv',
    '../dataset/mag/raw/relations/paper___has_topic___field_of_study/edge.csv', 
    '../dataset/mag/raw/relations/author___writes___paper/edge.csv', 
    '../dataset/mag/raw/relations/author___affiliated_with___institution/edge.csv'
    ]


    loader = Mag_Loader(node_paths, paper_feature_paths, edge_paths)
    df_list = loader.create_node_list()
    logging.info('Dataframe list has been populated')
    
    #upload nodes to Neo4j
    #upload_nodes(df_list)

    #upload relationships
    edge_list = loader.create_edge_list()
    upload_edges(edge_list)
    
    


def create_edge_list(edge_list, node_type = NodeType.PAPER.value):
    paper_cites_paper_query = """
    UNWIND $edge_list as edge
    MATCH (source: paper{id: toInteger(edge.source)})
    MATCH (target: paper{id: toInteger(edge.target)}) 
    MERGE (source)-[r:cites]->(target)
    """

    paper_has_topic_fieldofstudy_query = """
    UNWIND $edge_list as edge
    MATCH (source: paper{id: toInteger(edge.source)})
    MATCH (target: field_of_study{id: toInteger(edge.target)}) 
    MERGE (source)-[r:has_topic]->(target)
    """

    author_writes_paper_query = """
    UNWIND $edge_list as edge
    MATCH (source: author{id: toInteger(edge.source)})
    MATCH (target: paper{id: toInteger(edge.target)}) 
    MERGE (source)-[r:writes]->(target)
    """

    author_affiliatedwith_institution_query = """
    UNWIND $edge_list as edge
    MATCH (source: author{id: toInteger(edge.source)})
    MATCH (target: paper{id: toInteger(edge.target)}) 
    MERGE (source)-[r:affiliated_with]->(target)
    """

    connection = Neo4jConnection(config)

    if node_type == NodeType.PAPER.value:
        batch_loading_edge_list(connection, edge_list, paper_cites_paper_query)
    elif node_type == NodeType.AUTHOR.value:
        batch_loading_edge_list(connection, edge_list, author_writes_paper_query)
    elif node_type == NodeType.INSTITUTION.value:
        batch_loading_edge_list(connection, edge_list, author_affiliatedwith_institution_query)
    else:
        batch_loading_edge_list(connection, edge_list, paper_has_topic_fieldofstudy_query)

    

def create_nodes_neo4j(node_list, node_type = NodeType.PAPER.value):
    load_papers_query = """
    UNWIND $node_list as node
    CREATE( : paper{
        id: toInteger(node.idx),
        name: toInteger(node.name),
        year: toInteger(node.year), 
        feature: node.feature
    })
    """

    load_author_nodes_query = """
        UNWIND $node_list as node
        CREATE( : author {
            id: toInteger(node.idx),
            name: toInteger(node.name)
        })
    """

    load_institution_nodes_query = """
        UNWIND $node_list as node
        CREATE( : institution {
            id: toInteger(node.idx),
            name: toInteger(node.name)
        })
    """

    load_field_of_study_query = """
        UNWIND $node_list as node
        CREATE( : field_of_study {
            id: toInteger(node.idx),
            name: toInteger(node.name)
        })
    """
    
    connection = Neo4jConnection(config)

    if node_type == NodeType.PAPER.value:
        batch_loading(connection, node_list, load_papers_query)
    elif node_type == NodeType.AUTHOR.value:
        batch_loading(connection, node_list, load_author_nodes_query)
    elif node_type == NodeType.INSTITUTION.value:
        batch_loading(connection, node_list, load_institution_nodes_query)
    else: 
        batch_loading(connection, node_list, load_field_of_study_query)


def batch_loading_edge_list(connection,edge_list,query):
    batch_len = 50000

    for batch_start in range(0, len(edge_list), batch_len):
        batch_end = batch_start + batch_len
        records = edge_list.iloc[batch_start:batch_end].to_dict("records")
        connection.query(query, {"edge_list": records})

def batch_loading(connection,node_list,query):
    batch_len = 5000

    for batch_start in range(0, len(node_list), batch_len):
        batch_end = batch_start + batch_len
        records = node_list.iloc[batch_start:batch_end].to_dict("records")
        connection.query(query, {"node_list": records})


if __name__ == "__main__":
    main()
    