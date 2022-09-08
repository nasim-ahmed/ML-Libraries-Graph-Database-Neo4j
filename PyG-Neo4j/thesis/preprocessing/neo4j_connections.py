from neo4j import GraphDatabase
import pandas as pd
import yaml
import sys
import torch
#import torch_geometric.transforms as T
#from torch_geometric.nn import SAGEConv, to_hetero
#from torch_geometric.data import HeteroData
#from torch_geometric.transforms import ToUndirected, RandomLinkSplit
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
