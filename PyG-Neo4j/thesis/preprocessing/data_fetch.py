from neo4j_connections import Neo4jConnection
import yaml
import pandas as pd 

class GraphFetcher(object):
    def __init__(self, configuration):
        self.configuration = configuration 

    def load_config(self):
        global config
        with open(self.configuration, 'r') as config_file:
            config = yaml.load(config_file, Loader = yaml.FullLoader)


def main():
    configure = 'config.yaml'
    graphFetcher = GraphFetcher(configure)
    graphFetcher.load_config()
    connection = Neo4jConnection(config)
    query = '''
      MATCH (n:DRUGS) RETURN n.name LIMIT 25
    '''
    df = connection.query(query)
    print(df)

    

if __name__ == "__main__":
    main()