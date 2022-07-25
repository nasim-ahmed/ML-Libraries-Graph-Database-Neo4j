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

    def fetch_labels(self, connection): 
        query = '''
            call {
                match(p:patientunitstay)-[:HAS_RESULT]->(apr:apachepatientresult)
                where apr.apacheversion = 'IVa' and apr.actualiculos >= 1
                with p.uniquepid as uniquepid, min(p.patienthealthsystemstayid) as patienthealthsystemstayid
                match(p:patientunitstay) 
                where p.patienthealthsystemstayid = patienthealthsystemstayid
                with p.uniquepid as uniquepid, patienthealthsystemstayid, min(p.unitvisitnumber) as unitvisitnumber
                return  patienthealthsystemstayid, unitvisitnumber
            }
            match(p:patientunitstay)-[:HAS_RESULT]->(apr:apachepatientresult)
            where apr.apacheversion = 'IVa' and apr.actualiculos >= 1
            and p.patienthealthsystemstayid = patienthealthsystemstayid
            and p.unitvisitnumber = unitvisitnumber
            return p.patientunitstayid AS patientunitstayid, apr.predictedhospitalmortality AS predictedhospitalmortality,
            apr.actualhospitalmortality AS actualhospitalmortality, apr.predictediculos AS predictediculos, apr.actualiculos AS actualiculos 
        '''

        labels_df = connection.query(query)
        labels_df.to_csv (r'labels.csv', index = False, header=True)


    def fetch_diagnosis(self, connection): 
        df = pd.read_csv('labels.csv')
        col_one_list = df['patientunitstayid'].tolist()
        query = '''
        MATCH (p:patientunitstay) -[:HAS_DIAGNOSIS] -> (d: diagnosis)
        WHERE d.patientunitstayid IN $col_one_list AND d.diagnosisoffset < 1440
        return d.patientunitstayid AS patientunitstayid, d.diagnosisstring AS diagnosisstring
        '''

        result = connection.query(query, {"col_one_list": col_one_list})
        print(result['patientunitstayid'].nunique())


       


def main():
    configure = 'config.yaml'
    graphFetcher = GraphFetcher(configure)
    graphFetcher.load_config()
    connection = Neo4jConnection(config)

    '''Fetch the labels table'''
    #graphFetcher.fetch_labels(connection)
    graphFetcher.fetch_diagnosis(connection)
    

    

if __name__ == "__main__":
    main()