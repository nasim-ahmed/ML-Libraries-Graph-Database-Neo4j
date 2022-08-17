from neo4j_connections import Neo4jConnection
import yaml
import pandas as pd 
import json

class GraphFetcher(object):
    def __init__(self, configuration, eICU_path):
        self.configuration = configuration 
        self.eICU_path = eICU_path

    def load_config(self):
        global config
        with open(self.configuration, 'r') as config_file:
            config = yaml.load(config_file, Loader = yaml.FullLoader)

    def fetch_labels(self, connection): 
        query = '''
            call {
                match (pa: patient)-[:HAS_STAY]->(p:patientunitstay)-[:HAS_RESULT]->(apr:apachepatientresult)
                where apr.apacheversion = 'IVa' and apr.actualiculos >= 1
                with p.uniquepid as uniquepid, min(p.patienthealthsystemstayid) as patienthealthsystemstayid
                match(p:patientunitstay) 
                where p.patienthealthsystemstayid = patienthealthsystemstayid
                with p.uniquepid as uniquepid, patienthealthsystemstayid, min(p.unitvisitnumber) as unitvisitnumber
                return  patienthealthsystemstayid, unitvisitnumber
            }
            match (pa: patient)-[:HAS_STAY]->(p:patientunitstay)-[:HAS_RESULT]->(apr:apachepatientresult)
            where apr.apacheversion = 'IVa' and apr.actualiculos >= 1
            and p.patienthealthsystemstayid = patienthealthsystemstayid
            and p.unitvisitnumber = unitvisitnumber
            return pa.uniquepid AS uniquepid, p.patientunitstayid AS patientunitstayid, apr.predictedhospitalmortality AS predictedhospitalmortality,
            apr.actualhospitalmortality AS actualhospitalmortality, apr.predictediculos AS predictediculos, apr.actualiculos AS actualiculos 
        '''

        labels_df = connection.query(query)
        #labels_df.to_csv (r'../../../PyG-Neo4j/dataset/eicudata/labels.csv', index = False, header=True)
        labels_df.to_csv(r'{}labels.csv'.format(self.eICU_path), index=False, header=True)


    def fetch_diagnosis(self, connection): 
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_one_list = df['patientunitstayid'].tolist()
        query = '''
        MATCH (pa: patient)-[:HAS_STAY]-> (p:patientunitstay) -[:HAS_DIAGNOSIS] -> (d: diagnosis)
        WHERE d.patientunitstayid IN $col_one_list AND d.diagnosisoffset < 1440 AND pa.uniquepid = p.uniquepid AND d.patientunitstayid = p.patientunitstayid
        return pa.uniquepid AS uniquepid, d.patientunitstayid AS patientunitstayid, d.diagnosisstring AS diagnosisstring
        UNION
        MATCH (pa: patient)-[:HAS_STAY]-> (p:patientunitstay) -[:HAS_PASTHISTORY] -> (ph: pasthistory)
        WHERE ph.patientunitstayid IN $col_one_list AND ph.pasthistoryoffset < 1440 AND pa.uniquepid = p.uniquepid AND ph.patientunitstayid = p.patientunitstayid
        return pa.uniquepid AS uniquepid, ph.patientunitstayid AS patientunitstayid, ph.pasthistorypath AS diagnosisstring
        UNION
        MATCH (pa: patient)-[:HAS_STAY]-> (p:patientunitstay) -[:HAS_ADMISSION] -> (ad: admissiondx)
        WHERE ad.patientunitstayid IN $col_one_list AND ad.admitdxenteredoffset < 1440 AND pa.uniquepid = p.uniquepid AND ad.patientunitstayid = p.patientunitstayid
        return pa.uniquepid AS uniquepid, ad.patientunitstayid AS patientunitstayid, ad.admitdxpath AS diagnosisstring
        '''

        diagnosis_df = connection.query(query, {"col_one_list": col_one_list})
        diagnosis_df.to_csv (r'{}diagnoses.csv'.format(self.eICU_path), index = False, header=True)

    def flat_features(self, connection):
        df = pd.read_csv('../../../PyG-Neo4j/dataset/eicudata/labels.csv')
        col_one_list = df['patientunitstayid'].tolist()

        query = '''
        MATCH (p:patientunitstay) -[:HAS_APACHEAPS] -> (aps: apacheapsvar)
        MATCH (p:patientunitstay) -[:HAS_RESULT] -> (apr: apachepatientresult)
        MATCH (pat: patient)-[:HAS_STAY] -> (p:patientunitstay)
        WHERE p.patientunitstayid IN $col_one_list
        return distinct p.patientunitstayid AS patientunitstayid, pat.gender AS gender, pat.age AS age, pat.ethnicity AS ethnicity, p.admissionheight AS admissionheight, p.admissionweight AS admissionweight,
        p.apacheadmissiondx AS apacheadmissiondx, substring(p.unitadmittime24, 0,2) as hr, p.unittype AS unittype, p.unitadmitsource AS unitadmitsource, p.unitstaytype AS unitstaytype, apr.physicianspeciality AS physicianspeciality, aps.intubated AS intubated, aps.vent AS vent, aps.dialysis AS dialysis, aps.eyes AS eyes,
        aps.motor AS motor, aps.verbal AS verbal, aps.meds AS meds
        '''

        flat_features_df = connection.query(query, {"col_one_list": col_one_list})
        flat_features_df.to_csv (r'../../../PyG-Neo4j/dataset/eicudata/flat_features.csv', index = False, header=True)

        


def main():
    with open('paths.json', 'r') as f:
        eICU_path = json.load(f)["eICU_path"]

    configure = 'config.yaml'
    graphFetcher = GraphFetcher(configure, eICU_path)
    graphFetcher.load_config()
    connection = Neo4jConnection(config)


    '''Fetch the labels table'''
    #graphFetcher.fetch_labels(connection)
    graphFetcher.fetch_diagnosis(connection)
    #graphFetcher.flat_features(connection)


if __name__ == "__main__":
    main()