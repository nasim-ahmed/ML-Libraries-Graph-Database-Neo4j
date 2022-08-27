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
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_one_list = df['patientunitstayid'].tolist()

        query = '''
        MATCH (p:patientunitstay) -[:HAS_APACHEAPS] -> (aps: apacheapsvar)
        MATCH (p:patientunitstay) -[:HAS_RESULT] -> (apr: apachepatientresult)
        MATCH (pat: patient)-[:HAS_STAY] -> (p:patientunitstay)
        WHERE p.patientunitstayid IN $col_one_list
        return distinct p.patientunitstayid AS patientunitstayid, pat.uniquepid AS uniquepid, pat.gender AS gender, pat.age AS age, pat.ethnicity AS ethnicity, p.admissionheight AS admissionheight, p.admissionweight AS admissionweight,
        p.apacheadmissiondx AS apacheadmissiondx, substring(p.unitadmittime24, 0,2) as hr, p.unittype AS unittype, p.unitadmitsource AS unitadmitsource, p.unitstaytype AS unitstaytype, apr.physicianspeciality AS physicianspeciality, aps.intubated AS intubated, aps.vent AS vent, aps.dialysis AS dialysis, aps.eyes AS eyes,
        aps.motor AS motor, aps.verbal AS verbal, aps.meds AS meds
        '''

        flat_features_df = connection.query(query, {"col_one_list": col_one_list})
        flat_features_df.to_csv (r'{}flat_features.csv'.format(self.eICU_path), index = False, header=True)

    def commonLabs(self, connection):
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_one_list = df['patientunitstayid'].tolist()

        query = '''
         MATCH (p:patientunitstay) - [r:HAS_LAB] -> (l:lab)
         WHERE r.labresultoffset >= -1440 AND r.labresultoffset <= 1440 AND p.patientunitstayid IN $col_one_list 
         WITH l.labname AS labname, count(distinct r.patientunitstayid) AS count
         WHERE count > 0.25 * 89143
         RETURN labname, count
         ORDER BY count DESC;
        '''

        commonlabs_df = connection.query(query, {"col_one_list": col_one_list})
        commonlabs_df.to_csv (r'{}commonlabs.csv'.format(self.eICU_path), index = False, header=True)

    def timeserieslab(self, connection):
        labels_df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        patientunitstayid_list = labels_df['patientunitstayid'].tolist()
        commonlabs_df = pd.read_csv('{}commonlabs.csv'.format(self.eICU_path))
        labname_list = commonlabs_df['labname'].tolist()
        
        query = '''
         MATCH (p:patientunitstay) - [r:HAS_LAB] -> (l:lab)
         WHERE r.patientunitstayid IN $patientunitstayid_list AND l.labname IN $labname_list AND r.labresultoffset >= -1440 AND r.labresultoffset <= 1440
         RETURN p.uniquepid AS uniquepid, p.patientunitstayid AS patientunitstayid, r.labresultoffset AS labresultoffset, l.labname AS labname, r.labresult AS labresult
        '''

        timeserieslab_df = connection.query(query, {"patientunitstayid_list": patientunitstayid_list, "labname_list": labname_list})
        #Replace not available values with 0.0
        timeserieslab_df["labresult"].replace({"not available": 0.0}, inplace=True)
        timeserieslab_df.to_csv (r'{}timeserieslab.csv'.format(self.eICU_path), index = False, header=True)

    def commonresp(self, connection):
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_one_list = df['patientunitstayid'].tolist()

        query = '''
         MATCH (p:patientunitstay) - [r:HAS_RESPIRATORY_CHARTING] -> (rc:respiratorycharting)
         WHERE rc.respchartoffset >= -1440 AND rc.respchartoffset <= 1440 AND p.patientunitstayid IN $col_one_list 
         WITH rc.respchartvaluelabel AS respchartvaluelabel, count(distinct rc.patientunitstayid) AS count
         WHERE count > 0.13 * 89143
         RETURN respchartvaluelabel, count
         ORDER BY count DESC;
        '''

        commonresp_df = connection.query(query, {"col_one_list": col_one_list})
        commonresp_df.to_csv (r'{}commonresp.csv'.format(self.eICU_path), index = False, header=True)

    def timeseriesresp(self, connection):
        labels_df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        patientunitstayid_list = labels_df['patientunitstayid'].tolist()
        commonresp_df = pd.read_csv('{}commonresp.csv'.format(self.eICU_path))
        commonresp_list = commonresp_df['respchartvaluelabel'].tolist()
        
        query = '''
         MATCH (p:patientunitstay) - [r:HAS_RESPIRATORY_CHARTING] -> (rc:respiratorycharting)
         WHERE rc.patientunitstayid IN $patientunitstayid_list AND rc.respchartvaluelabel IN $commonresp_list AND rc.respchartoffset >= -1440 AND rc.respchartoffset <= 1440
         RETURN p.uniquepid AS uniquepid, rc.patientunitstayid AS patientunitstayid, rc.respchartoffset AS respchartoffset, rc.respchartvaluelabel AS respchartvaluelabel, rc.respchartvalue AS respchartvalue
        '''

        timeseriesresp_df = connection.query(query, {"patientunitstayid_list": patientunitstayid_list, "commonresp_list": commonresp_list})
        timeseriesresp_df.to_csv (r'{}timeseriesresp.csv'.format(self.eICU_path), index = False, header=True)

    def timeseriesperiodic(self, connection):
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_list = df['patientunitstayid'].tolist()

        query = '''
         MATCH (p:patientunitstay) - [v:HAS_VITALPERIODIC] -> (vp:vitalperiodic)
         WHERE vp.observationoffset >= -1440 AND vp.observationoffset <= 1440 AND vp.patientunitstayid IN $col_one_list
         RETURN p.uniquepid AS uniquepid, vp.patientunitstayid AS patientunitstayid, vp.observationoffset as observationoffset, vp.temperature AS temperature, vp.sao2 AS sao2, vp.heartrate AS heartrate, vp.respiration AS respiration, vp.cvp AS cvp,
    vp.systemicsystolic AS systemicsystolic, vp.systemicdiastolic AS systemicdiastolic, vp.systemicmean AS systemicmean, vp.st1 AS st1, vp.st2 AS st2, vp.st3 AS st3
         ORDER BY patientunitstayid, observationoffset;
        '''

        batch_len = 5000

        largeDF = pd.DataFrame()

        for batch_start in range(0, len(df), batch_len):
            batch_end = batch_start + batch_len
            records = col_list[batch_start:batch_end]
            timeseriesperiodic_df = connection.query(query, {"col_one_list": records})
            largeDF = pd.concat([largeDF,timeseriesperiodic_df])
            print(largeDF)
        largeDF.to_csv (r'{}timeseriesperiodic.csv'.format(self.eICU_path), index = False, header=True)

    def timeseriesaperiodic(self, connection):
        df = pd.read_csv('{}labels.csv'.format(self.eICU_path))
        col_list = df['patientunitstayid'].tolist()

        query = '''
         MATCH (p:patientunitstay) - [v:HAS_VITALAPERIODIC] -> (va:vitalaperiodic)
         WHERE va.observationoffset >= -1440 AND va.observationoffset <= 1440 AND va.patientunitstayid IN $col_one_list
         RETURN p.uniquepid AS uniquepid, va.patientunitstayid AS patientunitstayid, va.observationoffset AS observationoffset, va.noninvasivesystolic AS noninvasivesystolic, va.noninvasivediastolic AS noninvasivediastolic, va.noninvasivemean AS noninvasivemean
         ORDER BY patientunitstayid, observationoffset;
        '''

        batch_len = 5000

        largeDF = pd.DataFrame()

        for batch_start in range(0, len(df), batch_len):
            batch_end = batch_start + batch_len
            records = col_list[batch_start:batch_end]
            timeseriesaperiodic_df = connection.query(query, {"col_one_list": records})
            largeDF = pd.concat([largeDF, timeseriesaperiodic_df])
            print(largeDF)
        largeDF.to_csv (r'{}timeseriesaperiodic.csv'.format(self.eICU_path), index = False, header=True)

def main():
    #for debugging
    eICU_path = "/media/nasim/31c299f0-f952-4032-9bd8-001b141183e0/ML-Libraries-Graph-Database-Neo4j/PyG-Neo4j/app/eICU_data/"
    configure = '/media/nasim/31c299f0-f952-4032-9bd8-001b141183e0/ML-Libraries-Graph-Database-Neo4j/PyG-Neo4j/thesis/preprocessing/config.yaml'

    # with open('paths.json', 'r') as f:
    #     eICU_path = json.load(f)["eICU_path"]
    # configure = 'config.yaml'
   
    graphFetcher = GraphFetcher(configure, eICU_path)
    graphFetcher.load_config()
    connection = Neo4jConnection(config)


    '''Fetch the labels table'''
    #graphFetcher.fetch_labels(connection)
    #graphFetcher.fetch_diagnosis(connection)
    #graphFetcher.flat_features(connection)
    #graphFetcher.commonLabs(connection)
    #graphFetcher.timeserieslab(connection)
    #graphFetcher.commonresp(connection)
    #graphFetcher.timeseriesresp(connection)
    #graphFetcher.timeseriesperiodic(connection)
    #graphFetcher.timeseriesaperiodic(connection)


if __name__ == "__main__":
    main()