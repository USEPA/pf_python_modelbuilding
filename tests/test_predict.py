import os
from unittest import TestCase, main

from API_Utilities import DescriptorsAPI
from app import predictDB
from model_ws_utilities import call_do_predictions
from models.df_utilities import load_df_from_file

TEST_DATA_FOLDER = "test_data/"
import json

class TestPredict(TestCase):
    def setUp(self):
        self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
        self.descriptors_api = DescriptorsAPI()
        self.serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")

    def test_model_predict_1065(self):
        print('Running test_model_predict_1065')
        r = predictDB(smiles='c1ccccc1', model_id=1065, report_format='json')
        body_bytes = r.body
        generic_response_data = json.loads(body_bytes.decode('utf-8'))    
        print(generic_response_data)        # print(r.json())

    def test_model_predict_1065_array(self):
        
        smiles_list = []
        smiles_list.append("c1ccccc1")  # benzene
        # smiles_list.append("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F") # PFOA
        smiles_list.append("COCOCOCOCCCCCCOCCCCOCOCOCCC") # not in DssTox
        smiles_list.append("CCCCCCCc1ccccc1") # one of neighbors for test set doesnt have matching dtxsid for the dtxcid (not in dsstox_records table)
        smiles_list.append("C[Sb]") # passes standardizer, fails test descriptors
        smiles_list.append("C[As]C[As]C") # violates frag AD
        smiles_list.append("XX")  # fails standardizer
        smiles_list.append("CCC.Cl") # not mixture according to qsarReadySmiles
        smiles_list.append("CCCCC.CCCC") # mixture according to qsarReadySmiles
    
        # For LogP model, following uses additional code to get dsstox record for neighbor / analogs:
        smiles_list.append("[H][C@]12[C@@H](Cl)[C@H](Cl)[C@](C(Cl)Cl)([C@@H](Cl)[C@@H]1Cl)C2(C)C(Cl)Cl") # DTXCID001783033 
        smiles_list.append("[H][C@]12O[C@@]1([H])[C@@]1([H])[C@@]([H])([C@H]2Cl)[C@@]2(Cl)C(Cl)=C(Cl)[C@]1(Cl)C2(Cl)Cl") #DTXCID501783733
        smiles_list.append("[H][C@]12CC(Cl)(Cl)[C@](CCl)([C@@H](Cl)[C@@H]1Cl)C2(CCl)CCl") # DTXCID501782985
        smiles_list.append("[H][C@]12CO[S@@](=O)OC[C@@]1([H])[C@@]1(Cl)C(Cl)=C(Cl)[C@]2(Cl)C1(Cl)Cl") #DTXCID601783831, fails standardization!

        # r = predictDB(smiles = ['c1ccccc1', 'CCCC'], model_id=1065, report_format='json')
        r = predictDB(smiles = smiles_list, model_id=1065, report_format='json')
        body_bytes = r.body
        generic_response_data = json.loads(body_bytes.decode('utf-8'))    

        # print(generic_response_data)        # print(r.json())
        # print(len(smiles_list))
        # print(len(generic_response_data))
        
        self.assertEqual(len(smiles_list),len(generic_response_data))        

    def test_descriptors_get(self):
        r = self.descriptors_api.call_descriptors_get(self.serverAPIs,
                                                      "c1ccccc1",
                                                      "webtest")
        print(r)

    def test_descriptors_post(self):
        r = self.descriptors_api.call_descriptors_post(self.serverAPIs,
                                                       ["c1ccccc1", "CCCC"],
                                                       "webtest")
        print(r)


if __name__ == '__main__':
    main()
