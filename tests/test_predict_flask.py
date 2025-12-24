from unittest import TestCase, main
# import os

# from API_Utilities import DescriptorsAPI
# from model_ws_utilities import call_do_predictions
# from models.df_utilities import load_df_from_file

from app_flask import predictDB, info, details, model_coeffs, available_models

TEST_DATA_FOLDER = "test_data/"
host = "http://localhost:5004"
import requests


class TestPredict(TestCase):
    
    def test_info(self): 
        r = info("rf")
        print("\ninfo")
        print(r)
    
    def test_details(self):
        r = details("1065")
        print("\ndetails")
        print(r)
    
    def test_coeffs(self):
        r = model_coeffs("1615")
        print("\nmodel_coeffs")
        print(r)
    
    def test_available_models(self):
        r = available_models()
        print("\navailable models")
        print(r)
                    
    def test_predict_examples(self):
        smiles_list = []
        smiles_list.append("c1ccccc1")  # benzene
        smiles_list.append("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F") # PFOA
        smiles_list.append("COCOCOCOCCCCCCOCCCCOCOCOCCC") # not in DssTox
        smiles_list.append("CCCCCCCc1ccccc1") # for some reason only has 9 neighbors for test set
        smiles_list.append("C[Sb]") # passes standardizer, fails test descriptors
        smiles_list.append("C[As]C[As]C") # violates frag AD
        smiles_list.append("XX")  # fails standardizer
        smiles_list.append("CCC.Cl") # not mixture according to qsarReadySmiles
        smiles_list.append("CCCCC.CCCC") # mixture according to qsarReadySmiles
        
        for smiles in smiles_list:
            params = {'smiles': {smiles}, 'model_id': '1065','report_format':'json'}       
            url = host + "/api/predictor_models/models/predictDB"        
            r = requests.get(url, params) # need to use requests because cant pass parameters otherwise for flask       
            print(r.json())
            
    def test_predictDB_html_get(self):
        url = host + "/api/predictor_models/models/predictDB"
        params = {'smiles': "C1CCCCC1", 'model_id': '1065','report_format':'html'}        
        r = requests.get(url, params)        
        print(r.status_code)
    
    # def setUp(self):
    #     self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
    #     self.descriptors_api = DescriptorsAPI()
    #     self.serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
    #

if __name__ == '__main__':
    main()
