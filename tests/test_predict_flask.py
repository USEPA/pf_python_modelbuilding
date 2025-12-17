import os
from unittest import TestCase, main

from API_Utilities import DescriptorsAPI
from app_flask import predictDB, info, details, model_coeffs, available_models
from model_ws_utilities import call_do_predictions
from models.df_utilities import load_df_from_file

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
        
    
    def test_predictDB_get(self):
        url = host+"/api/predictor_models/models/predictDB"        
        r = requests.get(url, params={
            'smiles': 'C1CCCCC1',
            'model_id': '1065'})        
        print(r.json())
        
    def test_predictDB_html_get(self):
        url = host+"/api/predictor_models/models/predictDB"        
        r = requests.get(url, params={
            'smiles': 'C1CCCCC1',
            'model_id': '1065',
            'report_format': 'html'})        
        print(r)
    
    # def setUp(self):
    #     self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
    #     self.descriptors_api = DescriptorsAPI()
    #     self.serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")
    #
    # def test_model_predict_1065(self):
    #     r = predictDB('c1ccccc1', 1065)
    #     print(r)
    #
    # def test_model_predict_1065_array(self):
    #     r = predictDB(['c1ccccc1', 'CCCC'], 1065)
    #     print(r)
    #
    # def test_descriptors_get(self):
    #     r = self.descriptors_api.call_descriptors_get(self.serverAPIs,
    #                                                   "c1ccccc1",
    #                                                   "webtest")
    #     print(r)
    #
    # def test_descriptors_post(self):
    #     r = self.descriptors_api.call_descriptors_post(self.serverAPIs,
    #                                                    ["c1ccccc1", "CCCC"],
    #                                                    "webtest")
    #     print(r)


if __name__ == '__main__':
    main()
