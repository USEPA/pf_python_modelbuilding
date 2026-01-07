from unittest import TestCase, main
import requests
# import os

# from API_Utilities import DescriptorsAPI
# from model_ws_utilities import call_do_predictions
# from models.df_utilities import load_df_from_file

from app_flask import predictDB, info, details, model_coeffs, available_models

TEST_DATA_FOLDER = "test_data/"
host = "http://localhost:5004"


from dotenv import load_dotenv
load_dotenv()


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
                    
    def test_predict_examples_lookup_missing_dsstox(self):
        # there were a handful of datapoints where the dsstox info isnt in dsstox_records table:

        # CC1(C2C(Cl)C(Cl)C1(C(Cl)C2Cl)C(Cl)Cl)C(Cl)Cl    DTXCID001783033    LogP v1 modeling
        # ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl    DTXCID501783733    LogP v1 modeling
        # ClC1(Cl)CC2C(Cl)C(Cl)C1(CCl)C2(CCl)CCl    DTXCID501782985    LogP v1 modeling
        # O=[SH+]1OCC2C(CO1)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl    DTXCID601783831    LogP v1 modeling
        # C1CCCC2CCCCC21    DTXCID701521422    BP v1 modeling
        # CC1CCCC1CC    DTXCID201784601    BP v1 modeling
        # O=[SH+]1OCC2C(CO1)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl    DTXCID601783831    BP v1 modeling
        # ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl    DTXCID501783733    HLC v1 modeling
        # CC1CCCC1CC    DTXCID201784601    MP v1 modeling
        # ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl    DTXCID501783733    MP v1 modeling
        # CN1CC2(C=C)C3C1C1COC(CC21)C13C(=O)NC2=CC=CC=C12    DTXCID501782911    MP v1 modeling
        # C1CCCC2CCCCC21    DTXCID701521422    VP v1 modeling
        # CC1CCCC1CC    DTXCID201784601    VP v1 modeling
        # ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl    DTXCID501783733    VP v1 modeling
        # O=[SH+]1OCC2C(CO1)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl    DTXCID401783809    VP v1 modeling
        # ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl    DTXCID501783733    WS v1 modeling
        # O=[SH+]1OCC2C(CO1)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl    DTXCID401783809    WS v1 modeling
        
        smiles_list = []
        
        # For LogP model, following need help to get dsstox record for neighbor / analogs:
        smiles_list.append("CC1(C2C(Cl)C(Cl)C1(C(Cl)C2Cl)C(Cl)Cl)C(Cl)Cl") # DTXCID001783033 
        smiles_list.append("ClC1(Cl)C2(Cl)C3C4OC4C(Cl)C3C1(Cl)C(Cl)=C2Cl") #DTXCID501783733
        smiles_list.append("ClC1(Cl)CC2C(Cl)C(Cl)C1(CCl)C2(CCl)CCl") # DTXCID501782985
        smiles_list.append("O=[SH+]1OCC2C(CO1)C1(Cl)C(Cl)=C(Cl)C2(Cl)C1(Cl)Cl") #DTXCID601783831
        
        print("\ntest_predict_examples_lookup_missing_dsstox")
        for smiles in smiles_list:
            params = {'smiles': {smiles}, 'model_id': '1069','report_format':'json'} #1069 = LogP model       
            url = host + "/api/predictor_models/models/predictDB"        
            r = requests.get(url, params) # need to use requests because cant pass parameters otherwise for flask       
            print(r.json())
        
    
    def test_predict_examples(self):
        smiles_list = []
        smiles_list.append("c1ccccc1")  # benzene
        smiles_list.append("OC(=O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F") # PFOA
        smiles_list.append("COCOCOCOCCCCCCOCCCCOCOCOCCC") # not in DssTox
        smiles_list.append("CCCCCCCc1ccccc1") # one of neighbors for test set doesnt have matching dtxsid for the dtxcid (not in dsstox_records table)
        smiles_list.append("C[Sb]") # passes standardizer, fails test descriptors
        smiles_list.append("C[As]C[As]C") # violates frag AD
        smiles_list.append("XX")  # fails standardizer
        smiles_list.append("CCC.Cl") # not mixture according to qsarReadySmiles
        smiles_list.append("CCCCC.CCCC") # mixture according to qsarReadySmiles
        
        
        print("\ntest_predict_examples")
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
