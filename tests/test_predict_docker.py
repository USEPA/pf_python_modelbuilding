'''
Created on Jan 6, 2026

@author: TMARTI02
'''
import unittest
import requests
import json

# from numpy.ma.testutils import assert_equal

host = "http://v2626umcth882.rtord.epa.gov:8080"
# host = "http://localhost:5005"

class Test(unittest.TestCase):

    
    def get_smiles_list(self):
    
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
        return smiles_list


    def test_docker_get(self):
                        
        smiles_list = self.get_smiles_list()
        
        count_failed = 0
        
        for smiles in smiles_list:
            params = {'smiles': {smiles}, 'model_id': '1065','report_format':'json'}       
            url = host + "/api/predictor_models/predict"        
            response = requests.get(url, params) # need to use requests because cant pass parameters otherwise for flask       
            print(f"Status Code: {response.status_code}, smiles: {smiles}")
            
            if response.status_code !=200:
                count_failed+=1
        
        self.assertEqual(count_failed, 0)
    
    def test_docker_post(self):
        
        smiles_list = self.get_smiles_list()
        
        payload = json.dumps({
          "smiles": smiles_list,
          "model_id": 1065
        })
        
        headers = {
          'Content-Type': 'application/json'
        }
        url = host + "/api/predictor_models/predict"
        response = requests.request("POST", url, headers=headers, data=payload)
        self.assertEqual(len(response.json()), len(smiles_list))


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()