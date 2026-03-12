import json

import httpx
import requests
from indigo import Indigo

from utils import timer
import numpy as np

"""
This class assumes that server looks like: CIM_API_SERVER=https://hcd.rtpnc.epa.gov

"""

class DescriptorsAPI:

    def check_structure(self, qsarSmiles):
        indigo = Indigo()
        molecule = indigo.loadMolecule(qsarSmiles)
        if self.contains_unexpected_elements(molecule):
            return qsarSmiles + ": Molecule contains unsupported element", 400
        if molecule.countAtoms() == 1:
            return "Only one non-hydrogen atom", 400
        if molecule.countAtoms() == 0:
            return "Number of atoms equals zero", 400
        if not self.contains_carbon(molecule):
            return "Molecule does not contain carbon", 400
        return "ok", 200

    def contains_carbon(self, molecule):

        # Iterate over atoms in the molecule
        for atom in molecule.iterateAtoms():
            if atom.symbol() == "C":
                return True  # Return True if a carbon atom is found

        return False  # Return F

    def contains_unexpected_elements(self, molecule):

        # Define the set of allowed elements
        allowed_elements = {"C", "H", "O", "N", "F", "Cl", "Br", "I", "S", "P", "Si", "As", "Hg", "Sn"}

        # Use a set to store unique elements
        unique_elements = set()

        # Iterate over atoms in the molecule
        for atom in molecule.iterateAtoms():
            element = atom.symbol()
            unique_elements.add(element)

        # Check if there are any elements not in the allowed set
        for element in unique_elements:
            if element not in allowed_elements:
                return True

        return False

    @timer
    def calculate_descriptors(self, serverAPIs, qsarSmiles, descriptorService):

        if "test" in descriptorService.lower():
            check_results, code = self.check_structure(qsarSmiles)
            if code != 200:
                return check_results, code

        
        
        response = self.call_descriptors_get(server_host=serverAPIs, qsar_smiles=qsarSmiles,
                                                                descriptor_name=descriptorService)
        if response.status_code != 200:
            return response.text,response.status_code 
        
        df_prediction = self.response_to_df(response, qsarSmiles)
                
        return df_prediction, 200

    def call_descriptors_get(self, server_host: str, qsar_smiles: str, descriptor_name: str):
        # Construct the URL
        url = f"{server_host}/api/descriptors"

        # Set up query parameters
        params = {
            "type": descriptor_name,
            "smiles": qsar_smiles,
            "headers": "true"
            # some descriptors dont have header option? Should be fixed so this doesnt cause issue if must be false
        }

        response = requests.get(url, params=params)

        return response 

    def call_descriptors_post(self, server_host: str, qsar_smiles: list[str], descriptor_name: str):
        # Construct the URL
        url = f"{server_host}/api/descriptors"

        # Set up query parameters
        params = {
            "type": descriptor_name,
            "chemicals": qsar_smiles,
            "headers": "true"
            # some descriptors dont have header option? Should be fixed so this doesnt cause issue if must be false
        }

        response = requests.post(url, json=params)

        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json()
        else:
            # Handle the error appropriately
            return response.text

    def call_descriptors_post_with_status(self, server_host: str, qsar_smiles: list[str], descriptor_name: str):
        url = f"{server_host}/api/descriptors"
        params = {
            "type": descriptor_name,
            "chemicals": qsar_smiles,
            "headers": "true"
        }

        response = requests.post(url, json=params)
        if response.status_code == 200:
            return response.json(), 200
        return response.text, response.status_code

    async def call_descriptors_post_with_status_async(self, client: httpx.AsyncClient, server_host: str, qsar_smiles: list[str], descriptor_name: str):
        url = f"{server_host}/api/descriptors"
        params = {
            "type": descriptor_name,
            "chemicals": qsar_smiles,
            "headers": "true"
        }

        response = await client.post(url, json=params)
        if response.status_code == 200:
            return response.json(), 200
        return response.text, response.status_code

    def response_json_to_df(self, descriptor_dict, qsarSmiles, chemical_index=0):
        headers = descriptor_dict['headers']
        headers.insert(0, "Property")
        headers.insert(0, "ID")

        chemicals = descriptor_dict['chemicals']
        chemical = chemicals[chemical_index]

        descriptors = [float(descriptor) if descriptor is not None else np.nan for descriptor in chemical['descriptors']]

        descriptors.insert(0, None)
        descriptors.insert(0, qsarSmiles)

        import pandas as pd
        return pd.DataFrame([descriptors], columns=headers)

    def response_json_to_dfs(self, descriptor_dict, qsar_smiles_list, descriptor_headers=None):
        headers = list(descriptor_dict.get('headers') or [])
        chemicals = list(descriptor_dict.get('chemicals') or [])

        if len(chemicals) != len(qsar_smiles_list):
            return None

        fallback_headers = list(descriptor_headers or [])

        if not headers and fallback_headers:
            headers = fallback_headers

        headers.insert(0, "Property")
        headers.insert(0, "ID")

        import pandas as pd

        dfs = []
        for qsar_smiles, chemical in zip(qsar_smiles_list, chemicals):
            # Some descriptor service responses include per-chemical errors without "descriptors".
            # In that case return None for this row so caller can mark only that SMILES as failed.
            if not isinstance(chemical, dict) or 'descriptors' not in chemical:
                dfs.append(None)
                continue

            descriptors = [float(descriptor) if descriptor is not None else np.nan for descriptor in chemical['descriptors']]
            descriptors.insert(0, None)
            descriptors.insert(0, qsar_smiles)

            if len(descriptors) != len(headers):
                if fallback_headers and len(descriptors) == len(fallback_headers) + 2:
                    active_headers = ["ID", "Property", *fallback_headers]
                else:
                    return None
            else:
                active_headers = headers

            dfs.append(pd.DataFrame([descriptors], columns=active_headers))

        return dfs

    def response_to_df(self, response, qsarSmiles):
        
        descriptor_dict = response.json()
        
        
        headers = descriptor_dict['headers']
        headers.insert(0, "Property")
        headers.insert(0, "ID")

        chemicals = descriptor_dict['chemicals']
        chemical = chemicals[0]
        
        # descriptors = chemical['descriptors']
        
        # for some reason they were getting stored as strings when I was trying to access them later- will this break null descriptors for ones like padel or mordred
        descriptors = [float(descriptor) if descriptor is not None else np.nan for descriptor in chemical['descriptors']]
        
        # for descriptor in descriptors:
        #     print(type(descriptor), descriptor)
                
        # print(descriptors)
        
        descriptors.insert(0, None)
        descriptors.insert(0, qsarSmiles)

        # print(headers)
        # print(descriptors)
        import pandas as pd
        df = pd.DataFrame([descriptors], columns=headers)
        # print(df.shape)
        return df


class SearchAPI:
     
    @staticmethod
    def call_resolver_get(server_host, identifier):
        url = f"{server_host}/api/resolver/lookup"
        
        response = requests.get(url, params={"query": identifier})
        
        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json(), 200
        else:
            # Handle the error appropriately
            return response.text,  response.status_code 
    


class QsarSmilesAPI:

    @staticmethod
    def call_qsar_ready_standardize_post(server_host, smiles, full, workflow):
        if isinstance(smiles, (list, tuple)):
            chemicals_payload = [{"smiles": s} for s in smiles]
        else:
            chemicals_payload = [{"smiles": smiles}]

        # Construct the JSON body
        jo_body = {
            "full": full,
            "options": {"workflow": workflow},
            "chemicals": chemicals_payload
        }
        json_body = json.dumps(jo_body)

        # Make the POST request
        headers = {"Content-Type": "application/json"}
        url = f"{server_host}/api/stdizer/chemicals"
        response = requests.post(url, headers=headers, data=json_body)

        # print(response.text)

        # Check if the request was successful
        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json(), 200
        else:
            # Handle the error appropriately
            return response.text,  response.status_code

    @staticmethod
    async def call_qsar_ready_standardize_post_async(client: httpx.AsyncClient, server_host, smiles, full, workflow):
        if isinstance(smiles, (list, tuple)):
            chemicals_payload = [{"smiles": s} for s in smiles]
        else:
            chemicals_payload = [{"smiles": smiles}]

        jo_body = {
            "full": full,
            "options": {"workflow": workflow},
            "chemicals": chemicals_payload
        }

        headers = {"Content-Type": "application/json"}
        url = f"{server_host}/api/stdizer/chemicals"
        response = await client.post(url, headers=headers, json=jo_body)

        if response.status_code == 200:
            return response.json(), 200
        return response.text, response.status_code




if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    import os
    serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com")
    identifier='71-43-2X'
    
    chemicals, code = SearchAPI.call_resolver_get(serverAPIs, identifier)
    
    print(chemicals, code)
    
    if code == 200:
        for chemical in chemicals:
            print(json.dumps(chemical))
    else:
        print(chemicals)

    
    