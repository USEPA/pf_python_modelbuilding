import json
import os
import threading

import httpx
import requests
from requests.adapters import HTTPAdapter
from indigo import Indigo

from utils import timer
import numpy as np

"""
This class assumes that server looks like: CIM_API_SERVER=https://hcd.rtpnc.epa.gov

"""

_thread_local = threading.local()


def _get_pool_size(env_var: str, default: int) -> int:
    raw_value = os.getenv(env_var, str(default))
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return default


def get_requests_session():
    session = getattr(_thread_local, "requests_session", None)
    if session is not None:
        return session

    session = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=_get_pool_size("API_HTTP_POOL_CONNECTIONS", 32),
        pool_maxsize=_get_pool_size("API_HTTP_POOL_MAXSIZE", 64),
    )
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    _thread_local.requests_session = session
    return session

class DescriptorsAPI:

    @staticmethod
    def _preview_value(value, max_len: int = 500) -> str:
        if isinstance(value, dict):
            keys = list(value.keys())
            return f"dict keys={keys[:10]}"
        if isinstance(value, list):
            return f"list len={len(value)}"

        text = str(value).replace("\n", " ").strip()
        if len(text) > max_len:
            return text[:max_len] + "..."
        return text

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

    @timer
    def calculate_descriptors_batch(self, serverAPIs, qsarSmilesList, descriptorService):
        qsar_smiles_list = list(qsarSmilesList)

        if not qsar_smiles_list:
            import pandas as pd
            return pd.DataFrame(), 200

        if "test" in descriptorService.lower():
            for qsar_smiles in qsar_smiles_list:
                check_results, code = self.check_structure(qsar_smiles)
                if code != 200:
                    return check_results, code

        response = self.call_descriptors_post(
            server_host=serverAPIs,
            qsar_smiles=qsar_smiles_list,
            descriptor_name=descriptorService,
        )

        response_payload, status_code = response
        if status_code != 200:
            return (
                f"Descriptor batch endpoint returned HTTP {status_code}: "
                f"{self._preview_value(response_payload)}",
                status_code,
            )

        if not isinstance(response_payload, dict):
            return (
                "Descriptor batch endpoint returned unexpected payload type "
                f"{type(response_payload).__name__}: {self._preview_value(response_payload)}",
                500,
            )

        try:
            df_prediction = self.response_json_to_df(response_payload, qsar_smiles_list)
        except Exception as exc:
            return (
                f"Failed to parse descriptor batch response: {exc}; "
                f"payload={self._preview_value(response_payload)}",
                500,
            )

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

        response = get_requests_session().get(url, params=params)

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

        response = get_requests_session().post(url, json=params)

        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json(), response.status_code
        else:
            # Handle the error appropriately
            return response.text, response.status_code

    def response_to_df(self, response, qsarSmiles):
        descriptor_dict = response.json()
        return self.response_json_to_df(descriptor_dict, [qsarSmiles])

    def response_json_to_df(self, descriptor_dict, qsarSmilesList):
        headers = list(descriptor_dict['headers'])
        headers.insert(0, "Property")
        headers.insert(0, "ID")

        chemicals = descriptor_dict['chemicals']
        qsar_smiles_list = list(qsarSmilesList)

        if len(chemicals) != len(qsar_smiles_list):
            raise ValueError(
                f"descriptor response size mismatch: chemicals={len(chemicals)} smiles={len(qsar_smiles_list)}"
            )

        rows = []
        for qsar_smiles, chemical in zip(qsar_smiles_list, chemicals):
            descriptors = [
                float(descriptor) if descriptor is not None else np.nan
                for descriptor in chemical['descriptors']
            ]
            descriptors.insert(0, None)
            descriptors.insert(0, qsar_smiles)
            rows.append(descriptors)

        import pandas as pd
        return pd.DataFrame(rows, columns=headers)


class SearchAPI:
     
    @staticmethod
    def call_resolver_get(server_host, identifier):
        url = f"{server_host}/api/resolver/lookup"
        
        response = get_requests_session().get(url, params={"query": identifier})
        
        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json(), 200
        else:
            # Handle the error appropriately
            return response.text,  response.status_code 
    


class QsarSmilesAPI:

    @staticmethod
    def call_qsar_ready_standardize_post(server_host, smiles, full, workflow):
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            smiles_list = list(smiles)

        # Construct the JSON body
        jo_body = {
            "full": full,
            "options": {"workflow": workflow},
            "chemicals": [{"smiles": current_smiles} for current_smiles in smiles_list]
        }

        # Make the POST request
        url = f"{server_host}/api/stdizer/chemicals"
        response = get_requests_session().post(url, json=jo_body)

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

    
    
