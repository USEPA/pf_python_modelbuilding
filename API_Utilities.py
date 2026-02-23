import json

import requests
from indigo import Indigo

from utils import timer
import numpy as np
import logging
import socket
from urllib.parse import urlparse, urljoin


from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


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
    def call_resolver_get(server_host, identifier, timeout=(3.05, 10.0), retries=3):
        """
        GET {server_host}/api/resolver/lookup?query=identifier

        Returns:
          - (payload, 200) on success (payload is JSON if response is JSON; else text)
          - (payload, <status_code>) on non-2xx from upstream
          - (error_dict, 502/504) on DNS/connection/timeout errors
        """
        base = (server_host or "").strip().rstrip("/")
        url = urljoin(base + "/", "api/resolver/lookup")

        # Fail fast if DNS cannot resolve
        host = urlparse(base).hostname or base
        try:
            socket.getaddrinfo(host, 443)
        except socket.gaierror as e:
            msg = f"DNS resolution failed for {host}: {e}"
            logging.error(msg)
            return {"error": "DNS resolution failed", "host": host, "detail": str(e)}, 502

        # Session with retries/backoff on transient errors
        session = requests.Session()
        retry_cfg = Retry(
            total=retries,
            connect=retries,
            read=max(0, retries - 1),
            backoff_factor=0.5,
            status_forcelist=(502, 503, 504),
            allowed_methods=frozenset({"GET"}),
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry_cfg)
        session.mount("https://", adapter)
        session.mount("http://", adapter)

        try:
            resp = session.get(url, params={"query": identifier}, timeout=timeout)
            ctype = resp.headers.get("content-type", "")

            if resp.ok:
                # Prefer JSON if available
                if "application/json" in ctype:
                    return resp.json(), resp.status_code
                try:
                    return resp.json(), resp.status_code
                except ValueError:
                    return resp.text, resp.status_code
            else:
                # Bubble up upstream error payload
                if "application/json" in ctype:
                    try:
                        return resp.json(), resp.status_code
                    except ValueError:
                        return resp.text, resp.status_code
                return resp.text, resp.status_code

        except requests.exceptions.Timeout as e:
            logging.error(f"Resolver timeout calling {url}: {e}", exc_info=True)
            return {"error": "Upstream resolver timed out", "detail": str(e)}, 504

        except requests.exceptions.ConnectionError as e:
            logging.error(f"Resolver connection error calling {url}: {e}", exc_info=True)
            return {"error": "Upstream resolver unreachable", "detail": str(e)}, 502

        except requests.exceptions.RequestException as e:
            logging.error(f"Resolver request failed calling {url}: {e}", exc_info=True)
            return {"error": "Resolver request failed", "detail": str(e)}, 502
    


class QsarSmilesAPI:

    @staticmethod
    def call_qsar_ready_standardize_post(server_host, smiles, full, workflow):
        # Construct the JSON body
        jo_body = {
            "full": full,
            "options": {"workflow": workflow},
            "chemicals": [{"smiles": smiles}]
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

    
    