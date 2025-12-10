import json
import requests
from indigo import Indigo

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
        return "ok",200

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

    def calculate_descriptors(self, serverAPIs, qsarSmiles, descriptorService):

        if "test" in descriptorService.lower():
            check_results, code = self.check_structure(qsarSmiles)
            if code != 200:
                return check_results, code

        descriptorsResults = self.call_descriptors_get(server_host=serverAPIs, qsar_smiles=qsarSmiles,
                                                                descriptor_name=descriptorService)
        if "error" in descriptorsResults:
            return descriptorsResults, 400

        df_prediction = self.response_to_df(descriptorsResults, qsarSmiles)
        return df_prediction, 200


    def call_descriptors_get(self, server_host, qsar_smiles, descriptor_name):
        # Construct the URL
        url = f"{server_host}/api/descriptors"

        # Set up query parameters
        params = {
            "type": descriptor_name,
            "smiles": qsar_smiles,
            "headers": "true"  # some descriptors dont have header option? Should be fixed so this doesnt cause issue if must be false
        }

        response = requests.get(url, params=params)

        if response.status_code == 200:
            # Parse the response JSON and convert it to a list of Chemical objects
            return response.json()
        else:
            # Handle the error appropriately
            return response.text


    def response_to_df(self, descriptor_dict, qsarSmiles):
        headers = descriptor_dict['headers']
        headers.insert(0, "Property")
        headers.insert(0, "ID")

        chemicals = descriptor_dict['chemicals']
        chemical = chemicals[0]
        descriptors = chemical['descriptors']
        descriptors.insert(0, None)
        descriptors.insert(0, qsarSmiles)

        # print(headers)
        # print(descriptors)
        import pandas as pd
        df = pd.DataFrame([descriptors], columns=headers)
        # print(df.shape)
        return df

class QsarSmilesAPI:

    def call_qsar_ready_standardize_post(self, server_host, smiles, full, workflow):
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
            return response.json()
        else:
            # Handle the error appropriately
            return response.text

