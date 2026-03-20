import json
import logging
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

        try:
            df_prediction = self.response_to_df(response, qsarSmiles)
        except Exception as exc:
            return (
                f"Failed to parse descriptor response for smiles={self._preview_value(qsarSmiles)}: {exc}; "
                f"payload={self._preview_value(response.text)}",
                500,
            )
                
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
            "headers": True,
            # some descriptors dont have header option? Should be fixed so this doesnt cause issue if must be false
        }

        response = get_requests_session().get(url, params=params)

        return response 

    def _diagnose_descriptor_batch_400(self, url: str, descriptor_name: str, qsar_smiles: list[str], max_depth: int = 8):
        session = get_requests_session()

        def _payload(smiles_subset):
            return {
                "type": descriptor_name,
                "chemicals": list(smiles_subset),
                "chemIdType": "SMILES",
                "format": "JSON",
                "options": {
                    "headers": True,
                },
            }

        def _probe(smiles_subset):
            response = session.post(url, json=_payload(smiles_subset))
            return response.status_code, self._preview_value(response.text)

        def _walk(smiles_subset, depth):
            if not smiles_subset:
                return "empty_subset"

            if len(smiles_subset) == 1 or depth >= max_depth:
                status_code, preview = _probe(smiles_subset)
                if len(smiles_subset) == 1:
                    sample_preview = self._preview_value(smiles_subset[0])
                else:
                    sample_preview = self._preview_value(smiles_subset)
                return (
                    f"subset_size={len(smiles_subset)} status={status_code} "
                    f"sample={sample_preview} body={preview}"
                )

            mid = len(smiles_subset) // 2
            left = smiles_subset[:mid]
            right = smiles_subset[mid:]

            left_status, left_preview = _probe(left)
            right_status, right_preview = _probe(right)

            if left_status == 400 and right_status != 400:
                return "left_failed -> " + _walk(left, depth + 1)

            if right_status == 400 and left_status != 400:
                return "right_failed -> " + _walk(right, depth + 1)

            if left_status == 400 and right_status == 400:
                return (
                    "both_halves_failed -> "
                    f"left_size={len(left)} body={left_preview} | "
                    f"right_size={len(right)} body={right_preview}"
                )

            return (
                "whole_batch_failed_but_halves_passed -> likely batch_size_limit_or_payload_size_or_bad_combination; "
                f"left_size={len(left)} status={left_status} | right_size={len(right)} status={right_status}"
            )

        try:
            return _walk(list(qsar_smiles), 0)
        except Exception as exc:
            return f"diagnostic_failed={exc}"

    def call_descriptors_post(self, server_host: str, qsar_smiles: list[str], descriptor_name: str):
        # Construct the URL
        url = f"{server_host}/api/descriptors"
        payload = {
            "type": descriptor_name,
            "chemicals": qsar_smiles,
            "chemIdType": "SMILES",
            "format": "JSON",
            "options": {
                "headers": True,
            },
        }

        response = get_requests_session().post(url, json=payload)

        if response.status_code == 200:
            return response.json(), response.status_code

        if response.status_code == 400 and len(qsar_smiles) > 1:
            diagnostic = self._diagnose_descriptor_batch_400(url, descriptor_name, qsar_smiles)
            logging.warning(
                "Descriptor batch endpoint returned 400; descriptor_service=%s batch_size=%s diagnostic=%s",
                descriptor_name,
                len(qsar_smiles),
                diagnostic,
            )
            return (
                f"{response.text}; diagnostic={diagnostic}",
                response.status_code,
            )

        return response.text, response.status_code

    def response_to_df(self, response, qsarSmiles):
        descriptor_dict = response.json()
        return self.response_json_to_df(descriptor_dict, [qsarSmiles])

    @staticmethod
    def _coerce_descriptor_headers(candidate):
        if isinstance(candidate, (list, tuple)):
            return [str(item) for item in candidate]

        if isinstance(candidate, str):
            text = candidate.strip()
            if not text:
                return None

            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None

            if isinstance(parsed, (list, tuple)):
                return [str(item) for item in parsed]

            for delimiter in ("\t", ",", "|", ";"):
                if delimiter in text:
                    parts = [part.strip() for part in text.split(delimiter) if part.strip()]
                    if len(parts) > 1:
                        return parts

        if isinstance(candidate, dict):
            for nested_key in (
                "headers",
                "descriptorHeaders",
                "descriptor_headers",
                "descriptorNames",
                "descriptor_names",
                "columns",
                "column_names",
            ):
                nested_headers = DescriptorsAPI._coerce_descriptor_headers(candidate.get(nested_key))
                if nested_headers:
                    return nested_headers

        return None

    @staticmethod
    def _summarize_descriptor_chemical(chemical):
        if isinstance(chemical, dict):
            summary_parts = [f"dict keys={list(chemical.keys())[:10]}"]
            descriptors = chemical.get("descriptors")
            if isinstance(descriptors, dict):
                summary_parts.append(f"descriptors=dict keys={list(descriptors.keys())[:10]}")
            elif isinstance(descriptors, list):
                summary_parts.append(f"descriptors=list len={len(descriptors)}")
            elif descriptors is not None:
                summary_parts.append(f"descriptors_type={type(descriptors).__name__}")
            return " ".join(summary_parts)

        return f"type={type(chemical).__name__} preview={DescriptorsAPI._preview_value(chemical)}"

    @staticmethod
    def _extract_descriptor_headers(descriptor_dict):
        headers = DescriptorsAPI._coerce_descriptor_headers(descriptor_dict.get("headers"))
        if headers:
            return headers

        for parent_key in ("options", "info"):
            parent = descriptor_dict.get(parent_key)
            if not isinstance(parent, dict):
                continue

            for candidate_key in (
                "headers",
                "descriptorHeaders",
                "descriptor_headers",
                "descriptorNames",
                "descriptor_names",
                "columns",
                "column_names",
            ):
                headers = DescriptorsAPI._coerce_descriptor_headers(parent.get(candidate_key))
                if headers:
                    return headers

        chemicals = descriptor_dict.get("chemicals")
        if isinstance(chemicals, list):
            for chemical in chemicals[:5]:
                if not isinstance(chemical, dict):
                    continue

                descriptors = chemical.get("descriptors")

                if isinstance(descriptors, dict):
                    return list(descriptors.keys())

                for candidate_key in (
                    "headers",
                    "descriptorHeaders",
                    "descriptor_headers",
                    "descriptorNames",
                    "descriptor_names",
                ):
                    headers = DescriptorsAPI._coerce_descriptor_headers(chemical.get(candidate_key))
                    if headers:
                        return headers

        first_chemical_summary = "none"
        if isinstance(chemicals, list) and chemicals:
            first_chemical_summary = DescriptorsAPI._summarize_descriptor_chemical(chemicals[0])

        top_level_keys = list(descriptor_dict.keys())
        options_keys = list(descriptor_dict.get("options", {}).keys()) if isinstance(descriptor_dict.get("options"), dict) else []
        info_keys = list(descriptor_dict.get("info", {}).keys()) if isinstance(descriptor_dict.get("info"), dict) else []
        options_headers = None
        if isinstance(descriptor_dict.get("options"), dict):
            options_headers = descriptor_dict["options"].get("headers")
        raise KeyError(
            "headers"
            f" (top_level_keys={top_level_keys}, options_keys={options_keys}, info_keys={info_keys}, "
            f"chemicals_len={len(chemicals) if isinstance(chemicals, list) else 'n/a'}, "
            f"options_headers_type={type(options_headers).__name__ if options_headers is not None else 'missing'}, "
            f"first_chemical={first_chemical_summary})"
        )

    def response_json_to_df(self, descriptor_dict, qsarSmilesList):
        headers = self._extract_descriptor_headers(descriptor_dict)
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
            raw_descriptors = chemical['descriptors']

            if isinstance(raw_descriptors, dict):
                descriptor_values = [raw_descriptors.get(header_name) for header_name in headers[2:]]
            else:
                if not isinstance(raw_descriptors, (list, tuple)):
                    raise ValueError(
                        "descriptor response has unexpected descriptors type "
                        f"{type(raw_descriptors).__name__} for smiles={self._preview_value(qsar_smiles)}"
                    )
                descriptor_values = raw_descriptors
                expected_descriptor_count = len(headers) - 2
                if len(descriptor_values) != expected_descriptor_count:
                    raise ValueError(
                        "descriptor response column mismatch "
                        f"for smiles={self._preview_value(qsar_smiles)}: "
                        f"expected_descriptors={expected_descriptor_count} actual_descriptors={len(descriptor_values)} "
                        f"chemical={self._summarize_descriptor_chemical(chemical)}"
                    )

            descriptors = [
                float(descriptor) if descriptor is not None else np.nan
                for descriptor in descriptor_values
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
    def _build_standardize_chemical_payload(chemical, index):
        if isinstance(chemical, dict):
            payload = {
                "id": str(chemical.get("id", index)),
                "chemId": chemical.get("chemId") or chemical.get("sid") or chemical.get("cid") or chemical.get("name") or chemical.get("smiles") or "",
                "cid": chemical.get("cid", ""),
                "sid": chemical.get("sid", ""),
                "casrn": chemical.get("casrn", ""),
                "name": chemical.get("name", ""),
                "smiles": chemical.get("smiles", ""),
                "canonicalSmiles": chemical.get("canonicalSmiles", ""),
                "inchi": chemical.get("inchi", ""),
                "inchiKey": chemical.get("inchiKey", ""),
                "mol": chemical.get("mol", ""),
                "molFormula": chemical.get("molFormula", ""),
                "image": chemical.get("image", chemical.get("imageSrc", "")),
                "additionalProps": chemical.get("additionalProps", {}),
            }

            if chemical.get("averageMass") is not None:
                payload["averageMass"] = chemical["averageMass"]
            if chemical.get("monoisotopicMass") is not None:
                payload["monoisotopicMass"] = chemical["monoisotopicMass"]
            return payload

        smiles_value = str(chemical)
        return {
            "id": str(index),
            "chemId": smiles_value,
            "cid": "",
            "sid": "",
            "casrn": "",
            "name": "",
            "smiles": smiles_value,
            "canonicalSmiles": "",
            "inchi": "",
            "inchiKey": "",
            "mol": "",
            "molFormula": "",
            "image": "",
            "additionalProps": {},
        }

    @staticmethod
    def _build_standardize_payload(smiles, full, workflow):
        if isinstance(smiles, str):
            chemicals_input = [smiles]
        else:
            chemicals_input = list(smiles)

        return {
            "options": {
                "workflow": workflow or "",
                "run": "",
                "recordId": "",
            },
            "chemicals": [
                QsarSmilesAPI._build_standardize_chemical_payload(chemical, index)
                for index, chemical in enumerate(chemicals_input)
            ],
            "full": full,
        }

    @staticmethod
    def call_qsar_ready_standardize_post(server_host, smiles, full, workflow):
        jo_body = QsarSmilesAPI._build_standardize_payload(smiles, full, workflow)

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

    
    
