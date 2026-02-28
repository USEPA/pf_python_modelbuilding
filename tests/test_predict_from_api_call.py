'''
Created on Feb 27, 2026

@author: TMARTI02
'''
import webbrowser
import requests
import os

def runExampleFromService():


    # Define the parameters
    smiles = "c1ccccc1"
    model_id = "1065"
    # report_format = "json"
    report_format = "html"

    use_uvicorn = False

    # Define the base URL
    if use_uvicorn:
        base_url = "http://localhost:5005/api/predictor_models/predict"
    else:
        base_url = "http://localhost:5004/api/predictor_models/predictDB"

    # Set up the parameters as a dictionary
    params = {
        'smiles': smiles,
        'model_id': model_id,
        'report_format': report_format
    }

    # Define headers if necessary
    headers = {
        'Accept-Encoding': 'json'
    }

    # Make the GET request with parameters
    response = requests.get(base_url, headers=headers, params=params)

    # Check the response
    if not response.ok:
        print(f"Request failed with status code: {response.status_code}")
        return

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, model_id + "_report." + report_format)

    # Write the HTML to the specified file path
    with open(file_path, 'w') as f:
        f.write(response.text)

    webbrowser.open(f'file://{file_path}')


if __name__ == '__main__':
    runExampleFromService()