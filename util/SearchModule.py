import requests
import json


def runChemical(min_similarity, smiles):
    '''
    Searches the cheminformatics modules
    :param min_similarity:
    :param smiles:
    '''
    
    url = "https://hcd.rtpnc.epa.gov/api/search"

    payload = json.dumps({
        "searchType": "SIMILAR",

        # "inputType": "MOL",
        # "query": "\n  Ketcher  1162516302D 1   1.00000     0.00000     0\n\n 15 14  0     0  0            999 V2000\n    5.4643    0.1270    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0\n    7.1963    0.1270    0.0000 Si  0  0  0  0  0  0  0  0  0  0  0  0\n    6.3302    0.6270    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    2.0000    0.1270    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n   10.6605    0.1270    0.0000 O   0  0  0  0  0  0  0  0  0  0  0  0\n    4.5982   -0.3730    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    8.0624   -0.3730    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    7.6964    0.9930    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    6.6964   -0.7390    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    4.9642    0.9930    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    5.9642   -0.7390    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    8.9283    0.1270    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    3.7321    0.1270    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    2.8661   -0.3730    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n    9.7944   -0.3730    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0\n  1  3  1  0     0  0\n  1  6  1  0     0  0\n  1 10  1  0     0  0\n  1 11  1  0     0  0\n  2  3  1  0     0  0\n  2  7  1  0     0  0\n  2  8  1  0     0  0\n  2  9  1  0     0  0\n  4 14  2  0     0  0\n  5 15  2  0     0  0\n  6 13  1  0     0  0\n  7 12  1  0     0  0\n 12 15  1  0     0  0\n 13 14  1  0     0  0\nM  END\n",
        "inputType": "SMILES",
        "query": smiles,
        "offset": 0,
        "limit": None,
        "sortBy": "similarity",
        "sortDirection": "desc",
        "params": {
            "min-similarity": min_similarity,
            "max-similarity": "1.0",
            "similarity-type": "tanimoto",
            "mass-type": "monoisotopic-mass",
            "min-mass": None,
            "max-mass": None,
            "single-component": True,
            "formula": None,
            "formula-query": True,
            "filter-stereo": None,
            "filter-chiral": None,
            "filter-isotopes": None,
            "filter-charged": None,
            "filter-multicomponent": None,
            "filter-radicals": None,
            "filter-salts": None,
            "filter-polymers": None,
            "filter-sgroups": None,
            "include-elements": None,
            "exclude-elements": None,
            "hazard-name": None,
            "hazard-source": None,
            "hazard-route": None,
            "hazard-category": None,
            "hazard-code": None,
            "hazard-organism": None,
            "min-toxicity": None,
            # "min-authority": None,
            "min-authority": "Screening",
            "export-all-props": True
        }
    })
    headers = {
        'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)

    # print(response.text)

    response_obj = json.loads(response.text)

    print(min_similarity, response_obj["recordsCount"])
    return response_obj


def getSearchWithMaxCount(maxCount, smiles):
    '''
    Iterates until it gets the right count, but they have new functionality that this isnt needed
    :param maxCount:
    :param smiles:
    '''
    
    for i in range(95, 0, -2):
        sim = i / 100
        # print (sim/100)
        results = runChemical(sim, smiles)  # 26542-47-2

        count = results["recordsCount"]

        if count >= maxCount:
            return results


if __name__ == '__main__':
    smiles = "C[Si](C)(CCC=O)O[Si](C)(C)CCC=O"

    # maxCount = 50
    # results=getSearchWithMaxCount(maxCount,smiles)

    results = runChemical(0.27, smiles)  # 26542-47-2

    count=0

    for record in results['records']:
        # print(record['similarity'],record['SMILES'])
        smiles = ''

        if "smiles" in record:
            smiles = record['smiles']
            count = count + 1

        print(record['similarity'], smiles)

    print (count)