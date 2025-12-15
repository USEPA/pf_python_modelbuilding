import os
from unittest import TestCase, main

from API_Utilities import DescriptorsAPI
from app import predictDB
from model_ws_utilities import call_do_predictions
from models.df_utilities import load_df_from_file

TEST_DATA_FOLDER = "test_data/"


class TestPredict(TestCase):
    def setUp(self):
        self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
        self.descriptors_api = DescriptorsAPI()
        self.serverAPIs = os.getenv("CIM_API_SERVER", "https://cim-dev.sciencedataexperts.com/")

    def test_model_predict_1065(self):
        r = predictDB('c1ccccc1', 1065)
        print(r)

    def test_model_predict_1065_array(self):
        r = predictDB(['c1ccccc1', 'CCCC'], 1065)
        print(r)

    def test_descriptors_get(self):
        r = self.descriptors_api.call_descriptors_get(self.serverAPIs,
                                                      "c1ccccc1",
                                                      "webtest")
        print(r)

    def test_descriptors_post(self):
        r = self.descriptors_api.call_descriptors_post(self.serverAPIs,
                                                       ["c1ccccc1", "CCCC"],
                                                       "webtest")
        print(r)


if __name__ == '__main__':
    main()
