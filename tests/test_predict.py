from unittest import TestCase, main

from model_ws import loadModelFromDatabase
from model_ws_utilities import call_do_predictions
from models.df_utilities import load_df_from_file

TEST_DATA_FOLDER = "test_data/"


class TestPredict(TestCase):
    def setUp(self):
        self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
        self.model = loadModelFromDatabase(1067)

    def test_model_predict(self):
        call_do_predictions(self.df_ext_pred_set, self.model)


if __name__ == '__main__':
    main()
