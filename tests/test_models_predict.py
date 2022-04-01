# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:44:28 2022

@author: GSINCL01
"""
import pickle

from unittest import TestCase, main

from models.df_utilities import load_df_from_file


TEST_DATA_FOLDER = "test_data/"


def load_test_model(file_name):
    with open(TEST_DATA_FOLDER + file_name, 'rb') as file:
        model = pickle.load(file)
        return model


class TestPredict(TestCase):
    def setUp(self):
        self.rf_regr_model = load_test_model("bcf_rf.pickle")
        self.svm_regr_model = load_test_model("bcf_svm.pickle")
        self.xgb_regr_model = load_test_model("bcf_xgb.pickle")
        self.rf_class_model = load_test_model("llna_rf.pickle")
        self.svm_class_model = load_test_model("llna_svm.pickle")
        self.xgb_class_model = load_test_model("llna_xgb.pickle")
        
        self.df_ext_pred_set = load_df_from_file(TEST_DATA_FOLDER + "ext_pred_set.tsv")
        
    def run_test_model_predict(self, model):
        preds = model.do_predictions(self.df_ext_pred_set)
        self.assertEqual(len(preds), self.df_ext_pred_set.shape[0])
        self.assertTrue(all(isinstance(p, float) for p in preds))
        
    def test_rf_regr_predict(self):
        self.run_test_model_predict(self.rf_regr_model)
        
    def test_svm_regr_predict(self):
        self.run_test_model_predict(self.svm_regr_model)
        
    def test_xgb_regr_predict(self):
        self.run_test_model_predict(self.xgb_regr_model)
        # FAILS: Version conflict due to addition of enable_categorical with upgrade!
        
    def test_rf_class_predict(self):
        self.run_test_model_predict(self.rf_class_model)
        
    def test_svm_class_predict(self):
        self.run_test_model_predict(self.svm_class_model)
        
    def test_xgb_class_predict(self):
        self.run_test_model_predict(self.xgb_class_model)
        # FAILS: Version conflict due to addition of enable_categorical with upgrade!
        
        
if __name__ == '__main__':
    main()