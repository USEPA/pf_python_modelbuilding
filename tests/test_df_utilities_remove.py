# -*- coding: utf-8 -*-
import pandas as pd

from unittest import TestCase, main

from models.df_utilities import (
    do_remove_constant_descriptors, do_remove_correlated_descriptors, remove_log_p_descriptors
)


class TestRemove(TestCase):
    def setUp(self):
        self.df = pd.DataFrame({'Col1': [1, 1, 0, 0], 'Col2': [1, 0, 1, 0]})
        self.corr_threshold = 0.95
        self.df_corr = pd.DataFrame({'Col1': [1, 1, 0, 0], 'Col2': [1, 1, 0, 0]})
        self.df_neg_corr = pd.DataFrame({'Col1': [1, 1, 0, 0], 'Col2': [0, 0, 1, 1]})
        self.df_const = pd.DataFrame({'Col1': [1, 1, 0, 0], 'Col2': [0, 0, 0, 0]})
        self.df_logp = pd.DataFrame({'Col1': [1, 1, 0, 0], 
                                     'Col2': [1, 0, 1, 0],
                                     'XLOGP': [0, 0, 0, 0],
                                     'XLogP': [0, 0, 0, 0],
                                     'xlogp': [0, 0, 0, 0],
                                     'XLOGP2': [0, 0, 0, 0],
                                     'ALOGP': [0, 0, 0, 0],
                                     'ALOGP2': [0, 0, 0, 0]})
    
    def test_remove_constant_descriptors(self):
        df_after = do_remove_constant_descriptors(self.df_const, [])
        self.assertListEqual(list(df_after.columns), ['Col1'])
        
    def test_doesnt_remove_non_constant_descriptors(self):
        df_after = do_remove_constant_descriptors(self.df, [])
        self.assertEqual(df_after.shape, (4, 2))
        
    def test_remove_correlated_descriptors(self):
        df_corr_copy = self.df_corr.copy()
        do_remove_correlated_descriptors(df_corr_copy, self.corr_threshold)
        self.assertEqual(df_corr_copy.shape, (4, 1))
        
    def test_remove_negative_correlated_descriptors(self):
        df_neg_corr_copy = self.df_neg_corr.copy()
        do_remove_correlated_descriptors(df_neg_corr_copy, self.corr_threshold)
        self.assertEqual(df_neg_corr_copy.shape, (4, 1))
        
    def test_doesnt_remove_uncorrelated_descriptors(self):
        df_copy = self.df.copy()
        do_remove_correlated_descriptors(df_copy, self.corr_threshold)
        self.assertEqual(df_copy.shape, (4, 2))
        
    def test_remove_logp_descriptors(self):
        df_after = remove_log_p_descriptors(self.df_logp, "")
        self.assertListEqual(list(df_after.columns), ['Col1', 'Col2'])
        
    def test_doesnt_remove_non_logp_descriptors(self):
        df_after = remove_log_p_descriptors(self.df, "")
        self.assertEqual(df_after.shape, (4, 2))
    
    
if __name__ == '__main__':
    main()