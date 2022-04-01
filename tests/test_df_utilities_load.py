# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 15:10:29 2022

@author: GSINCL01
"""
from unittest import TestCase, main

from models.df_utilities import (
    load_df, load_df_from_file
)


TEST_DATA_FOLDER = "test_data/"


class TestLoad(TestCase):
    def setUp(self):
        self.tsv_str = "Col1\tCol2\r\n1\t1\r\n0\t1\r\n1\t0\r\n0\t0"
        self.csv_str = "Col1,Col2\r\n1,1\r\n0,1\r\n1,0\r\n0,0"
        self.tsv_inf_str = "Col1\tCol2\r\n1\t1\r\n0\t1\r\n1\t0\r\n0\tInf"
        self.tsv_nan_str = "Col1\tCol2\r\n1\t1\r\n0\t1\r\n1\t0\r\n0\tNaN"
        
        self.tsv_file_path = TEST_DATA_FOLDER + "test_tsv.tsv"
        self.csv_file_path = TEST_DATA_FOLDER + "test_csv.csv"
        self.tsv_inf_file_path = TEST_DATA_FOLDER + "test_tsv_inf.tsv"
        self.tsv_nan_file_path = TEST_DATA_FOLDER + "test_tsv_nan.tsv"
        
    def test_load_tsv(self):
        df = load_df(self.tsv_str)
        self.assertEqual(df.shape, (4, 2))
        
    def test_load_csv(self):
        df = load_df(self.csv_str)
        self.assertEqual(df.shape, (4, 2))
        
    def test_load_tsv_inf(self):
        df = load_df(self.tsv_inf_str)
        self.assertEqual(df.shape, (3, 2))
        
    def test_load_tsv_nan(self):
        df = load_df(self.tsv_nan_str)
        self.assertEqual(df.shape, (3, 2))
        
    def test_load_tsv_from_file(self):
        df = load_df_from_file(self.tsv_file_path)
        self.assertEqual(df.shape, (4, 2))
        
    def test_load_csv_from_file(self):
        df = load_df_from_file(self.csv_file_path)
        self.assertEqual(df.shape, (4, 2))
        
    def test_load_tsv_inf_from_file(self):
        df = load_df_from_file(self.tsv_inf_file_path)
        self.assertEqual(df.shape, (3, 2))
        
    def test_load_tsv_nan_from_file(self):
        df = load_df_from_file(self.tsv_nan_file_path)
        self.assertEqual(df.shape, (3, 2))


if __name__ == '__main__':
    main()