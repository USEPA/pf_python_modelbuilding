import pandas as pd

from unittest import TestCase, main

from models.df_utilities import (
    prepare_instances, prepare_instances_with_preselected_descriptors,
    prepare_instances_with_preselected_descriptors_gcm, prepare_prediction_instances
)


class TestRemove(TestCase):
    def setUp(self):
        self.df_data = pd.DataFrame({'ID': ["A", "B", "C", "D"],
                                     'Property': [0.1, 0.2, 0.3, 0.4],
                                     'XLOGP': [1, 1, 0, 0], 
                                     'XLOGP2': [0, 0, 1, 1],
                                     'Col1': [1, 0, 1, 0],
                                     'Col2': [0, 1, 0, 1],
                                     'Col3': [0, 0, 0, 0]})
        self.df_data_bin = pd.DataFrame({'ID': ["A", "B", "C", "D"],
                                         'Property': [1, 0, 1, 0]})
        self.descriptor_names = ['XLOGP', 'Col1', 'Col3']
                                         
    def test_prepare_instances_ids(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", False, False)
        self.assertListEqual(ids.tolist(), ["A", "B", "C", "D"])
        
    def test_prepare_instances_labels(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", False, False)
        self.assertListEqual(labels.tolist(), [0.1, 0.2, 0.3, 0.4])
        
    def test_prepare_instances_is_binary_false(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", False, False)
        self.assertFalse(is_binary)
        
    def test_prepare_instances_is_binary_true(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data_bin, "", False, False)
        self.assertTrue(is_binary)
        
    def test_prepare_instances_features_remove_logp_only(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", True, False)
        self.assertEqual(features.shape, (4, 2))
        
    def test_prepare_instances_column_names_remove_logp_only(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", True, False)
        self.assertEqual(column_names, ['Col1', 'Col2'])
        
    def test_prepare_instances_features_remove_corr_only(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", False, True)
        self.assertEqual(features.shape, (4, 2))
        
    def test_prepare_instances_column_names_remove_corr_only(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", False, True)
        self.assertEqual(column_names, ['XLOGP', 'Col1'])
        
    def test_prepare_instances_features_remove_logp_and_corr(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", True, True)
        self.assertEqual(features.shape, (4, 1))
        
    def test_prepare_instances_column_names_remove_logp_and_corr(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances(self.df_data, "", True, True)
        self.assertEqual(column_names, ['Col1'])
        
    def test_prepare_instances_with_preselected_descriptors_ids(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data, "", self.descriptor_names)
        self.assertListEqual(ids.tolist(), ["A", "B", "C", "D"])
        
    def test_prepare_instances_with_preselected_descriptors_labels(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data, "", self.descriptor_names)
        self.assertListEqual(labels.tolist(), [0.1, 0.2, 0.3, 0.4])
        
    def test_prepare_instances_with_preselected_descriptors_is_binary_false(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data, "", self.descriptor_names)
        self.assertFalse(is_binary)
        
    def test_prepare_instances_with_preselected_descriptors_is_binary_true(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data_bin, "", self.descriptor_names)
        self.assertTrue(is_binary)
        
    def test_prepare_instances_with_preselected_descriptors_features(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data, "", self.descriptor_names)
        self.assertEqual(features.shape, (4, 3))
        
    def test_prepare_instances_with_preselected_descriptors_column_names(self):
        ids, labels, features, column_names, is_binary =\
            prepare_instances_with_preselected_descriptors(self.df_data, "", self.descriptor_names)
        self.assertEqual(column_names, self.descriptor_names)
        
    def test_prepare_prediction_instances_ids(self):
        ids, labels, features = prepare_prediction_instances(self.df_data, self.descriptor_names)
        self.assertListEqual(ids.tolist(), ["A", "B", "C", "D"])
        
    def test_prepare_prediction_instances_labels(self):
        ids, labels, features = prepare_prediction_instances(self.df_data, self.descriptor_names)
        self.assertListEqual(labels.tolist(), [0.1, 0.2, 0.3, 0.4])
        
    def test_prepare_prediction_instances_features(self):
        ids, labels, features = prepare_prediction_instances(self.df_data, self.descriptor_names)
        self.assertEqual(features.shape, (4, 3))

    def test_prepare_instances_with_preselected_descriptors_gcm_appends_logp_martin(self):
        df = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D'],
            'Property': [0.1, 0.2, 0.3, 0.4],
            'As [+5 valence, one double bond]': [1, 1, 0, 0],
            'Another fragment': [0, 1, 0, 1],
            '-N=S=O': [0, 0, 1, 1],
            'LOGP_Martin': [1.5, 2.5, 3.5, 4.5],
        })

        ids, labels, features, column_names, is_binary = prepare_instances_with_preselected_descriptors_gcm(
            df, "training", min_count=1, add_xgboost_log_p=True
        )

        self.assertTrue('LOGP_Martin' in features.columns)
        self.assertEqual(list(features.columns[-1:]), ['LOGP_Martin'])
        self.assertEqual(features['LOGP_Martin'].tolist(), [1.5, 2.5, 3.5, 4.5])

    def test_prepare_instances_with_preselected_descriptors_gcm_uses_explicit_logp_columns(self):
        df = pd.DataFrame({
            'ID': ['A', 'B', 'C', 'D'],
            'Property': [0.1, 0.2, 0.3, 0.4],
            'As [+5 valence, one double bond]': [1, 1, 0, 0],
            'Another fragment': [0, 1, 0, 1],
            '-N=S=O': [0, 0, 1, 1],
            'LOGP_Martin': [1.5, 2.5, 3.5, 4.5],
            'LOGP_Martin_squared': [2.25, 6.25, 12.25, 20.25],
        })

        _, _, features, _, _ = prepare_instances_with_preselected_descriptors_gcm(
            df, "training", min_count=1, logp_columns=['LOGP_Martin']
        )

        self.assertListEqual(features['LOGP_Martin'].tolist(), [1.5, 2.5, 3.5, 4.5])
        self.assertNotIn('LOGP_Martin_squared', features.columns)
    
    
if __name__ == '__main__':
    main()
    