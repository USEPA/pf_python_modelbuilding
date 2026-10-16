import unittest
from unittest.mock import patch

import pandas as pd

from applicability_domain import applicability_domain_utilities as adu
from util import predict_constants as pc


class FragmentPreparationTests(unittest.TestCase):
    def test_fragment_ranges_do_not_prepare_unused_feature_matrix(self):
        first, last = 'As [+5 valence, one double bond]', '-N=S=O'
        training = pd.DataFrame({'ID': ['a', 'b'], 'Property': [2., 3.],
                                 first: [1., 2.], last: [0., 1.]})
        test = pd.DataFrame({'ID': ['inside', 'outside'], 'Property': [0., 0.],
                            first: [1., 3.], last: [1., 1.]})
        with patch.object(adu.dfu, 'prepare_instances', side_effect=AssertionError('unused preparation')):
            result, cutoff = adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                training, test, False, [first, last], pc.Applicability_Domain_TEST_Fragment_Counts)
        self.assertEqual(result['idTest'].tolist(), ['inside', 'outside'])
        self.assertEqual(result['AD'].tolist(), [True, False])
        self.assertIsNone(cutoff)
        self.assertEqual(result.iloc[0]['fragment_table'][0]['training_min'], 1.)
        self.assertEqual(result.iloc[0]['fragment_table'][0]['training_max'], 2.)
        self.assertEqual(result.iloc[0]['fragment_table'][0]['training_count'], 2)

    def test_other_domains_keep_feature_preparation(self):
        frame = pd.DataFrame({'ID': ['a'], 'Property': [2.], 'descriptor': [1.]})
        with patch.object(adu.dfu, 'prepare_instances', side_effect=RuntimeError('prepared')):
            with self.assertRaisesRegex(RuntimeError, 'prepared'):
                adu.generate_applicability_domain_with_preselected_descriptors_from_dfs(
                    frame, frame, False, ['descriptor'], pc.Applicability_Domain_TEST_Embedding_Cosine)


if __name__ == '__main__':
    unittest.main()
