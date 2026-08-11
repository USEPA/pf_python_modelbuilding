from unittest import TestCase
from unittest.mock import patch

import app


class TestMetadata(TestCase):
    def tearDown(self):
        app._metadata = None

    def test_get_metadata_uses_collected_model_details(self):
        model_details = [{"modelId": 1065}, {"modelId": 1066}]

        with patch("app.collect_model_details_for_metadata", return_value=model_details) as collect:
            app._metadata = None
            metadata = app.get_metadata()

        collect.assert_called_once_with([
            1065,
            1066,
            1067,
            1068,
            1069,
            1070,
            1754,
            1756,
            1757,
            1758,
            1763,
        ])
        self.assertEqual(model_details, metadata["endpoints"])
