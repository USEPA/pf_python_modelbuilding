from pathlib import Path
from unittest import TestCase

import yaml


class SwaggerModelExamplesTests(TestCase):
    def test_predict_get_lists_new_koc_models(self):
        spec = yaml.safe_load((Path(__file__).parents[1] / "swagger.yaml").read_text())
        parameters = spec["paths"]["/api/predictor_models/predict"]["get"]["parameters"]
        model_parameter = next(
            parameter for parameter in parameters if parameter["name"] == "model_id"
        )

        examples = model_parameter["examples"]
        actual = {example["value"]: example["summary"] for example in examples.values()}

        self.assertEqual("KOC-RF Martin 2026", actual[1754])
        self.assertEqual("KOC-XGB Martin 2026", actual[1756])
        self.assertEqual("KOC-REG Martin 2026", actual[1757])
        self.assertEqual("KOC-KNN Martin 2026", actual[1758])
        self.assertEqual("KOC-GCM Martin 2026", actual[1763])
