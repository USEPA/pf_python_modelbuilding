import unittest
from typing import Mapping

import pandas as pd

from models.RunDataset import ModelLoader


class _Session:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


class _Database:
    def __init__(self, session: _Session) -> None:
        self.session = session
        self.records: list[Mapping[str, object]] = []

    def create_many(
        self,
        table: str,
        records: list[Mapping[str, object]],
        commit: bool = True,
    ) -> list[int]:
        if table != "predictions":
            raise AssertionError(f"Unexpected table: {table}")
        self.records.extend(records)
        return list(range(1, len(records) + 1))


class RunDatasetPredictionPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.session = _Session()
        self.database = _Database(self.session)
        self.loader = ModelLoader.__new__(ModelLoader)
        self.loader.session = self.session
        self.loader.dbl = self.database

    def test_duplicate_chemical_and_fold_are_inserted_once(self) -> None:
        predictions = pd.DataFrame(
            {
                "canon_qsar_smiles": ["CCO", "CCO", "CCN", "CCO"],
                "pred": [1.25, 1.25, 2.5, 1.5],
                "cv_fold": [1, 1, 1, 2],
                "exp_prop_id": [101, 102, 103, 104],
            }
        )

        inserted_count = self.loader.load_predictions(
            user="tester",
            set_name="training cv",
            df=predictions,
            fk_model_id=42,
        )

        self.assertEqual(inserted_count, 3)
        self.assertEqual(len(self.database.records), 3)
        self.assertEqual(
            {record["canon_qsar_smiles"] for record in self.database.records},
            {"CCO", "CCN"},
        )
        self.assertEqual(
            sum(record["canon_qsar_smiles"] == "CCO" for record in self.database.records),
            2,
        )
        self.assertEqual(
            [record["fk_splitting_id"] for record in self.database.records],
            [2, 2, 3],
        )
        self.assertNotIn(1, [record["fk_splitting_id"] for record in self.database.records])
        self.assertEqual(self.session.commit_count, 1)

    def test_duplicate_chemical_and_fold_with_conflicting_predictions_fails(self) -> None:
        predictions = pd.DataFrame(
            {
                "canon_qsar_smiles": ["CCO", "CCO"],
                "pred": [1.25, 1.5],
                "cv_fold": [1, 1],
            }
        )

        with self.assertRaisesRegex(ValueError, "conflicting predicted"):
            self.loader.load_predictions(
                user="tester",
                set_name="training cv",
                df=predictions,
                fk_model_id=42,
            )

        self.assertEqual(self.database.records, [])
        self.assertEqual(self.session.commit_count, 0)


if __name__ == "__main__":
    unittest.main()
