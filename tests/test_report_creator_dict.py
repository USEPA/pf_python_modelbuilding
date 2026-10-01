from unittest import TestCase

from dominate import document

from report_creator_dict import ReportCreator


class TestReportCreatorDict(TestCase):
    def test_chemical_identifiers_table_handles_null_identifier_values(self):
        doc = document()

        with doc:
            ReportCreator.ChemicalIdentifiersSection().create_chemical_identifiers_table(
                {
                    "name": None,
                    "sid": None,
                    "cid": None,
                    "casrn": None,
                    "averageMass": None,
                }
            )

        html = str(doc)

        self.assertIn("Preferred name:", html)
        self.assertIn("DTXSID:", html)
        self.assertIn("DTXCID:", html)
        self.assertIn("CASRN:", html)
        self.assertIn("Molecular weight:", html)
        self.assertNotIn("None", html)
        self.assertGreaterEqual(html.count("N/A"), 5)
