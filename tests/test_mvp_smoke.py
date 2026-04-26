import unittest

import pandas as pd
from langchain_core.documents import Document

from healthcare_app.agents import AgentRuntime


class TestMVPSmoke(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame(
            [
                {
                    "address_stateOrRegion": "Bihar",
                    "address_city": "Siwan",
                    "pin": "841226",
                    "latitude": "26.2201",
                    "longitude": "84.3561",
                },
                {
                    "address_stateOrRegion": "Bihar",
                    "address_city": "Gopalganj",
                    "pin": "841428",
                    "latitude": "26.4680",
                    "longitude": "84.4433",
                },
            ]
        )
        self.runtime = AgentRuntime(llm=None, retriever=None, df=self.df)

    def test_extract_constraints_for_complex_query(self):
        query = "Find the nearest facility in rural Bihar for emergency appendectomy with part-time doctors"
        constraints = self.runtime._extract_constraints(query)
        self.assertEqual(constraints["state"], "bihar")
        self.assertTrue(constraints["needs_nearest"])
        self.assertTrue(constraints["needs_rural"])
        self.assertTrue(constraints["needs_part_time"])

    def test_apply_query_constraints_filters_procedure_miss(self):
        query = "Find emergency appendectomy care in rural Bihar"
        ranked = [
            (
                Document(
                    page_content="Rural hospital with emergency and appendix surgery, visiting doctor coverage.",
                    metadata={"state": "Bihar", "city": "Siwan", "pin": "841226", "completeness": 0.8},
                ),
                0.9,
                ["location_match"],
            ),
            (
                Document(
                    page_content="Urban clinic in Bihar with dermatology only.",
                    metadata={"state": "Bihar", "city": "Patna", "pin": "800001", "completeness": 0.8},
                ),
                0.85,
                ["location_match"],
            ),
        ]
        constrained, summary = self.runtime._apply_query_constraints(ranked, query)
        self.assertEqual(len(constrained), 1)
        self.assertIn("constraint_screen_pass", summary["matched_attributes"])

    def test_high_severity_contradictions_detected(self):
        claims = "advanced surgery | icu"
        evidence = "general ward support only"
        contradictions = self.runtime._high_severity_contradictions(claims, evidence, evidence)
        self.assertTrue(any("Surgery claim" in c for c in contradictions))
        self.assertTrue(any("ICU claim" in c for c in contradictions))


if __name__ == "__main__":
    unittest.main()

