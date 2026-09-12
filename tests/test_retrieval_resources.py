import io
import json
import unittest
from contextlib import redirect_stdout

from evals.resources import (
    cli_main,
    format_resource_report,
    measure_retrieval_resources,
)


class RetrievalResourceMeasurementTests(unittest.TestCase):
    def test_measurement_separates_index_and_context_payload(self):
        result = measure_retrieval_resources(
            repetitions=2,
            page_count=6,
            page_chars=2_000,
            top_k=3,
        )

        fixture = result["fixture"]
        context = result["context_payload"]
        self.assertEqual(result["benchmark_version"], 1)
        self.assertEqual(fixture["page_count"], 6)
        self.assertEqual(context["returned_chunks"], 3)
        self.assertGreater(fixture["index_file_bytes"], context["utf8_bytes"])
        self.assertLess(context["share_of_index"], 1.0)
        self.assertEqual(result["timing_ms"]["repetitions"], 2)
        self.assertEqual(
            len(result["timing_ms"]["first_retrieval"]["samples"]),
            2,
        )

    def test_cli_supports_human_and_json_reports(self):
        human_output = io.StringIO()
        with redirect_stdout(human_output):
            status = cli_main(
                [
                    "--repetitions",
                    "1",
                    "--pages",
                    "4",
                    "--page-chars",
                    "1500",
                ]
            )
        self.assertEqual(status, 0)
        self.assertIn("First retrieval:", human_output.getvalue())
        self.assertIn("Tool Result payload:", human_output.getvalue())

        json_output = io.StringIO()
        with redirect_stdout(json_output):
            status = cli_main(
                [
                    "--repetitions",
                    "1",
                    "--pages",
                    "4",
                    "--page-chars",
                    "1500",
                    "--json",
                ]
            )
        self.assertEqual(status, 0)
        result = json.loads(json_output.getvalue())
        self.assertIn("cached_retrieval", result["timing_ms"])
        self.assertIn(
            "payload excludes prior Context",
            format_resource_report(result),
        )

    def test_rejects_measurements_outside_resource_bounds(self):
        with self.assertRaisesRegex(ValueError, "repetitions"):
            measure_retrieval_resources(repetitions=0)
        with self.assertRaisesRegex(ValueError, "page_chars"):
            measure_retrieval_resources(page_chars=100)
        with self.assertRaisesRegex(ValueError, "top_k"):
            measure_retrieval_resources(top_k=6)


if __name__ == "__main__":
    unittest.main()
