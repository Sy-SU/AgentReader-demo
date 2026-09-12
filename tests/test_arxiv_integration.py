import os
import unittest

from tools import search_paper


RUN_LIVE_TESTS = os.getenv("RUN_LIVE_ARXIV_TESTS") == "1"


@unittest.skipUnless(
    RUN_LIVE_TESTS,
    "set RUN_LIVE_ARXIV_TESTS=1 to call the real arXiv API",
)
class ArxivIntegrationTests(unittest.TestCase):
    def test_live_search_returns_a_real_paper(self):
        result = search_paper("Attention Is All You Need")

        self.assertTrue(result["found"])
        self.assertIn(result["source"], {"arxiv", "crossref"})
        self.assertGreaterEqual(result["count"], 1)
        self.assertLessEqual(result["count"], 3)
        self.assertEqual(result["count"], len(result["papers"]))
        self.assertTrue(result["papers"][0]["title"])
        self.assertTrue(result["papers"][0]["authors"])
        self.assertTrue(result["papers"][0]["paper_url"])

    def test_live_arxiv_id_returns_direct_pdf_candidate(self):
        result = search_paper("Attention Is All You Need 1706.03762")

        self.assertTrue(result["found"])
        self.assertEqual(result["source"], "arxiv")
        self.assertEqual(result["count"], 1)
        paper = result["papers"][0]
        self.assertTrue(paper["arxiv_id"].startswith("1706.03762"))
        self.assertTrue(paper["pdf_url"].startswith("https://arxiv.org/pdf/"))


if __name__ == "__main__":
    unittest.main()
