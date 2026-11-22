import subprocess
import sys
import unittest
from pathlib import Path

from scripts import run_rag_demo as demo


class DemoTests(unittest.TestCase):
    def test_reference_comparison(self):
        result = subprocess.check_output(
            [sys.executable, "scripts/compare_classifiers.py"], text=True
        )
        for expected in ("89/100", "97/100", "86/100"):
            self.assertIn(expected, result)

    def test_sample_conversations(self):
        texts = demo.conversations([Path("data/examples/support.csv")])
        self.assertEqual(len(texts), 2)
        self.assertIn("Request return label", texts[0])

    def test_routing(self):
        self.assertTrue(demo.needs_web("What is the price?", 0.9))
        self.assertTrue(demo.needs_web("Returns?", 0.39))
        self.assertFalse(demo.needs_web("Returns?", 0.4))

    def test_invalid_label(self):
        result = subprocess.run(
            [sys.executable, "scripts/run_rag_demo.py", "--question", "Returns?",
             "--emotion", "neutral", "--urgency", "unknown"],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("invalid choice", result.stderr)
