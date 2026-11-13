"""Deployment smoke tests; never train or overwrite the saved model."""
import os
from pathlib import Path
import tempfile
import unittest

from streamlit.testing.v1 import AppTest


APP = Path(__file__).resolve().parents[1] / "app.py"


class DeploymentSmokeTest(unittest.TestCase):
    def test_dashboard_and_predictions_from_other_directory(self):
        original_directory = Path.cwd()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                app = AppTest.from_file(str(APP)).run(timeout=60)
                self.assertEqual(len(app.exception), 0, str(app.exception))
                self.assertEqual(app.metric[0].value, "10,000")
                self.assertEqual(len(app.dataframe[0].value), 10)
                for product_type in ("L", "M", "H"):
                    with self.subTest(product_type=product_type):
                        app.selectbox[0].select(product_type)
                        app.button[0].click().run(timeout=60)
                        self.assertEqual(len(app.exception), 0, str(app.exception))
                        self.assertTrue(len(app.success) or len(app.error))
                        self.assertEqual(app.metric[-1].label, "Estimated Failure Score")
                        score = float(app.metric[-1].value.rstrip("%"))
                        self.assertTrue(0 <= score <= 100)
            finally:
                os.chdir(original_directory)


if __name__ == "__main__":
    unittest.main()
