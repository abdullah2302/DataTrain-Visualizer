import io
import sys
import unittest
from pathlib import Path

from sklearn.model_selection import train_test_split

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from app import app


class TrainGraphResponseTests(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.csv_content = "\n".join(
            [
                "feature,target",
                "100,10",
                "101,20",
                "102,30",
                "103,40",
                "104,50",
                "105,60",
                "106,70",
                "107,80",
                "108,90",
                "109,100",
            ]
        )

    def _post_train(self, model):
        response = self.client.post(
            "/train",
            data={
                "file": (io.BytesIO(self.csv_content.encode("utf-8")), "sample.csv"),
                "target": "target",
                "model": model,
                "graph": "line",
                "train_ratio": "0.8",
            },
            content_type="multipart/form-data",
        )
        self.assertEqual(response.status_code, 200, response.get_json())
        return response.get_json()

    def _assert_aligned_graph_arrays(self, body):
        self.assertEqual(len(body["train_predictions"]), len(body["train_x_data"]))
        self.assertEqual(len(body["train_predictions"]), len(body["train_y_data"]))
        self.assertEqual(len(body["test_predictions"]), len(body["test_x_data"]))
        self.assertEqual(len(body["test_predictions"]), len(body["test_y_data"]))

        expected_train_idx, expected_test_idx = train_test_split(
            list(range(10)), train_size=0.8, random_state=42
        )
        self.assertEqual(body["train_x_data"], expected_train_idx)
        self.assertEqual(body["test_x_data"], expected_test_idx)

    def test_graph_arrays_aligned_for_unscaled_model(self):
        body = self._post_train("linear_regression")
        self.assertEqual(body["graph_type"], "line")
        self._assert_aligned_graph_arrays(body)

    def test_graph_arrays_aligned_for_scaled_model(self):
        body = self._post_train("svm_regressor")
        self.assertEqual(body["graph_type"], "line")
        self._assert_aligned_graph_arrays(body)


if __name__ == "__main__":
    unittest.main()
