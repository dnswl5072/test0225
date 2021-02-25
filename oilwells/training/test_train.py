import json
from pathlib import Path

import numpy as np
import pandas as pd

from oilwells.training.train import X_COLUMNS, get_model_metrics, train_model

SAMPLE_INPUT = {
    "temperature": 0,
    "pressure": 0,
    "load": 0,
    "production_rate": 1,
    "rotor_will_jam_in_next_two_weeks": True,
}


def test_train_model():
    train_df = pd.DataFrame.from_records([SAMPLE_INPUT,] * 2)
    with open(Path(__file__).parent.parent.absolute() / "parameters.json") as f:
        params = json.load(f)["training"]
    model = train_model(train_df, params)
    preds = model.predict(train_df[X_COLUMNS].values)
    np.testing.assert_equal(preds, [True, True])


def test_get_model_metrics():
    class MockModel:
        @staticmethod
        def predict(data):
            return [True, True]

    test_df = pd.DataFrame.from_records([SAMPLE_INPUT, {**SAMPLE_INPUT, **{"rotor_will_jam_in_next_two_weeks": False}}])
    metrics = get_model_metrics(MockModel(), test_df)
    assert "accuracy" in metrics
    accuracy = metrics["accuracy"]
    np.testing.assert_almost_equal(accuracy, 0.5)
