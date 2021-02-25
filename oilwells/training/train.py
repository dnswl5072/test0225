"""
Copyright (C) Microsoft Corporation. All rights reserved.​
 ​
Microsoft Corporation (“Microsoft”) grants you a nonexclusive, perpetual,
royalty-free right to use, copy, and modify the software code provided by us
("Software Code"). You may not sublicense the Software Code or any use of it
(except to your affiliates and to vendors to perform work on your behalf)
through distribution, network access, service agreement, lease, rental, or
otherwise. This license does not purport to express any claim of ownership over
data you may have shared with Microsoft in the creation of the Software Code.
Unless applicable law gives you more rights, Microsoft reserves all other
rights not expressly granted herein, whether by implication, estoppel or
otherwise. ​
 ​
THE SOFTWARE CODE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS
OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
MICROSOFT OR ITS LICENSORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THE SOFTWARE CODE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
"""

import json
import os

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

X_COLUMNS = ["temperature", "pressure", "load", "production_rate"]
Y_COLUMN = "rotor_will_jam_in_next_two_weeks"


def train_model(train_df, args):
    model = DummyClassifier(**args)
    model.fit(train_df[X_COLUMNS].values, train_df[Y_COLUMN].values)
    return model


def get_model_metrics(model, test_df):
    preds = model.predict(test_df[X_COLUMNS].values)
    accuracy = accuracy_score(preds, test_df[Y_COLUMN].values)
    metrics = {"accuracy": accuracy}
    return metrics


def main():
    # TODO: Local testing fix
    from pathlib import Path
    import sys

    sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    from prepare import prepare

    print("Running train.py")
    # Load the training data as dataframe
    root_dir = Path(__file__).absolute().parent.parent.parent
    data_dir = root_dir / "data"
    data_file = os.path.join(data_dir, "oilwells.csv")
    df = pd.read_csv(data_file, parse_dates=["time"])
    print("Preparing data")
    train_df, test_df = prepare.prepare_datasets(df)

    # Train the model
    with open(root_dir / "oilwells" / "parameters.json") as f:
        train_args = json.load(f)["training"]
    print("training models with parameters:", train_args)
    model = train_model(train_df, train_args)
    # Log the metrics for the model
    metrics = get_model_metrics(model, test_df)
    for (k, v) in metrics.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
