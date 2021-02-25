import argparse
import os

from azureml.core import Dataset, Datastore, Workspace
from azureml.core.run import Run

from prepare import prepare_datasets


def register_dataset(aml_workspace: Workspace, dataset_name: str, datastore_name: str, file_path: str) -> Dataset:
    datastore = Datastore.get(aml_workspace, datastore_name)
    dataset = Dataset.Tabular.from_delimited_files(path=(datastore, file_path))
    dataset = dataset.register(workspace=aml_workspace, name=dataset_name, create_new_version=True)

    return dataset


def main():
    # This is more or less a modified train_aml.py, from the perspective of dealing with loading the original dataset
    print("Running prepare_aml.py")

    parser = argparse.ArgumentParser("prepare")

    parser.add_argument("--dataset_version", type=str, help=("dataset version"))
    parser.add_argument(
        "--data_file_path",
        type=str,
        help=("data file path, if specified, a new version of the dataset will be registered"),
    )
    parser.add_argument("--caller_run_id", type=str, help=("caller run id, for example ADF pipeline run id"))
    parser.add_argument(
        "--dataset_name",
        type=str,
        help=(
            "Dataset name. Dataset must be passed by name to always get the desired dataset version\
              rather than the one used while the pipeline creation"
        ),
    )
    parser.add_argument("--train_ds", type=str, help=("output path for train dataset to be stored"))
    parser.add_argument("--test_ds", type=str, help=("output path for train dataset to be stored"))

    args = parser.parse_args()

    print("Argument [dataset_version]: %s" % args.dataset_version)
    print("Argument [data_file_path]: %s" % args.data_file_path)
    print("Argument [caller_run_id]: %s" % args.caller_run_id)
    print("Argument [dataset_name]: %s" % args.dataset_name)
    print("Argument [train_ds]: %s" % args.train_ds)
    print("Argument [test_ds]: %s" % args.test_ds)

    train_ds = args.train_ds
    test_ds = args.test_ds
    dataset_version = args.dataset_version
    data_file_path = args.data_file_path
    dataset_name = args.dataset_name

    run = Run.get_context()

    # Get the dataset
    if dataset_name:
        if data_file_path == "none":
            dataset = Dataset.get_by_name(run.experiment.workspace, dataset_name, dataset_version)  # NOQA: E402, E501
        else:
            dataset = register_dataset(
                run.experiment.workspace, dataset_name, os.environ.get("DATASTORE_NAME"), data_file_path,
            )
    else:
        e = "No dataset provided"
        print(e)
        raise Exception(e)

    # Link dataset to the step run so it is trackable in the UI
    run.input_datasets["original_data"] = dataset
    run.parent.tag("dataset_id", value=dataset.id)

    # Prepare the datasets
    df = dataset.to_pandas_dataframe()
    train_df, test_df = prepare_datasets(df)

    # Save the datasets to be used by later stages
    os.makedirs(train_ds, exist_ok=True)
    train_df.to_csv(f"{train_ds}/train.csv")
    os.makedirs(test_ds, exist_ok=True)
    test_df.to_csv(f"{test_ds}/test.csv")

    run.tag("run_type", value="prepare")
    print(f"tags now present for run: {run.tags}")

    run.complete()


if __name__ == "__main__":
    main()
