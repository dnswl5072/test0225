import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def add_time_to_failure(df):
    # failure if in failed or is being repaired:
    df["failure"] = df.state.str.match("failed|repairing")

    # get time to failure: https://stackoverflow.com/a/53426812
    df = df.sort_values(["id", "time"])

    def hours_to_failure(col):
        def f(x):
            s = x[col].iloc[::-1].cumsum()
            g = s.groupby(s).cumcount().iloc[::-1]
            # if something is failed then we're going to say the time to failure isn't defined
            g[x[col]] = np.nan
            # if the last values are in the non-failed state then their time to failure isn't defined
            if not x.failure.values[-1]:
                g[s == 0] = np.nan
            return g

        return f

    df["hours_to_failure"] = df.groupby("id").apply(hours_to_failure("failure")).values
    for issue in ("cracked_valve", "jammed_rotor", "broken_gear"):
        df["issue_failure"] = df.failure & (df.issue == issue)
        df[f"{issue}_hours_to_failure"] = df.groupby("id").apply(hours_to_failure("issue_failure")).values

    # drop columns we don't want in the data:
    df = df.drop(columns=["failure", "issue_failure"])

    # downsample to daily:
    start, end = df.time.min(), df.time.max()
    idx = pd.date_range(start, end, freq="24H")
    df = df.set_index("time")
    df = df.loc[idx]
    df.index.name = "time"
    df = df.reset_index()
    df = df.sort_values(["id", "time"])

    return df


def prepare_datasets(df):

    # Do all the stuff to add time to failure:
    df = add_time_to_failure(df)

    # Add on our class:
    df["rotor_will_jam_in_next_two_weeks"] = df["jammed_rotor_hours_to_failure"] < 24 * 14

    # Filter to only rows with a non-empty hours_to_failure:
    df = df[~df.jammed_rotor_hours_to_failure.isnull()]

    # Make a train/test split - esure we split by oilwell id so the same well isn't in both:
    train_idx, test_idx = next(
        GroupShuffleSplit(test_size=0.3, n_splits=1, random_state=7).split(df, groups=df.id.values)
    )
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    return train_df, test_df
