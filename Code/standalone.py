"""
Standalone
"""

import os
import pickle
import pandas

data_directory = "../Data/"


def get_employee(show=False):
    """
    Get employee information.

    Get employee information from data directory, and drop useless column. Save this with pickle format. Last modified at 2019-11-16T06:26:29+0900

    Args:
        show (bool): when this is true, show the data before returning

    Returns:
        DataFrame which contains employee data

    """
    _pickle_file = ".employee_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_excel(data_directory + "/Employee List.xlsx")

        _data.drop(columns="Unnamed: 0", inplace=True)

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data)

    return _data


if __name__ == "__main__":
    get_employee(show=True)
