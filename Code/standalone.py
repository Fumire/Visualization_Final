"""
Standalone
"""

import pandas

data_directory = "../Data/"


def get_employee(show=False):
    """
    Get employee information.

    Last modified at 2019-11-16T06:14:42+0900

    Args:
        show (bool): when this is true, show the data before returning

    Returns:
        DataFrame which contains employee data

    """
    data = pandas.read_excel(data_directory + "/Employee List.xlsx")

    if show:
        print(data)

    return data


if __name__ == "__main__":
    get_employee(show=True)
