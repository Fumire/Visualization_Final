"""
Standalone
"""

import os
import pickle
import matplotlib
import matplotlib.pyplot
import pandas

data_directory = "../Data/"


def get_employee_data(show=False):
    """
    Get employee information.

    Get employee information from data directory, and drop useless column. Save this with pickle format. Last modified: 2019-11-18T02:11:04+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains employee data

    """
    _pickle_file = ".employee_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_excel(data_directory + "/Employee List.xlsx")

        _data.drop(columns="Unnamed: 0", inplace=True)
        _data.columns = list(map(lambda x: x.strip(), _data.columns))
        _data_obj = _data.select_dtypes(["object"])
        _data[_data_obj.columns] = _data_obj.apply(lambda x: x.str.strip())

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_general_data(show=False):
    """
    Get general building data.

    Get general information of building. Save this with pickle format. Last modified: 2019-11-18T02:10:51+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains general building information
    """
    _pickle_file = ".general_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_csv(data_directory + "BuildingProxSensorData/csv/bldg-MC2.csv")

        _data.columns = list(map(lambda x: x.strip(), _data.columns))
        _data_obj = _data.select_dtypes(["object"])
        _data[_data_obj.columns] = _data_obj.apply(lambda x: x.str.strip())

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_hazium_data(data=None, show=False):
    """
    Get hazium information data.

    Get hazium data of building. Save this data with pickle format. Last modified: 2019-11-18T02:10:45+0900

    Args:
        data (None or Int): select which data to fetch. If this value is default(None), the function will return list of possibilities, rather than data
        show (bool): when this is true, show the data information before returning

    Returns:
        Dictionary: when data argument is None
        DataFrame: Otherwise, returns DataFrame which contains hazium data.
    """
    _data_location = data_directory + "BuildingProxSensorData/csv/"
    _data_index = {0: _data_location + "f1z8a-MC2.csv", 1: _data_location + "f2z2-MC2.csv", 2: _data_location + "f2z4-MC2.csv", 3: _data_location + "f3z1-MC2.csv"}

    if (data is None) or (data not in _data_index):
        return _data_index

    _pickle_file = ".hazium_data_" + str(data) + ".pkl"

    if os.path.isfile(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_csv(_data_index[data])

        _data.columns = list(map(lambda x: x.strip(), _data.columns))
        _data_obj = _data.select_dtypes(["object"])
        _data[_data_obj.columns] = _data_obj.apply(lambda x: x.str.strip())

        _data.columns = ["Date/Time", "Hazium Concentration"]

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_fixed_prox_data(show=False):
    """
    Get fixed prox data.

    Get fixed prox data of building. Save this with pickle format. Last modified: 2019-11-18T02:10:39+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains fixed prox data
    """
    _pickle_file = ".fixed_prox.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_csv(data_directory + "BuildingProxSensorData/csv/proxOut-MC2.csv")

        _data.columns = list(map(lambda x: x.strip(), _data.columns))
        _data_obj = _data.select_dtypes(["object"])
        _data[_data_obj.columns] = _data_obj.apply(lambda x: x.str.strip())

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_mobile_prox_data(show=False):
    """
    Get mobile prox data.

    Get mobile prox data of building. Save this with pickle format. Last modified: 2019-11-18T02:10:31+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains fixed prox data
    """
    _pickle_file = ".mobile_prox.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.read_csv(data_directory + "BuildingProxSensorData/csv/proxMobileOut-MC2.csv")

        _data.columns = list(map(lambda x: x.strip(), _data.columns))
        _data_obj = _data.select_dtypes(["object"])
        _data[_data_obj.columns] = _data_obj.apply(lambda x: x.str.strip())

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


if __name__ == "__main__":
    employee_data = get_employee_data(show=True)

    general_data = get_general_data(show=True)

    hazium_data = [get_hazium_data(data, True) for data in get_hazium_data()]

    fixed_prox_data = get_fixed_prox_data(show=True)

    mobile_prox_data = get_mobile_prox_data(show=True)

    print(sorted(list(set(fixed_prox_data["prox-id"]))))
