"""
Standalone
"""

import datetime
import os
import pickle
import time
import matplotlib
import matplotlib.pyplot
import pandas
import scipy
import sklearn
import sklearn.manifold

data_directory = "../Data/"
figure_directory = "figures/"

_x_limit, _y_limit = 189, 111


def current_time():
    """
    Get current time.

    Wait a second for avoiding duplication, and get current time for file name. Last modified: 2019-11-18T06:48:21+0900

    Args:
        None

    Returns:
        String: the time in the string format
    """
    time.sleep(1)
    return "_" + time.strftime("%m%d%H%M%S")


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

    Get general information of building. Save this with pickle format. Last modified: 2019-11-18T07:01:22+0900

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

        _data["Date/Time"] = pandas.to_datetime(_data["Date/Time"])

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

        _data["Date/Time"] = pandas.to_datetime(_data["Date/Time"])

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

        _data["timestamp"] = pandas.to_datetime(_data["timestamp"])

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

        _data["timestamp"] = pandas.to_datetime(_data["timestamp"])

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def draw_mobile_prox_data(verbose=True):
    """
    Draw mobile prox data for further analysis.

    Last modified: 2019-11-18T07:47:37+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    if verbose:
        print("Fetch data...")
    _data = get_mobile_prox_data()

    if verbose:
        print("Get date/floor")
    _dates = sorted(list(set(pandas.to_datetime(_data["timestamp"]).apply(lambda x: x.date()))))
    _floors = sorted(list(set(_data["floor"])))

    if verbose:
        print(len(_dates), _dates)
        print(len(_floors), _floors)

    for date in _dates:
        date_string = date.strftime("%Y-%m-%d")
        for floor in _floors:
            if verbose:
                print("drawing figure:", date, floor)

            data = _data[(pandas.to_datetime(_data["timestamp"]).apply(lambda x: x.date()) == date) & (_data["floor"] == floor)]

            matplotlib.use("Agg")
            matplotlib.rcParams.update({"font.size": 30})

            matplotlib.pyplot.figure()
            matplotlib.pyplot.scatter(data["x"], data["y"], alpha=0.3, s=200, marker="X")

            matplotlib.pyplot.title("Mobile prox Data in " + date_string + " on " + str(floor) + " Floor")
            matplotlib.pyplot.xlabel("X")
            matplotlib.pyplot.ylabel("Y")
            matplotlib.pyplot.xlim(0, _x_limit)
            matplotlib.pyplot.ylim(0, _y_limit)
            matplotlib.pyplot.grid(True)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(32, 18)
            fig.savefig(figure_directory + "MobileProxData" + current_time() + ".png")

            matplotlib.pyplot.close()
    else:
        if verbose:
            print("Drawing Done!!")


def get_tsne_mobile_prox_data(is_drawing=False, verbose=False):
    """
    Get tsne with mobile prox data.

    Calculate tsne with mobile prox data. Also, save the tnse for futher analysis. Last modified: 2019-11-18T15:10:48+0900

    Args:
        is_drawing (bool): If it is true, this function will draw the tsne plot.
        verbose (bool): Verbosity level

    Returns:
        tsne (DataFrame): DataFrame which contains tsne in two dimension
    """
    _pickle_file = ".tsne_mobile_prox_data.pkl"
    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        if verbose:
            print("Make TSNE")

        data = get_mobile_prox_data()
        data.drop(columns=["type", "prox-id"], inplace=True)
        data["timestamp"] = list(map(lambda x: datetime.datetime.timestamp(x), data["timestamp"]))

        _tsne = pandas.DataFrame(data=sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(data), columns=["TSNE-1", "TSNE-2"])
        _tsne["TSNE-1"] = scipy.stats.zscore(_tsne["TSNE-1"])
        _tsne["TSNE-2"] = scipy.stats.zscore(_tsne["TSNE-2"])
        _tsne["id"] = data.index

        with open(_pickle_file, "wb") as f:
            pickle.dump(_tsne, f)

    if is_drawing:
        if verbose:
            print("Drawing TSNE")

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(_tsne["TSNE-1"], _tsne["TSNE-2"], alpha=0.3, s=100)

        matplotlib.pyplot.title("TSNE of Mobile prox Data")
        matplotlib.pyplot.xlabel("Standardized TSNE-1")
        matplotlib.pyplot.ylabel("Standardized TSNE-2")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "TSNEMobileProxData" + current_time() + ".png")

        matplotlib.pyplot.close()

        if verbose:
            print("Drawing Done!!")

    return _tsne


def draw_tsne_mobile_prox_data_by_value(verbose=False):
    """
    Draw tsne plot with day / floor

    Draw tsne plot with day and floor. Last modified: 2019-11-19T01:59:59+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    _tsne = get_tsne_mobile_prox_data()
    _data = get_mobile_prox_data()

    if verbose:
        print("Drawing day")

    _dates = sorted(list(set(pandas.to_datetime(_data["timestamp"]).apply(lambda x: x.date()))))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for date in _dates:
        date_string = date.strftime("%Y-%m-%d")

        if verbose:
            print(">> Drawing figure:", date_string)

        drawing_data = _tsne[(pandas.to_datetime(_data["timestamp"]).apply(lambda x: x.date()) == date)]

        matplotlib.pyplot.scatter(drawing_data["TSNE-1"], drawing_data["TSNE-2"], alpha=0.3, s=200, label=date_string)

    matplotlib.pyplot.title("Mobile prox Data by Date")
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig(figure_directory + "DateTSNEMobileProxData" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Drawing Date Done!!")

    if verbose:
        print("Drawing Floor")

    _floors = sorted(list(set(_data["floor"])))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for floor in _floors:
        if verbose:
            print(">> Drawing floor:", floor)

        drawing_data = _tsne[(_data["floor"] == floor)]

        matplotlib.pyplot.scatter(drawing_data["TSNE-1"], drawing_data["TSNE-2"], alpha=0.3, s=200, label=str(floor))

    matplotlib.pyplot.title("Mobile prox Data by Floor")
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig(figure_directory + "FloorTSNEMobileProxData" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Drawing Floor Done!!")


if __name__ == "__main__":
    employee_data = get_employee_data(show=True)
    general_data = get_general_data(show=True)
    hazium_data = [get_hazium_data(data, True) for data in get_hazium_data()]
    fixed_prox_data = get_fixed_prox_data(show=True)
    mobile_prox_data = get_mobile_prox_data(show=True)

    # draw_mobile_prox_data(verbose=True)
    # tsne_mobile_prox_data = get_tsne_mobile_prox_data(is_drawing=True, verbose=True)
    draw_tsne_mobile_prox_data_by_value(verbose=True)
