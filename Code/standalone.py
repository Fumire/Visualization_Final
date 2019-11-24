"""
Standalone
"""

import datetime
import math
import multiprocessing
import os
import pickle
import time
import matplotlib
import matplotlib.pyplot
import numpy
import pandas
import pandas.plotting
import scipy
import sklearn
import sklearn.covariance
import sklearn.ensemble
import sklearn.manifold
import sklearn.neighbors
import sklearn.svm

pandas.plotting.register_matplotlib_converters()

data_directory = "../Data/"
figure_directory = "figures/"

_x_limit, _y_limit = 189, 111


def current_time():
    """
    Get current time.

    Get current time for file name. Last modified: 2019-11-19T06:50:09+0900

    Args:
        None

    Returns:
        String: the time in the string format
    """
    return "_" + time.strftime("%m%d%H%M%S")


def statistics(data):
    """
    Basic statistics.

    Give basic statistics along the given data.

    Args:
        data (list): List of values which is used for statistics.

    Returns:
        None
    """
    print("Min:", numpy.nanmin(data))
    print("Max:", numpy.nanmax(data))
    print("Average:", numpy.nanmean(data))
    print("25%, 50%, 75%:", numpy.nanpercentile(data, 25), numpy.nanpercentile(data, 50), numpy.nanpercentile(data, 75))
    print("std:", numpy.nanstd(data))


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


def get_general_zscore_data(show=False):
    """
    Get general data with standardized.

    Get general data with standardized. Save this for further analysis. Last modified: 2019-11-22T02:28:18+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains standardized general building information
    """
    _pickle_file = ".general_zscore_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = get_general_data()
        _data["Date/Time"] = list(map(lambda x: datetime.datetime.timestamp(x), _data["Date/Time"]))

        will_drop = []
        for column in _data.columns:
            if len(pandas.unique(_data[column])) == 1:
                will_drop.append(column)
            else:
                _data[column] = scipy.stats.zscore(_data[column])
        else:
            _data.drop(columns=will_drop, inplace=True)

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

    Get mobile prox data of building. Drop duplicate rows. Save this with pickle format. Last modified: 2019-11-19T06:11:49+0900

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

        _data.drop_duplicates(inplace=True)

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
            fig.savefig(figure_directory + "MobileProxData_" + date_string + current_time() + ".png")

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

        drawing_data = _tsne[list(pandas.to_datetime(_data["timestamp"]).apply(lambda x: x.date()) == date)]

        matplotlib.pyplot.scatter(drawing_data["TSNE-1"], drawing_data["TSNE-2"], alpha=0.3, s=200, label=date_string)

    matplotlib.pyplot.title("Mobile prox Data by Date")
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig(figure_directory + "DateTSNEMobileProxData_" + date_string + current_time() + ".png")

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

        drawing_data = _tsne[list(_data["floor"] == floor)]

        matplotlib.pyplot.scatter(drawing_data["TSNE-1"], drawing_data["TSNE-2"], alpha=0.3, s=200, label=str(floor))

    matplotlib.pyplot.title("Mobile prox Data by Floor")
    matplotlib.pyplot.xlabel("Standardized TSNE-1")
    matplotlib.pyplot.ylabel("Standardized TSNE-2")
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(24, 24)
    fig.savefig(figure_directory + "FloorTSNEMobileProxData_" + str(floor) + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Drawing Floor Done!!")


def get_tsne_general_data(is_drawing=False, verbose=False):
    """
    Get tsne with general data.

    Calculate tsne with general data, and save the tnse for further analysis. Last modified: 2019-11-20T19:16:00+0900

    Args:
        is_drawing (bool): If it is true, this function will draw the tsne plot.
        verbose (bool): Verbosity level.

    Returns:
        tsne (DataFrame): DataFrame which contains the tsne in two dimension.
    """
    _pickle_file = ".tsne_general_data.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        if verbose:
            print("Make TSNE")

        data = get_general_data()
        data["Date/Time"] = list(map(lambda x: datetime.datetime.timestamp(x), data["Date/Time"]))

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

        matplotlib.pyplot.title("TSNE of General Data")
        matplotlib.pyplot.xlabel("Standardized TSNE-1")
        matplotlib.pyplot.ylabel("Standardized TSNE-2")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "TSNEGeneralData" + current_time() + ".png")

        matplotlib.pyplot.close()

        if verbose:
            print("Drawing Done!!")

    return _tsne


def draw_general_data(verbose=False, relative=False):
    """
    Draw general data.

    Draw each data type of general data. Last modified: 2019-11-19T07:03:02+0900

    Args:
        verbose (bool): Verbosity level
        relative (bool): If this is true, draw relative data instead

    Returns:
        None
    """
    if verbose:
        print("Drawing General Data")

    _data = get_general_data()
    _columns = list(_data.columns)

    if verbose:
        print("Total columns:", len(_columns))

    for i, column in enumerate(_columns[1:]):
        if verbose:
            print(">> Drawing:", column)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        if relative:
            matplotlib.pyplot.plot(_data[_columns[0]], scipy.stats.zscore(_data[column]))
        else:
            matplotlib.pyplot.plot(_data[_columns[0]], _data[column])

        matplotlib.pyplot.title("General Data:" + column)
        matplotlib.pyplot.xlabel("Time")
        if relative:
            matplotlib.pyplot.ylabel("Value (Standardized)")
        else:
            matplotlib.pyplot.ylabel("Value")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "GeneralData_" + str(i) + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")


def get_regression_general_data(column, is_drawing=False, verbose=False):
    """
    Regreesion general data. (Actual)

    Get regression with general data by each columns. Save the best regression for further analysis. Returns score to find unusual events. Last modified: 2019-11-21T23:27:40+0900

    Args:
        column (string): Mandatory. column name to regress.
        is_drawing (bool): If this is True, draw the regression plot.
        verbose (bool): Verbosity level

    Returns:
        List: The score of best the algorithm amongst the algorithms which are executed.
    """
    _data = get_general_zscore_data()
    time_data = _data["Date/Time"]
    _values = dict()

    if column not in list(_data.columns)[1:]:
        print("Invalid column:", column)
        raise ValueError

    column_index = list(_data.columns).index(column)
    _pickle_file = ".regression_general_data_" + str(column_index) + ".pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")

        with open(_pickle_file, "rb") as f:
            _value, _values = pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        y_data = _data[column]
        _data.drop(columns=["Date/Time", column], inplace=True)
        _values["Real"] = y_data

        adaboost = sklearn.ensemble.AdaBoostRegressor(random_state=0)
        adaboost.fit(_data, y_data)

        _values["adaboost (%.2f)" % adaboost.score(_data, y_data)] = adaboost.predict(_data)

        _score, _value = adaboost.score(_data, y_data), adaboost.predict(_data)

        bagging = sklearn.ensemble.BaggingRegressor(random_state=0, n_jobs=1)
        bagging.fit(_data, y_data)

        _values["bagging (%.2f)" % bagging.score(_data, y_data)] = bagging.predict(_data)

        if _score < bagging.score(_data, y_data):
            _score, _value = bagging.score(_data, y_data), bagging.predict(_data)

        extratrees = sklearn.ensemble.ExtraTreesRegressor(random_state=0, n_jobs=1)
        extratrees.fit(_data, y_data)

        _values["extratrees (%.2f)" % extratrees.score(_data, y_data)] = extratrees.predict(_data)

        if _score < extratrees.score(_data, y_data):
            _score, _value = extratrees.score(_data, y_data), extratrees.predict(_data)

        gradientboosting = sklearn.ensemble.GradientBoostingRegressor(random_state=0)
        gradientboosting.fit(_data, y_data)

        _values["gradientboosting (%.2f)" % gradientboosting.score(_data, y_data)] = gradientboosting.predict(_data)

        if _score < gradientboosting.score(_data, y_data):
            _score, _value = gradientboosting.score(_data, y_data), gradientboosting.predict(_data)

        randomforest = sklearn.ensemble.RandomForestRegressor(random_state=0, n_jobs=1)
        randomforest.fit(_data, y_data)

        _values["randomforest (%.2f)" % randomforest.score(_data, y_data)] = randomforest.predict(_data)

        if _score < randomforest.score(_data, y_data):
            _score, _value = randomforest.score(_data, y_data), randomforest.predict(_data)

        with open(_pickle_file, "wb") as f:
            pickle.dump((_value, _values), f)

    _csv_file = "csv/regression_" + str(column_index) + ".csv"
    with open(_csv_file, "w") as f:
        names = sorted(list(_values.keys()))
        values = pandas.DataFrame.from_dict(_values)

        f.write(",".join(names))
        f.write("\n")

        for index, row in values.iterrows():
            f.write(",".join(["%6f" % row[name] for name in names]))
            f.write("\n")

    if is_drawing:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        for algorithm in _values:
            matplotlib.pyplot.plot(time_data, _values[algorithm], label=algorithm)

        matplotlib.pyplot.title("Regression of General Data: " + column)
        matplotlib.pyplot.xlabel("Time")
        matplotlib.pyplot.ylabel("Value")
        matplotlib.pyplot.grid(True)
        matplotlib.pyplot.legend()

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "RegressionGeneralData_" + str(column_index) + current_time() + ".png")

        matplotlib.pyplot.close()

    return _value


def regression_all_general_data(verbose=False, processes=100):
    """
    Regression general data. (Command)

    Get regression with general data by each columns. Collect its score and find unusual events. Last modified: 2019-11-20T19:16:14+0900

    Args:
        verbose (bool): Verbosity level
        processes (int): Number of threads

    Returns:

    """
    _data = get_general_zscore_data()

    with multiprocessing.Pool(processes=processes) as pool:
        values = pool.map(get_regression_general_data, list(_data.columns)[1:])

    values = pandas.DataFrame.from_records(values).T
    values.columns = list(_data.columns)[1:]

    print(values)


def draw_movement(verbose=False):
    """
    Draw movement with mobile prox data.

    Draw movement with mobile prox data by each individuals. Start point will be shown as red mark, and final point will be shown as blue mark. Last modified: 2019-11-24T23:24:56+0900
    """
    if verbose:
        print("Drawing movement")

    _data = get_both_prox_data()

    _names = sorted(list(set(list(map(lambda x: x[:-3], _data["prox-id"])))))
    _floors = sorted(list(set(_data["floor"])))

    for num, name in enumerate(_names):
        if verbose:
            print(">> Drawing:", name)

        for floor in _floors:
            if verbose:
                print(">>>> Floor:", floor)

            drawing_data = _data[(_data["prox-id"].str.contains(name)) & (_data["floor"] == floor)]
            x_data = list(drawing_data["x"])
            y_data = list(drawing_data["y"])

            matplotlib.use("Agg")
            matplotlib.rcParams.update({"font.size": 30})

            matplotlib.pyplot.figure()
            for i in range(1, len(x_data)):
                if x_data[i - 1] == x_data[i] and y_data[i - 1] == y_data[i]:
                    continue
                matplotlib.pyplot.arrow(x_data[i - 1], y_data[i - 1], x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1], alpha=0.7, length_includes_head=True, head_width=3, head_length=3, color="k")

            if len(x_data) > 0:
                matplotlib.pyplot.scatter(x_data[0], y_data[0], s=1000, marker="X", c="r")
                matplotlib.pyplot.scatter(x_data[-1], y_data[-1], s=1000, marker="X", c="b")

            matplotlib.pyplot.title("Movement of " + name + " in " + str(floor) + " floor")
            matplotlib.pyplot.xlabel("X")
            matplotlib.pyplot.ylabel("Y")
            matplotlib.pyplot.xlim(0, _x_limit)
            matplotlib.pyplot.ylim(0, _y_limit)
            matplotlib.pyplot.grid(True)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(32, 18)
            fig.savefig(figure_directory + "Movement_" + str(num) + "_" + str(floor) + current_time() + ".png")

            matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")


def change_zone_to_coordinates(changing_data, which=None):
    """
    Change zone data into coordinate.

    Get data with zone data and add coordinate data into original data. Last modified: 2019-11-24T22:19:49+0900

    Args:
        changing_data (DataFrame): Mandatory. The DataFrame which contains zone data.
        which (str): Mandatory. The zone data which is changed. If wrong argument is given, return correct argument.

    Returns:
        DataFrame: which contains coordinate data.
    """
    _data = {"prox": {1: {"1": (91, 56), "2": (24, 91), "3": (8, 40), "4": (79, 56), "5": (97, 94), "6": (96, 34), "7": (160, 8), "8": (182, 8)}, 2: {"1": (62, 56), "2": (19, 59), "3": (152, 103), "4": (79, 56), "5": (29, 35), "6": (129, 25), "7": (144, 81)}, 3: {"1": (59, 56), "2": (105, 56), "3": (15, 40), "4": (79, 56), "5": (165, 56), "6": (17, 100), "Server Room": (28, 23)}}, "energy": {1: {"1": (24, 91), "2": (97, 94), "3": (168, 84), "4": (170, 8), "5": (96, 34), "6": (82, 56), "7": (22, 28), "8": (82, 46)}, 2: {"1": (10, 103), "2": (97, 103), "3": (182, 101), "4": (182, 54), "5": (180, 8), "6": (92, 8), "7": (8, 10), "8": (8, 47), "9": (37, 58), "10": (108, 81), "11": (158, 81), "12A": (88, 84), "12B": (88, 58), "12C": (88, 32), "13": (79, 56), "14": (86, 34), "15": (161, 55), "16": (146, 33)}, 3: {"1": (15, 103), "2": (82, 103), "3": (99, 81), "4": (79, 56), "5": (96, 34), "6": (77, 8), "7": (8, 12), "8": (8, 60), "9": (31, 31), "10": (28, 64), "11A": (72, 83), "11B": (72, 57), "11C": (72, 37), "12": (165, 56)}}}

    if which not in _data:
        print("Select one:", list(_data.keys()))
        raise ValueError

    changing_data["x"] = [_data[which][f][z][0] for f, z in zip(changing_data["floor"], changing_data["zone"])]
    changing_data["y"] = [_data[which][f][z][1] for f, z in zip(changing_data["floor"], changing_data["zone"])]

    return changing_data


def get_both_prox_data(verbose=True):
    """
    Get both prox data.

    Get mobile / fixed prox data. Return merged data. Last modified: 2019-11-24T23:38:16+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        DataFrame: which contains mobile / fixed prox data.
    """
    _pickle_file = ".prox_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            return pickle.load(f)
    else:
        _mobile_data = get_mobile_prox_data()
        _fixed_data = change_zone_to_coordinates(get_fixed_prox_data(), "prox")

        _fixed_data.drop(columns=["zone"], inplace=True)
        _data = pandas.concat([_mobile_data, _fixed_data], ignore_index=True, verify_integrity=True)
        _data.sort_values(by=["timestamp"], inplace=True)

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

        return _data


def calculate_movement(verbose=False):
    """
    Calculate movement.

    Calucate movement for each individual by ID. Save the result for further analysis. Last modified: 2019-11-24T23:01:15+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        Dictionary: Dictionary which contains distance information for each individual
    """
    def calculate(x1, y1, x2, y2):
        return math.sqrt(math.pow(x1 - x2, 2) + math.pow(y1 - y2, 2))

    _pickle_file = ".movement_percentile.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            return pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        _data = get_both_prox_data()

        if verbose:
            print(_data)

        _names = sorted(list(set(list(map(lambda x: x[:-3], _data["prox-id"])))))
        _result = dict()

        for name in _names:
            if verbose:
                print(">> calculate:", name)

            distance = 0

            data = _data[(_data["prox-id"].str.contains(name))]
            x_data = list(data["x"])
            y_data = list(data["y"])

            for i in range(1, len(x_data)):
                distance += calculate(x_data[i - 1], y_data[i - 1], x_data[i], y_data[i])

            _result[name] = distance

        with open(_pickle_file, "wb") as f:
            pickle.dump(_result, f)

        return _result


def draw_movement_distribution(verbose=False):
    """
    Draw movement Distribution.

    Draw movement distribution with individual in sorting. Last modified: 2019-11-20T19:15:30+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    _data = calculate_movement()
    _values = sorted(list(_data.values()), reverse=True)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(_data)), _values)

    matplotlib.pyplot.title("Movement Distribution")
    matplotlib.pyplot.xlabel("Individual")
    matplotlib.pyplot.ylabel("Moving Distance")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "MovementDistribution" + current_time() + ".png")

    matplotlib.pyplot.close()


def get_abnormal_general_data(is_drawing=False, verbose=False):
    """
    Find abnormality in general data.

    Find and calculate abnormality in general data. Save this with pickle format. Last modified: 2019-11-22T05:50:23+0900

    Args:
        is_drawing (bool): If it is true, this function will draw the tsne plot with abnormality
        verbose (bool): Verbosity level.

    Returns:
        DataFrame: which contains TSNE along abnormality.
    """
    _pickle_file = ".abnormal_general_data.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        data = get_general_data()
        data["Date/Time"] = list(map(lambda x: datetime.datetime.timestamp(x), data["Date/Time"]))

        _tsne = get_tsne_general_data()

        elliptic = sklearn.covariance.EllipticEnvelope(random_state=0)
        _tsne["elliptic"] = list(map(lambda x: True if x == -1 else False, elliptic.fit_predict(data)))

        oneclasssvm = sklearn.svm.OneClassSVM(gamma="scale")
        _tsne["oneclasssvm"] = list(map(lambda x: True if x == -1 else False, oneclasssvm.fit_predict(data)))

        isolationforest = sklearn.ensemble.IsolationForest(random_state=0, n_jobs=100)
        _tsne["isolationforest"] = list(map(lambda x: True if x == -1 else False, isolationforest.fit_predict(data)))

        localoutlier = sklearn.neighbors.LocalOutlierFactor(n_jobs=100)
        _tsne["localoutlier"] = list(map(lambda x: True if x == -1 else False, localoutlier.fit_predict(data)))

        with open(_pickle_file, "wb") as f:
            pickle.dump(_tsne, f)

    if verbose:
        print(_tsne)

    if is_drawing:
        if verbose:
            print("Abnormality by Algorithms")

        drawing_data = list()
        for column in list(_tsne.columns)[3:]:
            drawing_data.append(list(map(lambda x: 1 if x else 0, list(_tsne[column]))))

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.pcolor(drawing_data)

        matplotlib.pyplot.title("Abnormality by Algorithms")
        matplotlib.pyplot.xlabel("Time")
        matplotlib.pyplot.ylabel("Algorithms")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks(list(map(lambda x: x + 0.5, range(len(list(_tsne.columns)[3:])))), list(_tsne.columns)[3:])

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "Abnormality" + current_time() + ".png")

        matplotlib.pyplot.close()

        if verbose:
            print("Total Abnormality")

        data = drawing_data[:]
        drawing_data = [0 for _ in data[0]]
        for elements in data:
            for i, element in enumerate(elements):
                drawing_data[i] += element

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.bar(range(len(drawing_data)), drawing_data)

        matplotlib.pyplot.title("Total Abnormality")
        matplotlib.pyplot.xlabel("Time")
        matplotlib.pyplot.ylabel("Abnormality Score")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "TotalAbnormality" + current_time() + ".png")

        matplotlib.pyplot.close()

        for algorithm in list(_tsne.columns)[3:]:
            if verbose:
                print(">> Drawing:", algorithm)

            x_data, o_data = _tsne.loc[(_tsne[algorithm])], _tsne.loc[~(_tsne[algorithm])]

            matplotlib.use("Agg")
            matplotlib.rcParams.update({"font.size": 30})

            matplotlib.pyplot.figure()
            matplotlib.pyplot.scatter(o_data["TSNE-1"], o_data["TSNE-2"], alpha=0.3, s=200, marker="o", label="Inlier")
            matplotlib.pyplot.scatter(x_data["TSNE-1"], x_data["TSNE-2"], alpha=1, s=100, marker="X", label="Outlier")

            matplotlib.pyplot.title("Abnormality: " + algorithm)
            matplotlib.pyplot.xlabel("Standardized TSNE-1")
            matplotlib.pyplot.ylabel("Standardized TSNE-2")
            matplotlib.pyplot.grid(True)
            matplotlib.pyplot.legend()

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(24, 24)
            fig.savefig(figure_directory + "OutlierTSNE_" + algorithm + current_time() + ".png")

            matplotlib.pyplot.close()

    return _tsne


def draw_hazium_data(verbose=False, which=None):
    """

    """
    if which is None:
        which = get_hazium_data()

    try:
        which = list(which)
    except TypeError:
        which = [which]

    for element in which:
        if element not in get_hazium_data():
            raise ValueError("Invalid argument")
    else:
        if verbose:
            print("Good arguments")


if __name__ == "__main__":
    # employee_data = get_employee_data(show=True)
    # general_data = get_general_data(show=True)
    # general_zscore_data = get_general_zscore_data(show=True)
    # hazium_data = [get_hazium_data(data, True) for data in get_hazium_data()]
    # fixed_prox_data = get_fixed_prox_data(show=True)
    # mobile_prox_data = get_mobile_prox_data(show=True)

    # draw_mobile_prox_data(verbose=True)
    # tsne_mobile_prox_data = get_tsne_mobile_prox_data(is_drawing=True, verbose=True)
    # draw_tsne_mobile_prox_data_by_value(verbose=True)
    # tsne_general_data = get_tsne_general_data(is_drawing=True, verbose=True)
    # abnormal_general_data = get_abnormal_general_data(is_drawing=True, verbose=True)
    # draw_general_data(verbose=True, relative=False)
    # draw_movement(verbose=True)
    # draw_hazium_data(verbose=True, which=None)

    # regression_all_general_data(verbose=True, processes=100)

    movement_information = calculate_movement(verbose=True)
    # draw_movement_distribution(verbose=True)
    statistics(list(movement_information.values()))
