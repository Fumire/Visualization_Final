"""
Standalone
"""

import datetime
import json
import math
import multiprocessing
import os
import pickle
import time
import PIL
import matplotlib
import matplotlib.image
import matplotlib.pyplot
import numpy
import pandas
import pandas.plotting
import scipy
import scipy.signal
import sklearn
import sklearn.covariance
import sklearn.ensemble
import sklearn.manifold
import sklearn.model_selection
import sklearn.neighbors
import sklearn.tree
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


def get_floor_data(floor=None, verbose=False):
    """
    Get floor data from json files.

    Get floor data from json files. Make them ready to use. Save this with pickle format. Last modified: 2019-11-25T09:31:19+0900

    Args:
        floor (int): This should be one of {0, 1, 2, 3}. If wrong argument is given, correct arguments will be shown.
        verbose (bool): Verbosity level

    Returns:
        DataFrame: which contains floor data from json file.
    """
    _file_name = {0: "general", 1: "floor1", 2: "floor2", 3: "floor3"}
    if floor not in _file_name:
        print("Invalid argument:", list(_file_name.keys()))
        raise ValueError

    _pickle_file = ".floor" + str(floor) + ".pkl"
    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        if verbose:
            print("Making:", floor)

        _data_location = data_directory + "BuildingProxSensorData/json/" + _file_name[floor] + "-MC2.json"

        if floor in [0, 1]:
            with open(_data_location, "r") as f:
                _original_data = json.load(f)

            _data = pandas.concat(objs=[pandas.DataFrame.from_dict(data=data["message"], orient="index").T for data in _original_data], ignore_index=True, verify_integrity=True)

            for column in list(set(_data.columns)):
                if column in ["Date/Time", "type"]:
                    continue
                _data[column] = list(map(lambda x: float(x), list(_data[column])))
        else:
            _data = pandas.read_json(_data_location)

        _data["Date/Time"] = pandas.to_datetime(_data["Date/Time"])

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if verbose:
        print(_data.info())

    return _data


def get_employee_data(show=False):
    """
    Get employee information.

    Get employee information from data directory, and drop useless column. Save this with pickle format. Last modified: 2019-11-25T00:35:01+0900

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

        _data["prox-id"] = [(f[0] + l).lower() for l, f in zip(_data["Last Name"], _data["First Name"])]

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

    Get hazium data of building with standardized. Save this data with pickle format. Last modified: 2019-11-27T09:52:45+0900

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

        _data["Hazium Concentration"] = scipy.stats.zscore(_data["Hazium Concentration"])

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


def get_tsne_prox_data(is_drawing=False, verbose=False):
    """
    Get tsne with both prox data.

    Calculate tsne with both prox data. Also, save the tnse for futher analysis. Last modified: 2019-11-25T00:19:34+0900

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

        data = get_both_prox_data()
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

        matplotlib.pyplot.title("TSNE of prox Data")
        matplotlib.pyplot.xlabel("Standardized TSNE-1")
        matplotlib.pyplot.ylabel("Standardized TSNE-2")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "TSNEProxData" + current_time() + ".png")

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
    _tsne = get_tsne_prox_data()
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


def draw_movement(verbose=False, different_alpha=True):
    """
    Draw movement with mobile prox data.

    Draw movement with mobile prox data by each individuals. Start point will be shown as red mark, and final point will be shown as blue mark. Last modified: 2019-11-24T23:24:56+0900

    Args:
        verbose (bool): Verbosity level
        different_alpha (bool): Use different alpha, or not.

    Returns:
        None
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

            img = PIL.Image.open(data_directory + "Building Layout/Prox Zones/F" + str(floor) + ".jpg")
            img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
            img.thumbnail((_x_limit, _y_limit))
            matplotlib.pyplot.imshow(img)

            for i in range(1, len(x_data)):
                if (x_data[i - 1] == x_data[i]) and (y_data[i - 1] == y_data[i]):
                    continue
                matplotlib.pyplot.arrow(x_data[i - 1], y_data[i - 1], x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1], alpha=5 / len(x_data) if different_alpha else 0.5, length_includes_head=True, head_width=3, head_length=3, color="k")

            if x_data and y_data:
                matplotlib.pyplot.scatter(x_data[0], y_data[0], s=1000, marker="o", c="r")
                matplotlib.pyplot.scatter(x_data[-1], y_data[-1], s=1000, marker="X", c="b")

            matplotlib.pyplot.title("Movement of " + name + " in " + str(floor) + " floor")
            matplotlib.pyplot.xlabel("X")
            matplotlib.pyplot.ylabel("Y")
            matplotlib.pyplot.xlim(0, _x_limit)
            matplotlib.pyplot.ylim(0, _y_limit)
            matplotlib.pyplot.grid(True)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(32, 18)
            fig.savefig(figure_directory + "Movement_" + name + "_" + str(floor) + current_time() + ".png")

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

    Calucate movement for each individual by ID. Save the result for further analysis. Last modified: 2019-11-27T03:16:57+0900

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
            _result = pickle.load(f)
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

    if verbose:
        tmp = list(zip(_result.values(), _result.keys()))
        _number = 5

        print("Minima:", _number)
        for name, value in sorted(tmp)[:_number]:
            print(name, "&", value, "\\\\")

        print("Maxima:", _number)
        for name, value in sorted(tmp, reverse=True)[:_number]:
            print(name, "&", value, "\\\\")

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

    Find and calculate abnormality in general data. Save this with pickle format. Last modified: 2019-12-01T00:13:35+0900

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
        fig.savefig(figure_directory + "AbnormalityGeneral" + current_time() + ".png")

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
        fig.savefig(figure_directory + "TotalAbnormalityGeneral" + current_time() + ".png")

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
            fig.savefig(figure_directory + "OutlierTSNEGeneral_" + algorithm + current_time() + ".png")

            matplotlib.pyplot.close()

    return _tsne


def draw_hazium_data(verbose=False, which=None):
    """
    Draw hazium data.

    Draw hazium data which is selected to draw. Last modified: 2019-11-25T08:40:17+0900

    Args:
        verbose (bool): Verbosity level
        which (list of int): List to draw. If wrong argument is given, return correct arguments.

    Returns:
        None
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
            print("Good arguments:", which)

    _data = [list(get_hazium_data(data=element)["Hazium Concentration"]) for element in which]
    _name = [get_hazium_data()[element][get_hazium_data()[element].rfind("/") + 1:] for element in which]

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.pcolor(_data)

    matplotlib.pyplot.title("Hazium Concentration")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Data Type")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks(list(map(lambda x: x + 0.5, which)), _name)
    matplotlib.pyplot.colorbar()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "Hazium" + current_time() + ".png")

    matplotlib.pyplot.close()


def draw_percentile_moving_distribution(verbose=False, minimum=0, maximum=100):
    """
    Moving distribution upon selected percentile.

    Draw movement of selected percentile id. Last modified: 2019-12-01T21:35:13+0900

    Args:
        verbose (bool): Verbosity level
        minimum (int): minimum percentile for drawing
        maximum (int): maximum percentile for drawing

    Returns:
        None
    """
    _moving_data = get_both_prox_data()

    _distance_data = calculate_movement()
    minimum_value, maximum_value = numpy.nanpercentile(list(_distance_data.values()), minimum), numpy.nanpercentile(list(_distance_data.values()), maximum)
    _names = list(filter(lambda x: (_distance_data[x] >= minimum_value) and (_distance_data[x] <= maximum_value), list(_distance_data.keys())))
    floors = sorted(list(set(_moving_data["floor"])))

    if verbose:
        print("Selected IDs:", len(_names), _names)
        print("Min:", minimum_value)
        print("Max:", maximum_value)

    for floor in floors:
        if verbose:
            print(">> Drawing:", floor, "floor")

        x_min, x_max = 0, 0
        y_min, y_max = 0, 0

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        for name in _names:
            drawing_data = _moving_data[(_moving_data["prox-id"].str.contains(name)) & (_moving_data["floor"] == floor)]
            x_data = list(drawing_data["x"])
            y_data = list(drawing_data["y"])

            for i in range(1, len(x_data)):
                if (x_data[i - 1] == x_data[i]) and (y_data[i - 1] == y_data[i]):
                    continue
                dx, dy = x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1]

                matplotlib.pyplot.arrow(0, 0, dx, dy, alpha=1 / len(x_data), length_includes_head=True, head_width=3, head_length=3, color="k")

                x_min, x_max = dx if dx < x_min else x_min, dx if dx > x_max else x_max
                y_min, y_max = dy if dy < y_min else y_min, dy if dy > y_max else y_max

        matplotlib.pyplot.title("Movement Direction/Distance of " + str(minimum) + "%-" + str(maximum) + "%")
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        matplotlib.pyplot.xlim(x_min * 1.1, x_max * 1.1)
        matplotlib.pyplot.ylim(y_min * 1.1, y_max * 1.1)
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "GeneralMovement_" + str(floor) + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")

    for floor in floors:
        if verbose:
            print("Drawing:", floor, "floor")

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()

        img = PIL.Image.open(data_directory + "Building Layout/Prox Zones/F" + str(floor) + ".jpg")
        img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
        img.thumbnail((_x_limit, _y_limit))
        matplotlib.pyplot.imshow(img)

        for name in _names:
            drawing_data = _moving_data[(_moving_data["prox-id"].str.contains(name)) & (_moving_data["floor"] == floor)]
            x_data = list(drawing_data["x"])
            y_data = list(drawing_data["y"])

            for i in range(1, len(x_data)):
                matplotlib.pyplot.arrow(x_data[i - 1], y_data[i - 1], x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1], alpha=1 / len(x_data), length_includes_head=True, head_width=3, head_length=3, color="k")

            if x_data and y_data:
                matplotlib.pyplot.scatter(x_data[0], y_data[0], s=1000, marker="o", c="r")
                matplotlib.pyplot.scatter(x_data[-1], y_data[-1], s=1000, marker="X", c="b")

        matplotlib.pyplot.title("Movement from " + str(minimum) + "% to " + str(maximum) + "%")
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        matplotlib.pyplot.xlim(0, _x_limit)
        matplotlib.pyplot.ylim(0, _y_limit)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "PercentileMovement_" + str(minimum) + "_" + str(maximum) + "_" + str(floor) + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")

    if verbose:
        _employee_data = get_employee_data()
        _employee_data = _employee_data.loc[(_employee_data["prox-id"].isin(_names))]

        _department_information = dict()
        for department in set(_employee_data["Department"]):
            _department_information[department] = _employee_data.loc[_employee_data["Department"] == department].count()["Department"]

        print(_department_information)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.pie(_department_information.values(), labels=_department_information.keys(), autopct="%1.1f%%")

        matplotlib.pyplot.title("Department Distribution of " + str(minimum) + "%-" + str(maximum) + "%")

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "DepartmentDistribution" + current_time() + ".png")

        matplotlib.pyplot.close()


def get_tsne_floor_data(floor=None, is_drawing=False, verbose=False):
    """
    Get TSNE data by floor.

    Get TSNE data by each floor. Last modified: 2019-12-01T00:17:01+0900

    Args:
        floor (int): specify the floor to draw.
        is_drawing (bool): If this is true, draw the TSNE data.
        verbose (bool): Verbosity level

    Returns:
        DataFrame: which contains TSNE data of specified floor.
    """
    _file_name = {0: "general", 1: "floor1", 2: "floor2", 3: "floor3"}
    if floor not in _file_name:
        print("Invalid argument:", list(_file_name.keys()))
        raise ValueError

    _pickle_file = ".tsne_floor" + str(floor) + ".pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        _data = get_floor_data(floor=floor)
        _data.drop(columns=["type"], inplace=True)
        _data["Date/Time"] = list(map(lambda x: datetime.datetime.timestamp(x), _data["Date/Time"]))

        _tsne = pandas.DataFrame(data=sklearn.manifold.TSNE(n_components=2, random_state=0).fit_transform(_data), columns=["TSNE-1", "TSNE-2"])
        _tsne["TSNE-1"] = scipy.stats.zscore(_tsne["TSNE-1"])
        _tsne["TSNE-2"] = scipy.stats.zscore(_tsne["TSNE-2"])
        _tsne["id"] = _data.index

        with open(_pickle_file, "wb") as f:
            pickle.dump(_tsne, f)

    if is_drawing:
        if verbose:
            print("Drawing TSNE")

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(_tsne["TSNE-1"], _tsne["TSNE-2"], alpha=0.3, s=100)

        matplotlib.pyplot.title("TSNE of " + _file_name[floor])
        matplotlib.pyplot.xlabel("Standardized TSNE-1")
        matplotlib.pyplot.ylabel("Standardized TSNE-2")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "TsneFloor_" + _file_name[floor] + current_time() + ".png")

        matplotlib.pyplot.close()

    return _tsne


def r_value(x, y):
    """
    Return r_value.

    This function will calculate R value. Reference: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.linregress.html. Last modified: 2019-11-25T21:23:30+0900

    Args:
        x (list): List of values.
        y (list): List of values.

    Returns:
        float: which is calculated R value
    """
    return scipy.stats.linregress(x, y)[2]


def draw_correlation_with_general_data(verbose=False, processes=100):
    """
    Draw correlation with general data.

    Draw correlation with each general data. Draw heatmap about R values between all columns. Last modified: 2019-12-01T22:04:48+0900

    Args:
        verbose (bool): Verbosity level
        processes (int): Number of threads.

    Returns:
        DataFrame: which contains R values between all columns
    """
    _pickle_file = ".correlation_general.pkl"
    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _values = pickle.load(f)
    else:
        if verbose:
            print("Calculate R value")

        _general_data = get_general_zscore_data()
        _general_data.drop(columns=["Date/Time"], inplace=True)

        _columns = sorted(list(_general_data.columns))
        _values = list()

        with multiprocessing.Pool(processes=processes) as pool:
            for x in _columns:
                if verbose:
                    print(">> Calculate:", x)
                data = _general_data.sort_values(by=x, kind="mergesort")
                _values.append(pool.starmap(r_value, [(data[x], data[y]) for y in _columns]))

        _values = pandas.DataFrame(data=_values, index=_columns, columns=_columns)

        with open(_pickle_file, "wb") as f:
            pickle.dump(_values, f)

    if verbose:
        print("Heatmap")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.pcolor(_values)

    matplotlib.pyplot.title("Correlation within General Data")
    matplotlib.pyplot.xlabel("Columns")
    matplotlib.pyplot.ylabel("Columns")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks([])
    matplotlib.pyplot.colorbar()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 24)
    fig.savefig(figure_directory + "CorrelationGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Bar graph")

    flat = _values.values.flatten()

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(flat)), sorted(flat, reverse=True))

    matplotlib.pyplot.title("R-value Distribution")
    matplotlib.pyplot.xlabel("Index")
    matplotlib.pyplot.ylabel("R-value")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "RvalueDistributionGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        statistics(flat)

    _columns = sorted(list(_values.columns))
    _extrema = list()
    _number = 5
    _general_data = get_general_zscore_data()

    if verbose:
        for x in _columns:
            for y in _columns:
                if x == y:
                    break
                _extrema.append((_values[x][y], x, y))

        print("Minima:", _number)
        for v, x, y in sorted(_extrema)[:_number]:
            print(x, "&", y, "&", v, "\\\\")

        print("Maxima:", _number)
        for v, x, y in sorted(_extrema, reverse=True):
            if v != 1:
                break
            print(x, "&", y, "&", v, "\\\\")

    if verbose:
        print("Draw negative correlation")

    for rank in range(_number):
        if verbose:
            print("Rank:", rank)

        v, x, y = sorted(_extrema)[rank]
        data = [_general_data[x], _general_data[y]]
        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(data)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(data[0], data[1], s=100, alpha=0.3, marker="o", c="b")
        matplotlib.pyplot.plot(numpy.arange(sorted(data[0])[0], sorted(data[0])[-1], 0.01), slope * numpy.arange(sorted(data[0])[0], sorted(data[0])[-1], 0.01) + intercept, alpha=1, c="k")

        matplotlib.pyplot.title("Correlation: " + "%.2f" % rvalue)
        matplotlib.pyplot.xlabel(x)
        matplotlib.pyplot.ylabel(y)
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "NegativeCorrelation_" + x + "_" + y + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")

    if verbose:
        print("Draw positive correlation")

    for rank in range(_number):
        if verbose:
            print("Rank:", rank)

        v, x, y = sorted(_extrema)[-rank - 1]
        data = [_general_data[x], _general_data[y]]
        slope, intercept, rvalue, pvalue, stderr = scipy.stats.linregress(data)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.scatter(data[0], data[1], s=100, alpha=0.3, marker="o", c="b")
        matplotlib.pyplot.plot(numpy.arange(sorted(data[0])[0], sorted(data[0])[-1], 0.01), slope * numpy.arange(sorted(data[0])[0], sorted(data[0])[-1], 0.01) + intercept, alpha=1, c="k")

        matplotlib.pyplot.title("Correlation: " + "%.2f" % rvalue)
        matplotlib.pyplot.xlabel(x)
        matplotlib.pyplot.ylabel(y)
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "PositiveCorrelation_" + x + "_" + y + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")

    with open("csv/CorrelationGeneral.csv", "w") as f:
        columns = sorted(list(_values.keys()))
        f.write("Index,")
        f.write(",".join(columns))
        f.write("\n")
        for column in columns:
            f.write(column)
            f.write(",")
            f.write(",".join([str(_values[column][another]) for another in columns]))
            f.write("\n")

    return _values


def draw_stacked_general_data(verbose=False):
    """
    Draw general builidng data in stacked form.

    Draw general building data in stacked form. Last modified: 2019-11-27T00:55:32+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    if verbose:
        print("Stacked General Data")

    _general_data = get_general_zscore_data()
    _x_data = _general_data["Date/Time"]
    _general_data.drop(columns=["Date/Time"], inplace=True)

    _columns = sorted(list(_general_data.columns))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for column in _columns:
        if verbose:
            print(">>", column)
        matplotlib.pyplot.plot(_x_data, _general_data[column], alpha=1 / len(_columns), c="b")

    matplotlib.pyplot.title("Stacked General Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Value (Standardized)")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.ylim(-5, 5)
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "StackedGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")


def draw_average_general_data(verbose=False):
    """
    Draw average value of general data.

    Draw average value of general data with timeline. Last modifided: 2019-11-27T00:45:33+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """

    _general_data = get_general_zscore_data()
    _x_data = _general_data["Date/Time"]
    _general_data.drop(columns=["Date/Time"], inplace=True)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(_x_data, _general_data.mean(axis=1))

    matplotlib.pyplot.title("Average of General Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Value (Standardized)")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "AverageGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()


def draw_median_general_data(verbose=False):
    """
    Draw median value of general data.

    Draw median value of general data with timeline. Last modified: 2019-11-27T00:46:20+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _x_data = _general_data["Date/Time"]
    _general_data.drop(columns=["Date/Time"], inplace=True)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(_x_data, _general_data.median(axis=1))

    matplotlib.pyplot.title("Median of General Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Value (Standardized)")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "MedianGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()


def draw_ultimate_general_data(verbose=True):
    """
    Draw statistics value of general data.

    Draw statistics value of general data, such as average, median, quantile value. Last modified: 2019-11-27T00:47:20+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _x_data = _general_data["Date/Time"]
    _general_data.drop(columns=["Date/Time"], inplace=True)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(_x_data, _general_data.mean(axis=1), label="average")
    matplotlib.pyplot.plot(_x_data, _general_data.median(axis=1), label="median")
    matplotlib.pyplot.plot(_x_data, _general_data.quantile(q=0.25, axis="columns"), label="q1")
    matplotlib.pyplot.plot(_x_data, _general_data.quantile(q=0.75, axis="columns"), label="q3")

    matplotlib.pyplot.title("Statistics of General Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Value (Standardized)")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "UltimateGeneral" + current_time() + ".png")

    matplotlib.pyplot.close()


def get_all_hazium_data(show=False):
    """
    Get all hazium data.

    Get all hazium data from all files. Save this with pickle format. Last modified: 2019-11-27T09:41:59+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains all of the hazium data
    """
    _pickle_file = ".all_hazium_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = pandas.DataFrame()

        for i, argument in enumerate(get_hazium_data().keys()):
            tmp = get_hazium_data(data=argument)
            _data[i] = tmp["Hazium Concentration"]
        else:
            _data["Date/Time"] = tmp["Date/Time"]

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_all_hazium_zscore_data(show=False):
    """
    Get all zscore Hazium data

    get all standardized hazium data from all file. Save this with pickle format. Last modified: 2019-12-02T07:05:53+0900

    Args:
        show (bool): when this is true, show the data information before returning

    Returns:
        DataFrame: which contains standardized hazium data
    """
    _pickle_file = ".all_hazium_zscore_data.pkl"

    if os.path.exists(_pickle_file):
        with open(_pickle_file, "rb") as f:
            _data = pickle.load(f)
    else:
        _data = get_all_hazium_data()
        _data.drop(columns=["Date/Time"], inplace=True)

        for column in _data.columns:
            _data[column] = scipy.stats.zscore(_data[column])

        with open(_pickle_file, "wb") as f:
            pickle.dump(_data, f)

    if show:
        print(_data.info())

    return _data


def get_tsne_hazium_data(is_drawing=False, verbose=False):
    """
    Get TSNE hazium data.

    Get TSNE with hazium data. Save this with pickle format. Last modified: 2019-11-27T09:42:47+0900

    Args:
        is_drawing (bool): If it is true, this function will draw the tsne plot
        verbose (bool): Verbosity level

    Returns:
        DataFrame: which contains TSNE data of hazium data
    """
    _pickle_file = ".tsne_hazium_data.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        if verbose:
            print("Make TSNE")

        data = get_all_hazium_data()
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

        matplotlib.pyplot.title("TSNE of Hazium Data")
        matplotlib.pyplot.xlabel("Standardized TSNE-1")
        matplotlib.pyplot.ylabel("Standardized TSNE-2")
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "TSNEHaziumData" + current_time() + ".png")

        matplotlib.pyplot.close()

        if verbose:
            print("Drawing Done!!")

    return _tsne


def get_abnormal_hazium_data(is_drawing=False, verbose=False):
    """
    Get abnormality of hazium data.

    Get abnormality of all hazium data. Save this with pickle format. Last modified: 2019-11-27T09:45:07+0900

    Args:
        is_drawing (bool): If it is true, this function will draw the many plot according to abnormality
        verbose (bool): Verbosity level

    Returns:
        DataFrame: which contains abnormality with TSNE data
    """
    _pickle_file = ".abnormal_hazium_data.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _tsne = pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        data = get_all_hazium_data()
        data["Date/Time"] = list(map(lambda x: datetime.datetime.timestamp(x), data["Date/Time"]))

        _tsne = get_tsne_hazium_data()

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
        fig.savefig(figure_directory + "AbnormalityHazium" + current_time() + ".png")

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
        fig.savefig(figure_directory + "TotalAbnormalityHazium" + current_time() + ".png")

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
            fig.savefig(figure_directory + "OutlierTSNEHazium_" + algorithm + current_time() + ".png")

            matplotlib.pyplot.close()

    return _tsne


def draw_correlation_with_hazium_data(verbose=False, processes=100):
    """

    """
    _hazium_data = get_all_hazium_data()
    _hazium_data.drop(columns=["Date/Time"], inplace=True)

    _pickle_file = ".correlation_hazium.pkl"
    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            _values = pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        _general_data = get_general_zscore_data()

        _columns = sorted(list(_general_data.columns))
        _values = list()

        with multiprocessing.Pool(processes=processes) as pool:
            for x in sorted(list(_hazium_data.columns)):
                if verbose:
                    print(">> Calculating:", x)
                _values.append(pool.starmap(r_value, [(_hazium_data[x], _general_data[y]) for y in _columns]))

        _values = pandas.DataFrame(data=_values, columns=_columns)

        with open(_pickle_file, "wb") as f:
            pickle.dump(_values, f)

    if verbose:
        print("Heatmap")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.pcolor(_values)

    matplotlib.pyplot.title("Correlation between General Data & Hazium Data")
    matplotlib.pyplot.xlabel("General Data")
    matplotlib.pyplot.ylabel("Hazium Data")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks(list(map(lambda x: x + 0.5, range(len(_hazium_data.columns)))), _hazium_data.columns)
    matplotlib.pyplot.colorbar()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(30, 24)
    fig.savefig(figure_directory + "CorrelationHazium" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Bar graph")

    flat = _values.values.flatten()

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(flat)), sorted(flat, reverse=True))

    matplotlib.pyplot.title("R-value Distribution")
    matplotlib.pyplot.xlabel("Index")
    matplotlib.pyplot.ylabel("R-value")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "RvalueDistributionHazium" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        statistics(flat)


def compare_abnormality(verbose=True):
    """
    Compare abnormality.

    Compare abnormality between general building data and hazium data. Draw the comparing for analysis. Last modified: 2019-11-29T12:08:57+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    abnormal_general_data = get_abnormal_general_data()
    abnormal_hazium_data = get_abnormal_hazium_data()

    algorithms = list(abnormal_general_data.columns)[3:]

    _values = list()

    if verbose:
        print("Calculating")

    for algorithm in algorithms:
        if verbose:
            print(">>", algorithm)
        general = list(map(lambda x: 1 if x else 0, abnormal_general_data[algorithm]))
        hazium = list(map(lambda x: 1 if x else 0, abnormal_hazium_data[algorithm]))

        _values.append(list(map(lambda x: x[0] + x[1], zip(general, hazium))))

    if verbose:
        print("Drawing Heatmap")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.pcolor(_values)

    matplotlib.pyplot.title("Abnormality between General Data and Hazium Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Algorithms")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.yticks(list(map(lambda x: x + 0.5, range(len(algorithms)))), algorithms)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "AbnormalBothHeatmap" + current_time() + ".png")

    matplotlib.pyplot.close()


def draw_movement_department(verbose=False):
    """
    Draw movement data by department.

    Draw movement data by department. Last modified: 2019-11-29T12:07:57+0900

    Args:
        verbose (bool): Verbosity level

    Returns:
        None
    """
    _employee_data = get_employee_data()
    _movement_data = calculate_movement()
    _prox_data = get_both_prox_data()

    _employee_data["movement"] = list(map(lambda x: _movement_data[x] if x in _movement_data else 0, _employee_data["prox-id"]))

    departments = sorted(list(set(_employee_data["Department"])))

    for department in departments:
        if verbose:
            print(">>", department)

        names = sorted(list(set(_employee_data.loc[(_employee_data["Department"] == department)]["prox-id"])))
        floors = sorted(list(set(_prox_data["floor"])))

        for floor in floors:
            if verbose:
                print(">>>>", floor)

            matplotlib.use("Agg")
            matplotlib.rcParams.update({"font.size": 30})

            matplotlib.pyplot.figure()
            for name in names:
                drawing_data = _prox_data.loc[(_prox_data["floor"] == floor) & (_prox_data["prox-id"].str.contains(name))]
                x_data = list(drawing_data["x"])
                y_data = list(drawing_data["y"])

                img = PIL.Image.open(data_directory + "Building Layout/Prox Zones/F" + str(floor) + ".jpg")
                img = img.transpose(PIL.Image.FLIP_TOP_BOTTOM)
                img.thumbnail((_x_limit, _y_limit))
                matplotlib.pyplot.imshow(img)

                for i in range(1, len(x_data)):
                    if (x_data[i - 1] == x_data[i]) and (y_data[i - 1] == y_data[i]):
                        continue
                    matplotlib.pyplot.arrow(x_data[i - 1], y_data[i - 1], x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1], alpha=1 / len(x_data), length_includes_head=True, head_width=3, head_length=3, color="k")

                if x_data and y_data:
                    matplotlib.pyplot.scatter(x_data[0], y_data[0], s=1000, marker="o", c="r")
                    matplotlib.pyplot.scatter(x_data[-1], y_data[-1], s=1000, marker="X", c="b")

            matplotlib.pyplot.title("Movment of " + str(department) + " Department on " + str(floor) + " floor")
            matplotlib.pyplot.xlabel("X")
            matplotlib.pyplot.ylabel("Y")
            matplotlib.pyplot.xlim(0, _x_limit)
            matplotlib.pyplot.ylim(0, _y_limit)

            fig = matplotlib.pyplot.gcf()
            fig.set_size_inches(32, 18)
            fig.savefig(figure_directory + "MovementDepartment_" + str(department) + "_" + str(floor) + current_time() + ".png")

            matplotlib.pyplot.close()

    if verbose:
        print("Done!!")


def draw_first_quarter_general_data(verbose=False, cycle=288):
    """
    Draw plot about first quarter general building data.

    Draw plot about first quater of general building data with given period. Last modified: 2019-12-02T00:51:34+0900

    Args:
        verbose (bool): Verbosity level
        cycle (int): Period to draw

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.head(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)
    _general_data["mean"] = _general_data.mean(axis=1)
    _general_data["median"] = _general_data.median(axis=1)
    _general_data["q1"] = _general_data.quantile(q=0.25, axis="columns")
    _general_data["q3"] = _general_data.quantile(q=0.75, axis="columns")

    if verbose:
        print(_general_data)

    for column in ["mean", "median", "q1", "q3"]:
        if verbose:
            print(column)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection="polar")
        for i in range(_general_data.shape[0] // cycle):
            ax.plot(2 * numpy.pi * numpy.arange(0, 1, 1 / cycle), _general_data[column][cycle * i:cycle * (i + 1)])

        matplotlib.pyplot.title("Cycle: " + str(cycle))

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "Polar_" + column + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Done!!")


def draw_second_quarter_general_data_actual(column, is_drawing=False):
    """
    Draw general building data in second-quarter. (Actual)

    Calcualte and draw the peak of general building data in given column. Last modified: 2019-12-02T02:39:17+0900

    Args:
        column (string): Mandatory. The column to be calculated the peak.
        is_drawing (bool): If this is true, draw the peak plot for analysis.

    Returns:
        list: which contains the width of every peak.
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.head(n=_general_data.shape[0] * 2 // 4).tail(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)
    _general_data["mean"] = _general_data.mean(axis=1)
    _general_data["median"] = _general_data.median(axis=1)
    _general_data["q1"] = _general_data.quantile(q=0.25, axis="columns")
    _general_data["q3"] = _general_data.quantile(q=0.75, axis="columns")

    peak_points, peak_dict = scipy.signal.find_peaks(_general_data[column], width=1)

    if is_drawing:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(range(len(_general_data[column])), _general_data[column])
        matplotlib.pyplot.scatter(peak_points, _general_data[column].iloc[peak_points], marker="X", s=100, c="r")

        matplotlib.pyplot.title(column)
        matplotlib.pyplot.xlabel("Time")
        matplotlib.pyplot.ylabel("Value (Standardized)")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "SecondQuarters_" + column + current_time() + ".png")

        matplotlib.pyplot.close()

    return list(peak_dict["widths"])


def draw_second_quarter_general_data(verbose=False, processes=100):
    """
    Draw general building data in second quarter. (Headquater)

    Draw general building data in second quarter. Last modified: 2019-12-02T04:58:28+0900

    Args:
        verbose (bool): Verbosity level
        processes (int): number of threads

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.head(n=_general_data.shape[0] * 2 // 4).tail(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)

    if verbose:
        print(_general_data)

    with multiprocessing.Pool(processes=processes) as pool:
        _width_lists = pool.map(draw_second_quarter_general_data_actual, _general_data.columns)

    for column in ["mean", "median", "q1", "q3"]:
        draw_second_quarter_general_data_actual(column, is_drawing=True)

    _widths = list()
    for l in _width_lists:
        _widths += list(l)

    if verbose:
        statistics(_widths)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(_widths)), sorted(_widths, reverse=True))

    matplotlib.pyplot.title("Peak Width Distribution")
    matplotlib.pyplot.xlabel("Counts")
    matplotlib.pyplot.ylabel("Peak Width")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "SecondQuaters_Width" + current_time() + ".png")

    matplotlib.pyplot.close()


def draw_third_quarter_general_data(verbose=False, cycle=288):
    """
    Draw general data in third quarter.

    Draw thir quarter of general building data.

    Args:
        verbose (bool): Verbosity level
        cycle (int): Cycle of polar plot

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.head(n=_general_data.shape[0] * 3 // 4).tail(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)
    _general_data["mean"] = _general_data.mean(axis=1)
    _general_data["median"] = _general_data.median(axis=1)
    _general_data["q1"] = _general_data.quantile(q=0.25, axis="columns")
    _general_data["q3"] = _general_data.quantile(q=0.75, axis="columns")

    for column in ["mean", "median", "q1", "q3"]:
        if verbose:
            print(">>", column)
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        fig = matplotlib.pyplot.figure()
        ax = fig.add_subplot(111, projection="polar")
        for i in range(_general_data.shape[0] // cycle):
            ax.plot(2 * numpy.pi * numpy.arange(0, 1, 1 / cycle), _general_data[column][cycle * i:cycle * (i + 1)])

        matplotlib.pyplot.title(column + " / Period: " + str(cycle))

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(24, 24)
        fig.savefig(figure_directory + "ThirdQuarter_" + column + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Done!!")


def draw_fourth_quarter_general_data_actual(column, is_drawing=False):
    """
    Draw general building data in fourth quarter. (Actual)

    Calculate and draw the under-peak of general building data in given column. Last modified: 2019-12-02T04:57:32+0900

    Args:
        column (string): Mandatory. The column to be calculated the under-peak.
        is_drawing (bool): If this is true, draw the under-peak plot of analysis

    Returns:
        list: which contains the width of every peak.
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.tail(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)
    _general_data["mean"] = _general_data.mean(axis=1)
    _general_data["median"] = _general_data.median(axis=1)
    _general_data["q1"] = _general_data.quantile(q=0.25, axis="columns")
    _general_data["q3"] = _general_data.quantile(q=0.75, axis="columns")

    peak_points, peak_dict = scipy.signal.find_peaks(-_general_data[column], width=1)

    if is_drawing:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.plot(range(len(_general_data[column])), _general_data[column])
        matplotlib.pyplot.scatter(peak_points, _general_data[column].iloc[peak_points], marker="X", s=100, c="r")

        matplotlib.pyplot.title(column)
        matplotlib.pyplot.xlabel("Time")
        matplotlib.pyplot.ylabel("Value (Standardized)")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "FourthQuarters_" + column + current_time() + ".png")

        matplotlib.pyplot.close()

    return list(peak_dict["widths"])


def draw_fourth_quarter_general_data(verbose=False, processes=100):
    """
    Draw general building data in fourth quarter. (Headquater)

    Draw general building data in fourth quarter. Last modified: 2019-12-02T04:58:53+0900

    Args:
        verbose (bool): Verbosity level
        processes (int): Number of processes

    Returns:
        None
    """
    _general_data = get_general_zscore_data()
    _general_data = _general_data.tail(n=_general_data.shape[0] // 4)

    _general_data.drop(columns=["Date/Time"], inplace=True)

    if verbose:
        print(_general_data)

    _widths = list()
    with multiprocessing.Pool(processes=processes) as pool:
        for l in pool.map(draw_fourth_quarter_general_data_actual, _general_data.columns):
            _widths += l

    for column in ["mean", "median", "q1", "q3"]:
        draw_fourth_quarter_general_data_actual(column, is_drawing=True)

    if verbose:
        statistics(_widths)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(_widths)), sorted(_widths, reverse=True))

    matplotlib.pyplot.title("Peak Width Distribution")
    matplotlib.pyplot.xlabel("Counts")
    matplotlib.pyplot.ylabel("Peak Width")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "FourthQuaters_Width" + current_time() + ".png")

    matplotlib.pyplot.close()


def calculate_abnormality_score(verbose=False):
    """
    Calculate abnormality score.

    Calcuate abnormality score for choosing the best algorithm for finding abnormality

    Args:
        verbose (bool): Verbosity level

    Returns:
        Dict of Dict: which contains score
    """
    _pickle_file = ".abnormality_score.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            score = pickle.load(f)
    else:
        abnormal_general_data = get_abnormal_general_data()
        general_data = get_general_zscore_data()

        general_data.drop(columns=["Date/Time"], inplace=True)

        algorithms = list(abnormal_general_data.columns)[3:]
        score = dict()

        classifiers = [("KNeighbor", sklearn.neighbors.KNeighborsClassifier(n_jobs=100)), ("SVC", sklearn.svm.SVC(random_state=0)), ("DecisionTree", sklearn.tree.DecisionTreeClassifier(random_state=0)), ("RandomForest", sklearn.ensemble.RandomForestClassifier(random_state=0, n_jobs=100)), ("AdaBoost", sklearn.ensemble.AdaBoostClassifier(random_state=0))]

        if verbose:
            print("Make score")

        for algorithm in algorithms:
            if verbose:
                print(">>", algorithm)

            score[algorithm] = dict()
            abnormal_data = list(map(lambda x: 1 if x else 0, list(abnormal_general_data[algorithm])))

            x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(general_data, abnormal_data, test_size=0.2, random_state=0)

            for name, clf in classifiers:
                if verbose:
                    print(">>>>", name)
                clf.fit(x_train, y_train)
                score[algorithm][name] = clf.score(x_test, y_test)

        if verbose:
            print("Done!!")

        with open(_pickle_file, "wb") as f:
            pickle.dump(score, f)

    if verbose:
        print("&", " & ".join(sorted(score.keys())), "\\\\")
        for algo in score[list(score.keys())[0]].keys():
            print(algo, "&", " & ".join(["%.3f" % score[key][algo] for key in sorted(score.keys())]), "\\\\")
        print("Mean &", " & ".join(["%.4f" % numpy.mean(list(score[key].values())) for key in sorted(score.keys())]), "\\\\")

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    for i, algorithm in enumerate(score):
        matplotlib.pyplot.bar(numpy.arange(len(score[algorithm])) + (i - len(score) // 2) * 0.2, list(score[algorithm].values()), width=0.2, label=algorithm)

    matplotlib.pyplot.title("Scores of Abnormality")
    matplotlib.pyplot.xlabel("Abnormality Algoritms")
    matplotlib.pyplot.ylabel("Score")
    matplotlib.pyplot.xticks(numpy.arange(len(score[algorithm])), score[algorithm].keys())
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "Scoring" + current_time() + ".png")

    matplotlib.pyplot.close()

    return score


def compare_column_actual(column):
    """
    """
    algorithm = "localoutlier"
    abnormal_index = get_abnormal_general_data()[algorithm]

    general_data = get_general_zscore_data()

    return abs(numpy.mean(general_data[abnormal_index][column]) - numpy.mean(general_data[~(abnormal_index)][column]))


def compare_column(verbose=False, processes=100):
    """
    """
    general_data = get_general_zscore_data()

    general_data.drop(columns=["Date/Time"], inplace=True)

    with multiprocessing.Pool(processes=processes) as pool:
        pvalues = [(a, b) for a, b in zip(pool.map(compare_column_actual, sorted(general_data.columns)), sorted(general_data.columns))]
    pvalues = sorted(pvalues, reverse=True)

    if verbose:
        print(pvalues)
        statistics(list(map(lambda x: x[0], pvalues)))

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.bar(range(len(pvalues)), list(map(lambda x: x[0], pvalues)))

    matplotlib.pyplot.title("Differences of Mean Distribution")
    matplotlib.pyplot.xlabel("Columns")
    matplotlib.pyplot.ylabel("Differences of Mean")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "dMeanDistribution" + current_time() + ".png")

    matplotlib.pyplot.close()

    if verbose:
        print("Before filtering:", len(pvalues))
    pvalues = list(filter(lambda x: x[0] > 0.5, pvalues))
    if verbose:
        print("After filtering:", len(pvalues))

    return list(map(lambda x: x[1], pvalues))


def compare_general_hazium(verbose=False, processes=100):
    """
    """
    general_data = get_general_zscore_data()
    hazium_data = get_all_hazium_data()

    general_data.drop(columns=["Date/Time"], inplace=True)
    hazium_data.drop(columns=["Date/Time"], inplace=True)

    abnormal_index = get_abnormal_general_data()["localoutlier"]

    selected_columns = compare_column()
    # selected_columns = sorted(general_data.columns)

    with multiprocessing.Pool(processes=processes) as pool:
        normal_values = list()
        abnormal_values = list()
        abnormal_increasing_values = list()
        abnormal_decreasing_values = list()

        for x in hazium_data.columns:
            normal_values.append(pool.starmap(r_value, [(hazium_data[~(abnormal_index)][x], general_data[~(abnormal_index)][y]) for y in selected_columns]))
            abnormal_values.append(pool.starmap(r_value, [(hazium_data[(abnormal_index)][x], general_data[(abnormal_index)][y]) for y in selected_columns]))
            abnormal_increasing_values.append(pool.starmap(r_value, [(hazium_data[(abnormal_index) & (general_data[y] > hazium_data[x])][x], general_data[(abnormal_index) & (general_data[y] > hazium_data[x])][y]) for y in selected_columns]))
            abnormal_decreasing_values.append(pool.starmap(r_value, [(hazium_data[(abnormal_index) & (general_data[y] < hazium_data[x])][x], general_data[(abnormal_index) & (general_data[y] < hazium_data[x])][y]) for y in selected_columns]))

    for name, value in [("Normal", normal_values), ("Abnormal", abnormal_values), ("Higher", abnormal_increasing_values), ("Lower", abnormal_decreasing_values)]:
        if verbose:
            print(">>", name)

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.pcolor(value)

        matplotlib.pyplot.title(name + ": " + "%.2f" % numpy.mean(value))
        matplotlib.pyplot.xlabel("General Data")
        matplotlib.pyplot.ylabel("Hazium Data")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.yticks([])
        matplotlib.pyplot.colorbar()

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "Heatmap_" + name + current_time() + ".png")

        matplotlib.pyplot.close()

    max_x, max_y, v = None, None, -2
    for i, x in enumerate(hazium_data.columns):
        for j, y in enumerate(selected_columns):
            if abnormal_increasing_values[i][j] + abnormal_decreasing_values[i][j] > v:
                max_x, max_y, v = x, y, abnormal_increasing_values[i][j] + abnormal_decreasing_values[i][j]

    if verbose:
        print(max_x, "&", max_y, ":", v)

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(general_data[(abnormal_index) & (general_data[max_y] < hazium_data[max_x])][max_y], hazium_data[(abnormal_index) & (general_data[max_y] < hazium_data[max_x])][max_x], label="Lower")
    matplotlib.pyplot.scatter(general_data[(abnormal_index) & (general_data[max_y] > hazium_data[max_x])][max_y], hazium_data[(abnormal_index) & (general_data[max_y] > hazium_data[max_x])][max_x], label="Higher")

    matplotlib.pyplot.title("Scatter Map")
    matplotlib.pyplot.xlabel(max_y)
    matplotlib.pyplot.ylabel("Hazium: " + str(max_x))
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "Scatter" + current_time() + ".png")

    matplotlib.pyplot.close()

    matplotlib.use("Agg")
    matplotlib.rcParams.update({"font.size": 30})

    matplotlib.pyplot.figure()
    matplotlib.pyplot.plot(range(len(general_data[abnormal_index][max_y])), general_data[abnormal_index][max_y], label="Abnormal")
    matplotlib.pyplot.plot(range(len(hazium_data[abnormal_index][max_x])), hazium_data[abnormal_index][max_x], label="Hazium")

    matplotlib.pyplot.title("Plot of General Data & Hazium Data")
    matplotlib.pyplot.xlabel("Time")
    matplotlib.pyplot.ylabel("Value (Standardized)")
    matplotlib.pyplot.xticks([])
    matplotlib.pyplot.grid(True)
    matplotlib.pyplot.legend()

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "Plot" + current_time() + ".png")

    matplotlib.pyplot.close()


def get_prox_data_frequency(verbose=False, is_drawing=False):
    """
    """
    _pickle_file = ".prox_frequency.pkl"

    if os.path.exists(_pickle_file):
        if verbose:
            print("Pickle exists")
        with open(_pickle_file, "rb") as f:
            frequency_data = pickle.load(f)
    else:
        if verbose:
            print("Calculating...")

        prox_data = get_both_prox_data()
        frequency_data = pandas.DataFrame(data=[], columns=["timestamp", "number"])

        pointing, max_time = datetime.datetime(year=2016, month=5, day=31), datetime.datetime(year=2016, month=6, day=14)

        while pointing < max_time:
            frequency_data = frequency_data.append(other={"timestamp": pointing, "number": len(prox_data[(pointing <= prox_data["timestamp"]) & (prox_data["timestamp"] < (pointing + datetime.timedelta(minutes=5)))])}, ignore_index=True)

            pointing += datetime.timedelta(minutes=5)

        with open(_pickle_file, "wb") as f:
            pickle.dump(frequency_data, f)

    if verbose:
        print(frequency_data)
        statistics(list(frequency_data["number"]))

    if is_drawing:
        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        matplotlib.pyplot.bar(range(len(frequency_data["number"])), sorted(frequency_data["number"], reverse=True))

        matplotlib.pyplot.title("Distribution of Frequency for prox Data")
        matplotlib.pyplot.xlabel("Counts")
        matplotlib.pyplot.ylabel("Frequency")
        matplotlib.pyplot.xticks([])
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "FrequencyDistribution" + current_time() + ".png")

        matplotlib.pyplot.close()

    return frequency_data


if __name__ == "__main__":
    # employee_data = get_employee_data(show=True)
    # general_data = get_general_data(show=True)
    # general_zscore_data = get_general_zscore_data(show=True)
    # hazium_data = [get_hazium_data(data, True) for data in get_hazium_data()]
    # fixed_prox_data = get_fixed_prox_data(show=True)
    # mobile_prox_data = get_mobile_prox_data(show=True)

    # draw_mobile_prox_data(verbose=True)
    # tsne_prox_data = get_tsne_prox_data(is_drawing=True, verbose=True)
    # draw_tsne_mobile_prox_data_by_value(verbose=True)
    # tsne_general_data = get_tsne_general_data(is_drawing=True, verbose=True)
    # abnormal_general_data = get_abnormal_general_data(is_drawing=True, verbose=True)
    # draw_general_data(verbose=True, relative=False)
    # draw_hazium_data(verbose=True, which=None)

    # regression_all_general_data(verbose=True, processes=100)

    # draw_movement(verbose=True, different_alpha=True)
    # movement_information = calculate_movement(verbose=True)
    # draw_movement_distribution(verbose=True)
    # [draw_percentile_moving_distribution(verbose=True, minimum=i, maximum=i + 25) for i in range(0, 100, 25)]

    # draw_hazium_data(verbose=True)
    # floor_data = get_floor_data(floor=2, verbose=True)

    # floor_data = [get_floor_data(floor=i, verbose=True) for i in range(4)]

    # general_correlation_data = draw_correlation_with_general_data(verbose=True)
    # draw_stacked_general_data(verbose=True)
    # draw_average_general_data(verbose=True)
    # draw_median_general_data(verbose=True)
    # draw_ultimate_general_data(verbose=True)

    # abnormal_hazium_data = get_abnormal_hazium_data(is_drawing=True, verbose=True)
    # draw_correlation_with_hazium_data(verbose=True, processes=100)
    # compare_abnormality(verbose=True)
    # draw_movement_department(verbose=True)

    # draw_first_quarter_general_data(verbose=True, cycle=288)
    # draw_second_quarter_general_data(verbose=True, processes=100)
    # draw_third_quarter_general_data(verbose=True, cycle=288)
    # draw_fourth_quarter_general_data(verbose=True, processes=100)
    # calculate_abnormality_score(verbose=True)
    # compare_column(verbose=True)
    # compare_general_hazium(verbose=True)
    get_prox_data_frequency(verbose=True, is_drawing=True)
