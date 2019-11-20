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
import pandas
import pandas.plotting
import scipy
import sklearn
import sklearn.manifold

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


def get_regression_general_data(column, is_drawing=True, verbose=False):
    """
    Regreesion general data. (Actual)

    Get regression with general data by each columns. Save the best regression for further analysis. Returns score to find unusual events. Last modified: 2019-11-19T08:52:35+0900

    Args:
        column (string): Mandatory. column name to regress.
        is_drawing (bool): If this is True, draw the regression plot.
        verbose (bool): Verbosity level

    Returns:
        List: The score of best the algorithm amongst the algorithms which are executed.
    """
    _data = get_general_data()
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
        _data.drop(columns=["Date/Time"], inplace=True)
        _values["Real"] = y_data

        svr = sklearn.svm.SVR(gamma="scale")
        svr.fit(_data, y_data)
        _values["svr (%.2f)" % svr.score(_data, y_data)] = svr.predict(_data)

        _score, _value = svr.score(_data, y_data), svr.predict(_data)

        if verbose:
            print("SVR:", _score)

        nusvr = sklearn.svm.NuSVR(gamma="scale")
        nusvr.fit(_data, y_data)
        _values["nusvr (%.2f)" % nusvr.score(_data, y_data)] = nusvr.predict(_data)

        if verbose:
            print("NuSVR:", nusvr.score(_data, y_data))

        if _score < nusvr.score(_data, y_data):
            _score, _value = nusvr.score(_data, y_data), nusvr.predict(_data)

        linearsvr = sklearn.svm.LinearSVR(random_state=0, max_iter=10000)
        linearsvr.fit(_data, y_data)
        _values["linearsvr (%.2f)" % linearsvr.score(_data, y_data)] = linearsvr.predict(_data)

        if verbose:
            print("LinearSVR:", linearsvr.score(_data, y_data))

        if _score < linearsvr.score(_data, y_data):
            _score, _value = linearsvr.score(_data, y_data), linearsvr.predict(_data)

        with open(_pickle_file, "wb") as f:
            pickle.dump((_value, _values), f)

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
    _data = get_general_data()

    with multiprocessing.Pool(processes=processes) as pool:
        values = pool.map(get_regression_general_data, list(_data.columns)[1:])

    print(values)


def draw_movement_mobile_prox_data(verbose=False):
    """
    Draw movement with mobile prox data.

    Draw movement with mobile prox data by each individuals. Start point will be shown as red mark, and final point will be shown as blue mark. Last modified: 2019-11-20T19:17:16+0900
    """
    if verbose:
        print("Drawing movement")

    _data = get_mobile_prox_data()
    _names = sorted(list(set(list(map(lambda x: x[:-3], _data["prox-id"])))))

    for num, name in enumerate(_names):
        if verbose:
            print(">> Drawing:", name)

        drawing_data = _data[(_data["prox-id"].str.contains(name))]
        x_data = list(drawing_data["x"])
        y_data = list(drawing_data["y"])

        matplotlib.use("Agg")
        matplotlib.rcParams.update({"font.size": 30})

        matplotlib.pyplot.figure()
        for i in range(1, len(x_data)):
            if x_data[i - 1] == x_data[i] and y_data[i - 1] == y_data[i]:
                continue
            matplotlib.pyplot.arrow(x_data[i - 1], y_data[i - 1], x_data[i] - x_data[i - 1], y_data[i] - y_data[i - 1], alpha=0.7, length_includes_head=True, head_width=3, head_length=3, color="k")
        matplotlib.pyplot.scatter(x_data[0], y_data[0], s=1000, marker="X", c="r")
        matplotlib.pyplot.scatter(x_data[-1], y_data[-1], s=1000, marker="X", c="b")

        matplotlib.pyplot.title("Movement of " + name)
        matplotlib.pyplot.xlabel("X")
        matplotlib.pyplot.ylabel("Y")
        matplotlib.pyplot.xlim(0, _x_limit)
        matplotlib.pyplot.ylim(0, _y_limit)
        matplotlib.pyplot.grid(True)

        fig = matplotlib.pyplot.gcf()
        fig.set_size_inches(32, 18)
        fig.savefig(figure_directory + "Movement_" + str(num) + current_time() + ".png")

        matplotlib.pyplot.close()

    if verbose:
        print("Drawing Done!!")


def calculate_movement(verbose=False):
    """
    Calculate movement.

    Calucate movement for each individual by ID. Save the result for further analysis.

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

        _data = get_mobile_prox_data()
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


def draw_movement(verbose=False):
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
    matplotlib.pyplot.xticks()
    matplotlib.pyplot.grid(True)

    fig = matplotlib.pyplot.gcf()
    fig.set_size_inches(32, 18)
    fig.savefig(figure_directory + "MovementDistribution" + current_time() + ".png")

    matplotlib.pyplot.close()


if __name__ == "__main__":
    employee_data = get_employee_data(show=True)
    general_data = get_general_data(show=True)
    hazium_data = [get_hazium_data(data, True) for data in get_hazium_data()]
    fixed_prox_data = get_fixed_prox_data(show=True)
    mobile_prox_data = get_mobile_prox_data(show=True)

    # draw_mobile_prox_data(verbose=True)
    # tsne_mobile_prox_data = get_tsne_mobile_prox_data(is_drawing=True, verbose=True)
    # draw_tsne_mobile_prox_data_by_value(verbose=True)
    # tsne_general_data = get_tsne_general_data(is_drawing=True, verbose=True)
    # draw_general_data(verbose=True, relative=False)
    # draw_movement_mobile_prox_data(verbose=True)

    # regression_all_general_data(verbose=True)

    movement_information = calculate_movement(verbose=True)
    draw_movement()
