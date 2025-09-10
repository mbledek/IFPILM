import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from scipy.signal import savgol_filter, correlate, correlation_lags
import numpy as np
from scipy.interpolate import interp1d
import os
import pandas as pd

# Data about the number of cars per hour starting from 0, 1, 2 AM
weekend = [1896, 1654, 981, 697, 674, 983, 1435, 1985, 2891, 4073, 5268, 6079,
           6539, 6580, 6469, 6312, 6186, 6207, 6269, 6064, 5403, 4201, 3003, 2038]
week = [934, 529, 371, 381, 624, 1992, 4785, 7075, 7253, 6521, 6710, 6983,
        7200, 7505, 8137, 9161, 9582, 9284, 8651, 7507, 6297, 4732, 3077, 1710]


def data_from_file(filename, step=1):
    time = []
    position = []
    voltage = []

    # Read data from file
    with open(filename, "r") as f:
        raw_data = f.read()

    # Split each line into Time, Position and Voltage readings
    for line in raw_data.split("\n")[1::step]:
        if line != '' and (line != "Time [ms]	Position [mm]	Voice-coil voltage [V]" or
                           line != '"","Channel 1 Last (C)","Channel 2 Last (C)"'):
            if "\t" in line:
                line = line.split("\t")
                # Time calculated into seconds, Position in millimeters and Voltage in Volts
                time.append(float(line[0]))
                position.append(float(line[1]))
                voltage.append(float(line[2]))
            else:
                line = line.replace('"', '').split(',')
                if not line[1] == '':
                    time.append(int(line[0])*1000)
                    position.append(float(line[1]))
                    voltage.append(float(line[2]))

    return np.array(time)/1000.0, np.array(position), np.array(voltage)


def nice_plot(x_axis, y_axis, x_name: str, y_name: str, title: str, over_x=None, over_y=None,
              subtitle="", dates='', added=False):
    if over_x is None:
        over_x = []
    if over_y is None:
        over_y = []
    _, ax = plt.subplots(figsize=(16, 10))
    # Plot given x and y data
    print("Starting plot")
    ax.plot(x_axis[::100], y_axis[::100])

    start_day = pd.to_datetime(min(x_axis[::100])).normalize()
    end_day = pd.to_datetime(max(x_axis[::100])).normalize()

    # Mark a difference between every other day by making the background of the plot gray or white
    for i, day in enumerate(pd.date_range(start_day, end_day, freq="D")):
        if i % 2 == 0:  # every other day
            ax.axvspan(day, day + pd.Timedelta(days=1), facecolor="lightgray", alpha=1)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=480))
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    plt.grid(True, linewidth="0.5")
    if len(over_x) > 0:
        ax2 = ax.twinx()
        ax2.set_ylabel(subtitle)
        ax2.plot(over_x, over_y, "C1")
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_major_locator(mdates.MinuteLocator(interval=480))

    # Save plots under correct names
    if added:
        if "Derivative" in title:
            plt.savefig(f"plots/{dates} derivative added {subtitle}.png", bbox_inches='tight')
        elif "noise" in title.lower():
            plt.savefig(f"plots/{dates} Noise {subtitle}.png", bbox_inches='tight')
        else:
            plt.savefig(f"plots/{dates} added {subtitle}.png", bbox_inches='tight')
    else:
        if "Derivative" in title:
            plt.savefig(f"plots/{dates} derivative  {subtitle}.png", bbox_inches='tight')
        elif "noise" in title.lower():
            plt.savefig(f"plots/{dates} Noise {subtitle}.png", bbox_inches='tight')
        else:
            plt.savefig(f"plots/{dates} {subtitle}.png", bbox_inches='tight')

    plt.close()
    # plt.show()


def inverse_datetime(datetime: str):
    # Turn 20250820 into 20.08.2025
    return f"{datetime[6:8]}.{datetime[4:6]}.{datetime[0:4]}"


def noise_filter(data):
    # Use Savitzky-Golay filter
    return savgol_filter(data, window_length=4001, polyorder=3, mode='interp')


def derivative_chunked(x, y, window=101):
    y = noise_filter(y)
    k = window // 2
    dy_dx = np.empty_like(y)

    # Interior: wide central difference
    dy_dx[k:-k] = (y[2*k:] - y[:-2*k]) / (x[2*k:] - x[:-2*k])

    # Edges: fall back to smaller differences
    dy_dx[:k] = (y[1:k+1] - y[0]) / (x[1:k+1] - x[0])   # forward
    dy_dx[-k:] = (y[-1] - y[-k-1:-1]) / (x[-1] - x[-k-1:-1])  # backward

    return noise_filter(dy_dx)


def merge_data(data1, data2, time=False, added=False):
    output = list(data1)
    if len(data1) > 0:
        if time:
            output.extend(list(np.array(data2) + data1[-1]))
        else:
            if added:
                output.extend(list(np.array(data2) + data1[-1]))
            else:
                output.extend(list(np.array(data2)))
    else:
        output.extend(list(np.array(data2)))
    return np.array(output)


def crop_lags(lags, values, limit):
    mask = (lags >= -limit) & (lags <= limit)
    return lags[mask], values[mask]


def xcorr_coeff(x, y_data):
    # Interpolate shorter dataset to match the length of first dataset
    interp_func = interp1d(np.arange(len(y_data)), y_data, kind='linear')
    y = interp_func(np.linspace(0, len(y_data) - 1, len(x)))
    x = (x - np.mean(x))
    y = (y - np.mean(y))
    # Calculate the correlation
    c = correlate(x, y, mode='full')
    lags = correlation_lags(len(x), len(y), mode='full')
    overlap = (len(x) - np.abs(lags))
    denom = overlap * np.std(x) * np.std(y)
    output = np.where(np.abs(c / denom) < 1.0, c / denom, 0.0)

    lags, output = crop_lags(lags, output, 23040)

    # Find min and max values of correlation
    max_idx = np.argmax(output)
    min_idx = np.argmin(output)
    max_corr, max_lag = output[max_idx], lags[max_idx]
    min_corr, min_lag = output[min_idx], lags[min_idx]

    return output, lags, (max_corr, max_lag), (min_corr, min_lag)


def weather(filename, index):
    x_data, y_data = [], []
    with open(filename, "r", encoding="utf-8") as f:
        csv_data = f.read().split('\n')

    for item in csv_data[1:-1]:
        x_data.append(dt.datetime(int(item[0:4]), int(item[5:7]), int(item[8:10]), int(item[11:13]), int(item[14:16])))
        y_data.append(float(item.split(';')[index].replace(" m/s", "")))

    return x_data, y_data


def single(folders):
    for foldername in folders:
        t, pos, _ = data_from_file(f"/dane dobowe/{foldername}/TB/00")

        # Calculate what time was the first line of data collected
        dt2 = dt.datetime.fromtimestamp(os.path.getmtime(f"/dane dobowe/{foldername}/TB/00"))
        dt2 = dt2 - dt.timedelta(milliseconds=(float(t[-1]) + float(t[0])))

        start_time = dt.datetime(int(foldername[0:4]), int(foldername[4:6]), int(foldername[6:8]),
                                 dt2.hour, dt2.minute, dt2.second)

        t2, _, temp2 = data_from_file(f"/dane dobowe/{foldername}/picolog.csv")
        # Fix PT100 data to a correct timezone
        fixed = []
        for i in range(len(t2)):
            fixed.append(dt.datetime.utcfromtimestamp(t2[i]) + dt.timedelta(hours=2))

        nice_plot([start_time + dt.timedelta(milliseconds=ms) for ms in t*1000.0], pos,
                  "Time of day [hours]", "Displacement [mm]", inverse_datetime(foldername),
                  fixed, temp2, "Temp outside chamber")


def multiple(added, derivative):

    # WEEK 1
    # dates = "25-02"
    # index = [2, 3, 4, 5, 6, 8]
    # title = ["Temperatura odczuwalna [C]", "Temperatura powietrza [C]", "Temperatura nawierzchni 0cm [C]",
    #          "Temperatura nawierzchni -5cm [C]", "Temperatura nawierzchni -30cm [C]", "Prędkość wiatru [m s]"]
    # folders = ["20250825", "20250826", "20250827", "20250828", "20250829"]

    # WEEK 2
    dates = "01-08"
    index = [2, 3, 4, 5, 6, 8, 14, 15]
    title = ["Temperatura odczuwalna [C]", "Temperatura powietrza [C]", "Temperatura nawierzchni 0cm [C]",
             "Temperatura nawierzchni -5cm [C]", "Temperatura nawierzchni -30cm [C]", "Prędkość wiatru [m s]",
             "Windy prędkość wiatru [m s]", "Windy powiewy wiatru [m s]"]
    folders = ["20250901", "20250902", "20250903", "20250904", "20250905"]

    # WEEK 3
    # dates = "08-09"
    # folders = ["20250908"]
    merged_time, merged_pos, merged_temp, merged_t2 = [], [], [], []
    final_time, final_pos, final_temp, final_t2 = [], [], [], []
    starts, ends = [], []

    print("Starting getting data...")
    for foldername in folders:
        t, pos, _ = data_from_file(f"/dane dobowe/{foldername}/TB/00")
        merged_time.append(t)
        merged_pos.append(pos)
        dt2 = dt.datetime.fromtimestamp(os.path.getmtime(f"/dane dobowe/{foldername}/TB/00"))
        ends.append(dt2)
        # Calculate what time was the first line of data collected
        dt2 = dt2 - dt.timedelta(milliseconds=(float(t[-1]) - float(t[0])))
        starts.append(dt.datetime(int(foldername[0:4]), int(foldername[4:6]), int(foldername[6:8]),
                                  dt2.hour, dt2.minute, dt2.second))

        t2, _, temp2 = data_from_file(f"/dane dobowe/{foldername}/picolog.csv")
        fixed = []
        # Fix PT100 data to a correct timezone
        for i in range(len(t2)):
            fixed.append(dt.datetime.utcfromtimestamp(t2[i]) + dt.timedelta(hours=2))
        merged_t2.append(fixed)
        merged_temp.append(temp2)

    # Properly merge data from all files
    while len(merged_time) > 0:
        final_time = merge_data(final_time, merged_time[0], time=True)
        merged_time.pop(0)
        final_pos = merge_data(final_pos, merged_pos[0], time=False, added=added)
        merged_pos.pop(0)
        final_temp = merge_data(final_temp, merged_temp[0], time=False, added=False)
        merged_temp.pop(0)
        final_t2 = merge_data(final_t2, merged_t2[0], time=False, added=False)
        merged_t2.pop(0)

    start_time = dt.datetime(int(folders[0][0:4]), int(folders[0][4:6]), int(folders[0][6:8]), 11, 45)
    datetimes = [dt.datetime(int(folders[0][0:4]), int(folders[0][4:6]), int(folders[0][6:8]))
                 + dt.timedelta(hours=i) for i in range(168)]

    if derivative:
        print("Getting derivative...")
        # Calculate the derivative
        dy_dx = derivative_chunked(final_time[::100], final_pos[::100], window=1000)

    # Converts time to datetime format
    final_time = np.array([start_time + dt.timedelta(milliseconds=ms) for ms in np.array(final_time) * 1000])

    print("Data got")

    # Calculate noise
    noise_first = final_pos - noise_filter(final_pos)
    noise_only = noise_first - noise_filter(noise_first)
    noise_only = np.where(abs(noise_only) < 0.01, noise_only, float('nan'))
    nice_plot(final_time, noise_only, "Time of day [hours]", "Noise", "Noise", dates=dates)

    # Calculate rolling standard deviation of noise
    df = pd.DataFrame({'signal': final_pos, 'noise': noise_only})
    df['noise_std'] = df['noise'].rolling(window=40000).std()

    new_noise = np.where(df['noise_std'] < 0.0012, df['noise_std'], float('nan'))

    print("Noise calculated")

    nice_plot(final_time, new_noise, "Time of day [hours]", "Rolling standard deviation",
              "Rolling standard deviation of noise",
              datetimes, week*5+weekend*2, "Number of cars per hour", dates)

    # Skip to every 100th point to speed up code (the accuracy was only needed for noise calculation)
    final_time = final_time[::100]
    final_pos = final_pos[::100]

    print("Getting PT100...")
    if not derivative:
        # Calculate correlation between original data and PT100 data
        c, lags, (max_corr, max_lag), (min_corr, min_lag) = xcorr_coeff(final_pos, final_temp)
        print(f"\n-------------\nCORRELATION BETWEEN Displacement and PT100\n-------------")
        print(f"Max: {'%.2f' % max_corr} for lag: {max_lag}, time: {'%.2f' % (max_lag * 2.5 / 60 / 60)} hours")
        print(f"Min: {'%.2f' % min_corr} for lag: {min_lag}, time: {'%.2f' % (min_lag * 2.5 / 60 / 60)} hours")

        nice_plot(final_time, final_pos,
                  "Time of day [hours]", "Displacement [mm]", f"Merged {dates}",
                  final_t2, final_temp, "PT100 temp [C]", dates, added)

    else:
        # Calculate correlation between derivative and PT100 data
        c, lags, (max_corr, max_lag), (min_corr, min_lag) = xcorr_coeff(dy_dx, final_temp)
        print(f"\n-------------\nCORRELATION BETWEEN Derivative and PT100\n-------------")
        print(f"Max: {'%.2f' % max_corr} for lag: {max_lag}, time: {'%.2f' % (max_lag * 2.5 / 60 / 60)} hours")
        print(f"Min: {'%.2f' % min_corr} for lag: {min_lag}, time: {'%.2f' % (min_lag * 2.5 / 60 / 60)} hours")

        nice_plot(final_time, dy_dx,
                  "Time of day [hours]", "Derivative", f"Derivative of merged {dates}",
                  final_t2, final_temp, "PT100 temp [C]", dates, added)

    print("\nGetting weather...")
    for i in range(len(index)):
        # Get weather data
        weather_x, weather_y = weather(f"Pogoda {dates}.csv", index[i])

        # Trim weather data to start when our data was collected
        weather_x = [d for d in weather_x if d > start_time]
        weather_y = weather_y[-len(weather_x):]

        if not derivative:
            # Calculate correlation between original data and weather
            c, lags, (max_corr, max_lag), (min_corr, min_lag) = xcorr_coeff(final_pos, weather_y)
            print(f"\n-------------\nCORRELATION BETWEEN Displacement and {title[i]}\n-------------")
            print(f"Max: {'%.2f' % max_corr} for lag: {max_lag}, time: {'%.2f' % (max_lag * 2.5 / 60 / 60)} hours")
            print(f"Min: {'%.2f' % min_corr} for lag: {min_lag}, time: {'%.2f' % (min_lag * 2.5 / 60 / 60)} hours")

            nice_plot(final_time, final_pos,
                      "Time of day [hours]", "Displacement [mm]", f"Merged {dates}",
                      weather_x, weather_y, title[i], dates, added)

        else:
            # Calculate correlation between derivative and weather
            c, lags, (max_corr, max_lag), (min_corr, min_lag) = xcorr_coeff(dy_dx, weather_y)
            print(f"\n-------------\nCORRELATION BETWEEN Derivative and {title[i]}\n-------------")
            print(f"Max: {'%.2f' % max_corr} for lag: {max_lag}, time: {'%.2f' % (max_lag * 2.5 / 60 / 60)} hours")
            print(f"Min: {'%.2f' % min_corr} for lag: {min_lag}, time: {'%.2f' % (min_lag * 2.5 / 60 / 60)} hours")

            nice_plot(final_time, dy_dx,
                      "Time of day [hours]", "Derivative", f"Derivative of merged {dates}",
                      weather_x, weather_y, title[i], dates, added)

        # Show correlation plot if needed
        # plt.plot(lags, c)
        # plt.xlabel("Lag")
        # plt.ylabel("Correlation")
        # plt.title(f"Cross-correlation: final_pos and {title[i]}")
        # plt.axhline(0, color='black')
        # plt.show()


if __name__ == "__main__":
    # Use multiple() when analyzing multiple days of data
    # MAKE SURE ALL INITIAL DATA IS SET PROPERLY INSIDE FUNCTION (dates, folders, etc.)
    multiple(added=False, derivative=False)

    # Use single() when analyzing single day of data
    # single(["20250903"])
