import matplotlib.pyplot as plt
import os
import math
import numpy as np
from scipy.signal import medfilt, savgol_filter
from hall import analyze_hall, prepare_time
from ppt import analyze_ppt, damped_cos

"""
Work folder defines the folder where this script will look for files with data with a specific structure of data:
Time [ms]	Position [mm]	Voice-coil voltage [V]

PPT_calibrated defines the folder where this script will look for calibration results for PPT engine.
PPT_calibrator_impulse is the impulse bit used to calibrate the calibrator for PPT engine.
Hall_calibrator_mass is the mass of weight used to calibrate after Hall engine fires.

"""

# Turn on or off calculating the mass
mass_debug = False
# Turn on or off plotting
enable_plot = True

work_folder = "/dane eksperymentalne/silnik PPT/20250618"
PPT_calibrated = "/dane eksperymentalne/silnik PPT/kalibrator"
PPT_calibrator_impulse = 8.52e-5  # μNs
Hall_calibrator_mass = 2.6862 / 2  # grams


def data_from_file(filename):
    time = []
    position = []
    voltage = []

    # Read data from file
    with open(filename, "r") as f:
        raw_data = f.read()

    # Split each line into Time, Position and Voltage readings
    for line in raw_data.split("\n"):
        if line != '' and line != "Time [ms]	Position [mm]	Voice-coil voltage [V]":
            line = line.split("\t")
            # Time calculated into seconds, Position in millimeters and Voltage in Volts
            time.append(float(int(line[0]) / 1000))
            position.append(float(line[1]))
            voltage.append(float(line[2]))

    return time, position, voltage


def nice_plot(x_axis, y_axis, x_name, y_name, title, lines=None, over_x=None, over_y=None):
    _, ax = plt.subplots(figsize=(12, 6))
    # Plot given x and y data
    ax.plot(x_axis, y_axis)
    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(title)
    plt.grid(True, linewidth="0.5")
    plt.locator_params(axis='x', nbins=4)

    if lines:
        for line in lines:
            ax.plot([line[0], line[2]], [line[1], line[3]], 'r')

    if over_x:
        for i in range(len(over_x)):
            ax.plot(over_x[i], over_y[i], f"C{i+1}")

    plt.show()


def inverse_datetime(datetime: str):
    # Turn 20250820 into 20.08.2025
    return f"{datetime[6:8]}.{datetime[4:6]}.{datetime[0:4]}"


def get_impulse_mass():
    # Function used to calculate mass from a given impulse bit and given folder of calibration results
    mass_list = []
    for file in os.listdir(PPT_calibrated):
        time, position, _ = data_from_file(os.path.join(PPT_calibrated, file))
        _, _, result = analyze_ppt(position, time)
        mass_part = PPT_calibrator_impulse / (abs(result[0] * result[1]))
        mass_list.append(mass_part)

    mass = sum(mass_list) / len(mass_list)
    return mass


def noise_filter(data):
    return savgol_filter(medfilt(data, kernel_size=5), window_length=11, polyorder=3, mode='interp')


def main():
    errors = []
    for test in os.listdir(work_folder):
        print(f"Opening file {test}...")
        # Data lists collecting all lines that should be displayed on the chart
        data, plot_x, plot_y = [], [], []
        t, pos_old, _ = data_from_file(os.path.join(work_folder, test))

        pos = noise_filter(pos_old)
        # Add filtered data to plot over original data
        plot_x.append(t)
        plot_y.append(pos)

        if len(t) > 3000:
            points = []
            print("I think it's a Hall engine!")
            date = inverse_datetime(work_folder.split("/")[-1])
            res = analyze_hall(t, pos)

            # Add fitted mean line to first impulse (engine on) to plot over original data
            plot_x.append(t[res[0][0]:res[0][1]])
            plot_y.append(np.array(t[res[0][0]:res[0][1]]) * res[0][2][4] + res[0][2][5])

            points.append([t[res[0][1]], t[res[0][1]] * res[0][2][4] + res[0][2][5]])

            for i in range(1, len(res)):
                t_adj = []
                for j in range(res[i][0], res[i][1]):
                    t_adj.append(np.array(t[j] - t[res[i][0]]))
                # Add fitted drift to every impulse to plot over original data
                plot_x.append(t[res[i][0]:res[i][1]])
                plot_y.append(np.array(t_adj) * res[i][2][4] + res[i][2][5])
                points.append([t[res[i][0]], t_adj[0] * res[i][2][4] + res[i][2][5]])
                try:
                    points.append([t[res[i][1]+9], t_adj[-1] * res[i][2][4] + res[i][2][5]])
                except IndexError:
                    pass

            if len(res) > 3:
                print("Checking accuracy of code...")
                for i in range(0, len(points), 4):
                    dt = abs(math.dist((points[i][0], points[i][1]), (points[i+1][0], points[i+1][1])))
                    dc = abs(math.dist((points[i+2][0], points[i+2][1]), (points[i+3][0], points[i+3][1])))

                    calibr_force = Hall_calibrator_mass * 9.8123
                    thrust_force = "%.6f" % (calibr_force * dt / dc)
                    error = "%.2f" % ((abs(float(thrust_force) - calibr_force)/calibr_force) * 100)
                    print(f"Error: {error}%")

                    errors.append(((abs(float(thrust_force) - calibr_force)/calibr_force) * 100))
            else:
                # Calculates differences between: engine on/off and engine off/calibrator on using fitted first
                # degree polynomial and the distance between two points: last point of the previous polynomial and
                # first point of the next polynomial. Better accuracy than distance between straight lines
                dt = abs(math.dist((points[0][0], points[0][1]), (points[1][0], points[1][1])))
                dc = abs(math.dist((points[2][0], points[2][1]), (points[3][0], points[3][1])))

                calibr_force = Hall_calibrator_mass * 9.8123
                thrust_force = "%.6f" % (calibr_force * dt / dc)
                print(f"Thrust force: {thrust_force} mN")

            print("Data analyzed. Displaying...")

            # Plot all collected data on the chart: x values are Time, y values are Displacement, with correct
            # labels and title, with lines given in 'data' list as mentioned previously

            nice_plot(x_axis=t, y_axis=pos, x_name="time [s]", y_name="displacement [mm]",
                      title=str(date + " - " + test),
                      lines=data, over_x=plot_x, over_y=plot_y)

        else:
            print("I think it's a PPT engine!")
            if str(work_folder.split("/")[-1]) == "kalibrator":
                date = "kalibrator"
            else:
                date = inverse_datetime(work_folder.split("/")[-1])

            try:
                start, end, popt = analyze_ppt(pos, t)

                if start:
                    # Plot vertical lines depicting the start and end of an impulse
                    data.append([t[start], np.min(pos), t[start], np.max(pos)])
                    data.append([t[end], np.min(pos), t[end], np.max(pos)])

                    if not mass_debug:
                        impulse_bit = "%.6f" % abs(popt[0] * popt[1] * get_impulse_mass() * 1e6)

                        if str(work_folder.split("/")[-1]) == "kalibrator":
                            error = "%.2f" % ((abs((float(impulse_bit)/1e6) -
                                                   PPT_calibrator_impulse) / PPT_calibrator_impulse) * 100)
                            print(f"Error: {error}%")

                            errors.append(((abs((float(impulse_bit)/1e6) -
                                                PPT_calibrator_impulse) / PPT_calibrator_impulse) * 100))
                        else:
                            print("Data analyzed. Displaying...")
                            print(f"Impulse bit: {impulse_bit} μNs")

                    # Add fitted damped cosine wave to plot over original data
                    plot_x.append(t[start:end])
                    plot_y.append(damped_cos(prepare_time(t[start:end]), *popt))
            except TypeError:
                print("start error")

            if enable_plot:
                nice_plot(x_axis=t, y_axis=pos_old, x_name="time [s]", y_name="displacement [mm]",
                          title=str(date + " - " + test),
                          lines=data, over_x=plot_x, over_y=plot_y)

    if errors:
        print("\n-------------\nERROR RESULTS\n-------------\n")
        print(f"Max error: {'%.2f' % max(np.array(errors))}%")
        print(f"Min error: {'%.2f' % min(np.array(errors))}%")
        print(f"Average error: {'%.2f' % (sum(np.array(errors))/len(errors))}%")


if __name__ == "__main__":
    main()
