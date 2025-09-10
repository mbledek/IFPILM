from hall import *


def damped_cos(t, amp, omega, gamma, phi, b):
    return amp * np.exp(-gamma * np.array(t)) * np.cos(omega * np.array(t) + phi) - b


def round_to_closest(time_list, time_picked):
    return min(time_list, key=lambda x: abs(x - time_picked))


def analyze_ppt(disp, time):
    time = np.array(time)
    length = len(disp)
    start_index, start_value, end_index, index = 0, 0, 0, 0

    for ij in range(75, length-4):
        if (disp[ij-2]-disp[ij+2]) > 0.002:
            start_index = ij + 1
            break

    if start_index == 0:
        for ij in range(185, length-4):
            if (disp[ij-2]-disp[ij+2]) > 0.0005:
                start_index = ij + 1
                break

        if start_index == 0:
            print("Start error")
            return

    t_adj = prepare_time(time[start_index:])

    # 2) frequency estimate
    omega0, f_peak, _ = estimate_frequency_fft(t_adj, disp[start_index:])

    end_index = list(time).index(round_to_closest(time, (time[start_index] + (2/f_peak))))

    t_adj = prepare_time(time[start_index:end_index])

    # 3) gamma estimate via envelope
    gamma0 = estimate_gamma_envelope(t_adj, disp[start_index:end_index])

    # 4) amplitude & phase estimate
    a0, phi0 = estimate_a_phi(t_adj, disp[start_index:end_index], omega0, gamma0)

    p0 = [a0, omega0, gamma0, phi0, 0.0001]

    # Fit a cosine curve with offset to first two waves to get an amplitude and a frequency
    result, _ = curve_fit(damped_cos, t_adj, disp[start_index:end_index], p0=p0)

    return start_index, end_index, result
