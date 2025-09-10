import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import welch


def damped_cos_full(t, amp, omega, gamma, phi, a, b):
    return amp * np.exp(-gamma * t) * np.cos(omega * t + phi) + a * t + b


def damped_cos_only_with_trend(t, amp, omega, gamma, phi, slope, intercept):
    return amp * np.exp(-gamma * t) * np.cos(omega * t + phi) + (slope * t + intercept)


def straight(t, a, b):
    return a * t + b


def prepare_data(disp):
    """
    Analyze_hall analyzes given displacement data and looks for jumps in value suggesting switching on or off of
    the engine or turning on the calibrator. Data returned is in a format of
    [[index_of_jump, mean_value_before_jump], [index_of_jump, ...]]
    If data ends, then it returns the index of last datapoint and mean value before ending of data.
    """

    length = len(disp)
    all_values = []

    starter = 0
    while length - starter > 0:
        for j in range(starter + 150, length):
            mean = float((sum(disp[starter:j]) / (len(disp[starter:j]))))
            if mean == 0.0:
                mean = 0.000001

            # If statement detects jumps if the value/mean is larger than 5 or smaller than 0.5 AND value - mean is
            # larger than 0.05. The values presented are hand-picked, so that electronic noise doesn't get detected
            # as a jump in value.
            if (5 < abs(disp[j] / mean) or abs(disp[j] / mean) < 0.5) and abs(disp[j] - mean) > 0.05:
                all_values.append([j, mean])
                starter = j
                break
            if length - j < 2:
                all_values.append([j, mean])
                return all_values


def prepare_time(t):
    return np.array(t)-t[0]


def estimate_trend(t, y):
    a = np.vstack([t, np.ones_like(t)]).T
    slope, intercept = np.linalg.lstsq(a, y, rcond=None)[0]
    return slope, intercept


def estimate_frequency_fft(t, y_detr):
    fs = 1.0 / np.mean(np.diff(t))
    f, pxx = welch(y_detr, fs=fs, nperseg=min(1024, len(y_detr)))
    # ignore very low freqs (trend)
    f_min = 0.45
    idx0 = np.where(f >= f_min)[0]
    if idx0.size == 0:
        idx_peak = np.argmax(pxx)
    else:
        idx_peak = idx0[0] + np.argmax(pxx[idx0])
    f_peak = f[idx_peak]

    if f_peak > 2.0:
        f_peak = 1.1
    omega0 = 2.0 * np.pi * f_peak

    return omega0, f_peak, fs


def estimate_gamma_envelope(t, y_detr):
    absenv = np.abs(y_detr)
    # smooth envelope a bit with moving average
    window_len = max(3, int(0.02 / np.mean(np.diff(t))))  # ~20ms window approximate
    if window_len % 2 == 0:
        window_len += 1
    kernel = np.ones(window_len) / window_len
    absenv_sm = np.convolve(absenv, kernel, mode='same')
    # pick maxima by simple local comparison
    peaks_idx = (np.r_[True, absenv_sm[1:] > absenv_sm[:-1]] & np.r_[absenv_sm[:-1] > absenv_sm[1:], True])
    peaks_idx = np.where(peaks_idx & (absenv_sm > 0))[0]
    # guard: if no reliable peaks, fallback to small gamma initial
    if peaks_idx.size >= 6:
        peak_t = t[peaks_idx]
        peak_amp = absenv_sm[peaks_idx]
        # use only top peaks by amplitude to avoid noisy tiny peaks
        topk = max(6, int(len(peak_amp) * 0.3))
        order = np.argsort(peak_amp)[-topk:]
        x = peak_t[order]
        ylog = np.log(peak_amp[order] + 1e-16)
        # robust linear fit: use least squares (could use ransac/huber if needed)
        a_lin = np.vstack([x, np.ones_like(x)]).T
        s, itc = np.linalg.lstsq(a_lin, ylog, rcond=None)[0]
        gamma0 = max(1e-6, -s)
    else:
        gamma0 = 0.05  # fallback guess
    return gamma0


def estimate_a_phi(t, y_detr, omega0, gamma0):
    # remove decay: multiply by exp(+gamma*t)
    y_undamped = y_detr * np.exp(gamma0 * t)
    c = np.vstack([np.cos(omega0 * t), np.sin(omega0 * t)]).T
    coef, *_ = np.linalg.lstsq(c, y_undamped, rcond=None)
    ccoef, dcoef = coef
    a0 = np.hypot(ccoef, dcoef)
    # choose phi so that A*cos(omega t + phi) matches y
    phi0 = np.arctan2(-dcoef, ccoef)
    return a0, phi0


def analyze_hall(time, disp):
    output = []
    points = prepare_data(disp)

    popt, _ = curve_fit(straight, time[0:points[0][0]], disp[0:points[0][0]])

    a, b = popt[0], popt[1]

    output.append([0, points[0][0]+2, [0, 0, 0, 0, a, b]])

    for i in range(1, len(points)):
        start = points[i-1][0]+2
        end = points[i][0]-7

        t = prepare_time(time[start:end]).copy()
        y = disp[start:end].copy()

        # 1) trend estimate
        slope, intercept = estimate_trend(t, y)
        y_detr = y - (slope * t + intercept)

        # 2) frequency estimate
        omega0, f_peak, _ = estimate_frequency_fft(t, y_detr)

        # 3) gamma estimate via envelope
        gamma0 = estimate_gamma_envelope(t, y_detr)

        # 4) amplitude & phase estimate
        a0, phi0 = estimate_a_phi(t, y_detr, omega0, gamma0)

        # prepare automatic p0 (for oscillation)
        p0 = [a0, omega0, gamma0, phi0, slope, intercept]

        # set tighter bounds around estimates
        omega_low = 0.5 * omega0
        omega_high = 1.5 * omega0
        gamma_low = 0.0
        gamma_high = max(10.0 * gamma0, 50.0)  # cap at 50s^-1 by default
        a_low = 0.0
        a_high = max(1.5 * (max(y) - min(y)), a0)

        # bounds for two-stage fit (we will fix a,b)
        bounds_osc = ([a_low, omega_low, gamma_low, -2 * np.pi],
                      [a_high, omega_high, gamma_high, 2 * np.pi])

        def model_fixed_trend(t_, amp, omega, gamma, phi):
            return damped_cos_only_with_trend(t_, amp, omega, gamma, phi, slope, intercept)

        # try:
        popt_osc, pcov_osc = curve_fit(model_fixed_trend, t, y, p0=p0[:4], bounds=bounds_osc, maxfev=80000)

        if popt_osc is not None:
            p0_full = [float(popt_osc[0]), float(popt_osc[1]), float(popt_osc[2]),
                       float(popt_osc[3]), float(slope), float(intercept)]
        else:
            p0_full = p0

        lower_full = [a_low, 0.0, 0.0, -2*np.pi, -np.inf, -np.inf]
        upper_full = [a_high, omega_high, gamma_high,  2*np.pi, np.inf, np.inf]
        try:
            popt_full, pcov_full = curve_fit(damped_cos_full, t, y, p0=p0_full,
                                             bounds=(lower_full, upper_full), maxfev=80000)
        except Exception:
            popt_full, pcov_full, cond_full = None, None, np.inf

        # -------------------------
        # Plot results if requested
        # -------------------------
        # import matplotlib.pyplot as plt
        # plt.figure(figsize=(12,6))
        # plt.plot(t, y, label='data', color='C0', linewidth=1)
        # # if popt_osc is not None:
        # #     plt.plot(t, model_fixed_trend(t, *popt_osc), label='fit (trend fixed)', color='C2', linewidth=2)
        # if popt_full is not None:
        #     print("t used in part:")
        #     print(t)
        #     plt.plot(t, damped_cos_full(t, *popt_full), label='fit (full)', color='C1', linewidth=1.5)
        # plt.plot(t, slope*t + intercept, '--', label='trend', color='red')
        # plt.xlabel('time (scaled)')
        # plt.ylabel('position')
        # plt.legend()
        # plt.grid(True)
        # plt.title(f'Rows {start}..{end}')
        # plt.show()
        #
        # plt.plot(t, y-damped_cos_full(t, *popt_full))
        # plt.grid(True)
        # plt.show()

        output.append([start, end, popt_full])

    return output
