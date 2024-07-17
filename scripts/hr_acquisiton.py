import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import cheby2, filtfilt, find_peaks, welch, butter, cheb2ord, buttord, freqz
from scipy.optimize import curve_fit
from scipy.stats import mode

def read_csv_file(select_data=True, filepath=None):
    if select_data:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not filepath:
            print("No file selected")
            return None
    return pd.read_csv(filepath)

def plot_raw_data(time_support, raw_data):
    fig, axs = plt.subplots(4, 1, sharex=True)
    axs[0].plot(time_support, raw_data[:, 1])
    axs[0].set_title("RAW acceleration - x")
    axs[0].set_ylabel("Acceleration [m/s^2]")

    axs[1].plot(time_support, raw_data[:, 2])
    axs[1].set_title("RAW acceleration - y")
    axs[1].set_ylabel("Acceleration [m/s^2]")

    axs[2].plot(time_support, raw_data[:, 3])
    axs[2].set_title("RAW acceleration - z")
    axs[2].set_ylabel("Acceleration [m/s^2]")

    axs[3].plot(time_support, raw_data[:, 4])
    axs[3].set_title("RAW acceleration - total")
    axs[3].set_xlabel("Seconds [s]")
    axs[3].set_ylabel("Acceleration [m/s^2]")

    fig.suptitle("RAW data")
    plt.show()

def apply_filter(data, filter_type, cutoff_freq, perc, rp, attenuation, fs, btype='low'):
    fNy = fs / 2
    if filter_type == "cheby2":
        n, Wn = cheb2ord(cutoff_freq / fNy, (perc * cutoff_freq) / fNy, rp, attenuation)
        b, a = cheby2(n, attenuation, Wn, btype=btype)
    else:
        n, Wn = buttord(cutoff_freq / fNy, (perc * cutoff_freq) / fNy, rp, attenuation)
        b, a = butter(n, Wn, btype=btype)
    return filtfilt(b, a, data)

def main():
    select_data = True
    starting_sample = 300
    ending_sample = 90000
    use_cheby2 = True

    low_freq_vec = np.arange(0.01, 0.52, 0.1)
    high_freq_vec = np.arange(1, 7.5, 0.5)
    percH_values = np.arange(1, 2.1, 0.1)
    percL_values = np.arange(0.5, 1.05, 0.05)
    max_ripple_values = [0.5, 1, 1.5, 2, 3]
    attenuation_values = [15, 20, 30, 45]

    high_cutoff_freq = 5
    low_cutoff_freq = 0.5
    percH = 1.2
    percL = 0.8
    rpH = 1
    rpL = 1
    attenuation = 10
    maximum_filter_order = 20

    prototype_filter = {
        'low_stopband': 0.5,
        'high_stopband': 4
    }

    save_figures = False
    filter_data_classic = False
    filter_data_optimised = False
    filter_data_attenuation = not filter_data_classic
    plot_raw_data_flag = True

    raw_data = read_csv_file(select_data=select_data)
    if raw_data is None:
        return

    raw_data = raw_data.iloc[starting_sample:min(ending_sample, len(raw_data))]
    raw_data_t = raw_data.iloc[:, 0].values
    raw_data_x = raw_data.iloc[:, 1].values
    raw_data_y = raw_data.iloc[:, 2].values
    raw_data_z = raw_data.iloc[:, 3].values

    raw_data_total = np.sqrt(raw_data_x**2 + raw_data_y**2 + raw_data_z**2)
    time_interval = np.diff(raw_data_t)
    fs_mode_result = mode(time_interval)
    if fs_mode_result.count.size == 0:
        raise ValueError("Failed to compute the mode of time intervals")
    fs = int(np.ceil(1 / fs_mode_result.mode[0]))

    time_support = raw_data_t

    if plot_raw_data_flag:
        plot_raw_data(time_support, raw_data.values)

    if filter_data_attenuation:
        fNy = fs / 2
        stability = True

        while stability:
            filtData_x = apply_filter(raw_data_x, "cheby2", low_cutoff_freq, percL, rpL, attenuation, fs, btype='high')
            filtData_y = apply_filter(raw_data_y, "cheby2", low_cutoff_freq, percL, rpL, attenuation, fs, btype='high')
            filtData_z = apply_filter(raw_data_z, "cheby2", low_cutoff_freq, percL, rpL, attenuation, fs, btype='high')
            filtData_total = apply_filter(raw_data_total, "cheby2", low_cutoff_freq, percL, rpL, attenuation, fs, btype='high')

            filtData_x = apply_filter(filtData_x, "cheby2", high_cutoff_freq, percH, rpH, attenuation, fs, btype='low')
            filtData_y = apply_filter(filtData_y, "cheby2", high_cutoff_freq, percH, rpH, attenuation, fs, btype='low')
            filtData_z = apply_filter(filtData_z, "cheby2", high_cutoff_freq, percH, rpH, attenuation, fs, btype='low')
            filtData_total = apply_filter(filtData_total, "cheby2", high_cutoff_freq, percH, rpH, attenuation, fs, btype='low')

            if attenuation >= 20:
                print("\n The filter is not stable \n")
                stability = False

            attenuation += 2

    min_peak_prominence = 0.04
    min_peak_time_distance = 0.35
    min_threshold = 0.00001

    min_peak_distance = fs * min_peak_time_distance

    pks, _ = find_peaks(filtData_total, height=min_threshold, distance=min_peak_distance, prominence=min_peak_prominence)

    plt.figure()
    plt.plot(pks / fs, filtData_total[pks], 'o')
    plt.plot(np.arange(len(filtData_total)) / fs, filtData_total)
    plt.xlabel('Time [s]')
    plt.ylabel('Acceleration [m/s^2]')
    plt.title('HR peak detected on Original signal')
    plt.legend(['Heart beat', 'Filtered signal'])
    plt.show()

    dt_medio_sec = np.diff(pks) / fs
    instantaneous_hr = 60 / dt_medio_sec
    hr_time = (pks[:-1] + pks[1:]) / 2 / fs

    plt.figure()
    plt.plot(hr_time, instantaneous_hr)
    plt.xlabel('Time [s]')
    plt.ylabel('Heart Rate [BPM]')
    plt.title('Evolution of Heart Rate')
    plt.show()

    first_3_sec_indices = hr_time <= 3
    mean_hr_first_3_sec = np.mean(instantaneous_hr[first_3_sec_indices])
    print(f'\n Mean HR during the first 3 seconds: {mean_hr_first_3_sec} BPM \n')

    around_60_sec_indices = (hr_time >= 58.5) & (hr_time <= 61.5)
    mean_hr_around_60_sec = np.mean(instantaneous_hr[around_60_sec_indices])
    print(f'\nMean HR in the 3 seconds around the 60th second: {mean_hr_around_60_sec} BPM \n')

    n = 10
    num_windows = len(dt_medio_sec) // (n - 1)
    hr_evolution = np.zeros(num_windows)
    hr_evolution_time = np.zeros(num_windows)

    for i in range(num_windows):
        start_idx = i * (n - 1)
        end_idx = start_idx + n - 1
        if end_idx < len(dt_medio_sec):
            hr_evolution[i] = 60 / np.mean(dt_medio_sec[start_idx:end_idx])
            hr_evolution_time[i] = np.mean(hr_time[start_idx:end_idx])
        else:
            break

    print(f'\n The median HR frequency is {np.median(hr_evolution):.1f} \n')

    plt.figure()
    plt.plot(hr_evolution_time, hr_evolution)
    plt.title('Heart Rate Evolution Over Time - RAW')
    plt.xlabel('Time [s]')
    plt.ylabel('Heart Rate (bpm)')
    plt.grid(True)
    plt.show()

    if save_figures:
        plot_name = 'Heart rate evolution - RAW'
        plt.savefig(f'{plot_name}.png')

    use_exponential = True  # Assuming this variable is set earlier in the script
    if use_exponential:
        exp_data = hr_evolution
        X = hr_evolution_time
        Y = exp_data

        window_size = 20
        threshold_factor_higher = 0.5
        threshold_factor_lower = 0.5
        overlap_percentage = 0.2
        step_size = window_size * (1 - overlap_percentage / 100)

        cleanedX = []
        cleanedY = []
        discardedX = []
        discardedY = []

        i = 0
        while i <= len(Y):
            start_idx = max(0, int(round(i)))
            end_idx = min(len(Y), int(round(i + window_size - 1)))

            window_data = Y[start_idx:end_idx]

            window_mean = np.mean(window_data)
            window_std = np.std(window_data)

            for j in range(start_idx, end_idx):
                if window_mean - threshold_factor_lower * window_std <= Y[j] <= window_mean + threshold_factor_higher * window_std:
                    cleanedX.append(X[j])
                    cleanedY.append(Y[j])
                else:
                    discardedX.append(X[j])
                    discardedY.append(Y[j])

            i += step_size

        plt.figure()
        plt.plot(X, Y, 'b*', label='Original Data')
        plt.plot(cleanedX, cleanedY, 'go', label='Cleaned Data')
        plt.plot(discardedX, discardedY, 'rx', label='Discarded Data')
        plt.grid(True)
        plt.title(f'Data Cleaning with {window_size}-Sample Window')
        plt.xlabel('Time [s]')
        plt.ylabel('Heart Rate [BPM]')
        plt.legend(loc='best')
        plt.show()

        x = np.array(cleanedX)
        y = np.array(cleanedY)

        y_mean = np.mean(y)
        y_std = np.std(y)
        y_scaled = (y - y_mean) / y_std

        def model_func(x, a, b, c):
            return a * np.exp(-b * x) + c

        initial_guess_a = np.max(y_scaled) - np.min(y_scaled)
        initial_guess_b = 0.05
        initial_guess_c = np.min(y_scaled)
        beta0 = [initial_guess_a, initial_guess_b, initial_guess_c]

        popt, _ = curve_fit(model_func, x, y_scaled, p0=beta0)

        print(f'Decay parameters:\n a = {popt[0]:.4f}\n b = {popt[1]:.4f}\n c = {popt[2]:.4f}')

        y_fitted_scaled = model_func(x, *popt)
        y_fitted = y_fitted_scaled * y_std + y_mean

        plt.figure()
        plt.plot(x, y, 'b*', label='Noisy Y')
        plt.plot(x, y_fitted, 'r-', label='Fitted Y')
        plt.grid(True)
        plt.title('Exponential Regression')
        plt.xlabel('Time [s]')
        plt.ylabel('Heart Rate [BPM]')
        plt.legend(loc='north')
        formula_string = f'Y = {popt[0]:.3f} * exp(-{popt[1]:.3f} * X) + {popt[2]:.3f}'
        plt.text(np.mean(x), np.max(y), formula_string, fontsize=15, fontweight='bold')
        plt.show()

if __name__ == "__main__":
    main()