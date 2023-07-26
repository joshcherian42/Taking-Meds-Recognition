import os
import math
import settings
import itertools
import operator
import numpy as np
from scipy.fftpack import fft


coefficient_freqs = ['2.3', '3.125', '3.91', '4.69', '5.469',
                     '6.25', '7.03', '7.8125', '8.594', '9.375',
                     '10.125', '10.9375', '11.719', '12.5']

eps = 0.00000001  # Offset to avoid divide-by-zero error


def median(sorted_x):
    sorted_x.sort()
    if len(sorted_x) % 2 == 0:
        median = (sorted_x[int(len(sorted_x) / 2)] + sorted_x[int(len(sorted_x) / 2 - 1)]) / 2
    else:
        median = sorted_x[int(round(len(sorted_x) / 2))]
    return median


def euclidean_distance(x, y, z):
    euclid = list()
    for i in range(1, len(x)):
        euclid.append(math.sqrt(math.pow(x[i], 2) + math.pow(y[i], 2) + math.pow(z[i], 2)))
    return euclid


def feature_utils(time, x_acc, y_acc, z_acc):

    # Acc
    heights_x_acc = side_height(x_acc, time)
    heights_y_acc = side_height(y_acc, time)
    heights_z_acc = side_height(z_acc, time)

    x_peaks_acc = peaks(x_acc)
    y_peaks_acc = peaks(y_acc)
    z_peaks_acc = peaks(z_acc)

    x_valleys_acc = valleys(x_acc)
    y_valleys_acc = valleys(y_acc)
    z_valleys_acc = valleys(z_acc)

    stdev_peaks_x_acc = 0.0
    stdev_peaks_y_acc = 0.0
    stdev_peaks_z_acc = 0.0

    sorted_x_acc = list(x_acc)
    sorted_y_acc = list(y_acc)
    sorted_z_acc = list(z_acc)

    if not x_peaks_acc:
        avg_peaks_x_acc = median(sorted_x_acc)
    else:
        avg_peaks_x_acc = average(x_peaks_acc)
        stdev_peaks_x_acc = stdev(x_peaks_acc)

    if not y_peaks_acc:
        avg_peaks_y_acc = median(sorted_y_acc)
    else:
        avg_peaks_y_acc = average(y_peaks_acc)
        stdev_peaks_y_acc = stdev(y_peaks_acc)

    if not z_peaks_acc:
        avg_peaks_z_acc = median(sorted_z_acc)
    else:
        avg_peaks_z_acc = average(z_peaks_acc)
        stdev_peaks_z_acc = stdev(z_peaks_acc)

    stdev_valleys_x_acc = 0.0
    stdev_valleys_y_acc = 0.0
    stdev_valleys_z_acc = 0.0

    if not x_valleys_acc:
        avg_valleys_x_acc = median(x_acc)
    else:
        avg_valleys_x_acc = average(x_valleys_acc)
        stdev_valleys_x_acc = stdev(x_valleys_acc)

    if not y_valleys_acc:
        avg_valleys_y_acc = median(y_acc)
    else:
        avg_valleys_y_acc = average(y_valleys_acc)
        stdev_valleys_y_acc = stdev(y_valleys_acc)

    if not z_valleys_acc:
        avg_valleys_z_acc = median(z_acc)
    else:
        avg_valleys_z_acc = average(z_valleys_acc)
        stdev_valleys_z_acc = stdev(z_valleys_acc)

    avg_height_x_acc = 0.0
    stdev_heights_x_acc = 0.0
    avg_height_y_acc = 0.0
    stdev_heights_y_acc = 0.0
    avg_height_z_acc = 0.0
    stdev_heights_z_acc = 0.0

    if heights_x_acc:
        stdev_heights_x_acc = stdev(heights_x_acc)
        avg_height_x_acc = average(heights_x_acc)

    if heights_y_acc:
        stdev_heights_y_acc = stdev(heights_y_acc)
        avg_height_y_acc = average(heights_y_acc)

    if heights_z_acc:
        stdev_heights_x_acc = stdev(heights_z_acc)
        avg_height_x_acc = average(heights_z_acc)

    axis_overlap_acc = axis_order(x_acc, y_acc) + axis_order(y_acc, z_acc) + axis_order(x_acc, z_acc)

    return [avg_height_x_acc, avg_height_y_acc, avg_height_z_acc,
            stdev_heights_x_acc, stdev_heights_y_acc, stdev_heights_z_acc,
            x_peaks_acc, y_peaks_acc, z_peaks_acc,
            avg_peaks_x_acc, avg_peaks_y_acc, avg_peaks_z_acc,
            stdev_peaks_x_acc, stdev_peaks_y_acc, stdev_peaks_z_acc,
            x_valleys_acc, y_valleys_acc, z_valleys_acc,
            avg_valleys_x_acc, avg_valleys_y_acc, avg_valleys_z_acc,
            stdev_valleys_x_acc, stdev_valleys_y_acc, stdev_valleys_z_acc,
            axis_overlap_acc]


def gen_features(file_path, window_size):
    """Extracts features

    Given a file containing sensor data, this function separates the file into windows and extract features from each window

    Args:
        file_path (str): File from which features are to be extracted
        window_size (int): size of windows to segment data into

    """

    print('Generating Training Data for: ' + os.path.normpath(file_path).split(os.path.sep)[-1].rstrip())
    print('')
    window_size = int(window_size)

    raw_data = np.genfromtxt(file_path.rstrip(), delimiter=',', dtype=None, names=True, unpack=True, encoding=None)

    # Right Hand
    T_right = raw_data["Time_Right"]
    X_acc_right = raw_data["Acc_X_Right"]
    Y_acc_right = raw_data["Acc_Y_Right"]
    Z_acc_right = raw_data["Acc_Z_Right"]
    X_gyro_right = raw_data["Gyro_X_Right"]
    Y_gyro_right = raw_data["Gyro_Y_Right"]
    Z_gyro_right = raw_data["Gyro_Z_Right"]
    X_gravity_right = raw_data["Gravity_X_Right"]
    Y_gravity_right = raw_data["Gravity_Y_Right"]
    Z_gravity_right = raw_data["Gravity_Z_Right"]

    # Left Hand
    T_left = raw_data["Time_Left"]
    X_acc_left = raw_data["Acc_X_Left"]
    Y_acc_left = raw_data["Acc_Y_Left"]
    Z_acc_left = raw_data["Acc_Z_Left"]
    X_gyro_left = raw_data["Gyro_X_Left"]
    Y_gyro_left = raw_data["Gyro_Y_Left"]
    Z_gyro_left = raw_data["Gyro_Z_Left"]
    X_gravity_left = raw_data["Gravity_X_Left"]
    Y_gravity_left = raw_data["Gravity_Y_Left"]
    Z_gravity_left = raw_data["Gravity_Z_Left"]

    activity = raw_data['Activity']

    window_time = 0
    cur_window_time = window_time
    overlap_time = 0
    overlap_set = False

    while cur_window_time < len(T_right) and T_right[len(T_right) - 1] - T_right[window_time] >= window_size:

        # Set the first overlap time
        if T_right[window_time] - T_right[cur_window_time] >= window_size / 2 and not overlap_set:
            overlap_time = window_time
            overlap_set = True

        # Extract features if the window has become larger than the window size
        if T_right[window_time] - T_right[cur_window_time] >= window_size:

            if 'Test' in file_path:
                train_test = 'TEST'
            elif 'Training' in file_path:
                train_test = 'TRAIN'

            parse_data(T_right[cur_window_time:window_time],
                       T_left[cur_window_time:window_time],

                       # Right Hand
                       X_acc_right[cur_window_time:window_time],
                       Y_acc_right[cur_window_time:window_time],
                       Z_acc_right[cur_window_time:window_time],
                       X_gyro_right[cur_window_time:window_time],
                       Y_gyro_right[cur_window_time:window_time],
                       Z_gyro_right[cur_window_time:window_time],
                       X_gravity_right[cur_window_time:window_time],
                       Y_gravity_right[cur_window_time:window_time],
                       Z_gravity_right[cur_window_time:window_time],

                       # Left Hand
                       X_acc_left[cur_window_time:window_time],
                       Y_acc_left[cur_window_time:window_time],
                       Z_acc_left[cur_window_time:window_time],
                       X_gyro_left[cur_window_time:window_time],
                       Y_gyro_left[cur_window_time:window_time],
                       Z_gyro_left[cur_window_time:window_time],
                       X_gravity_left[cur_window_time:window_time],
                       Y_gravity_left[cur_window_time:window_time],
                       Z_gravity_left[cur_window_time:window_time],

                       activity[cur_window_time:window_time],
                       cur_window_time,
                       window_time,
                       os.path.normpath(file_path).split(os.path.sep)[-1].rstrip(),
                       train_test)

            cur_window_time = overlap_time
            window_time = overlap_time
            overlap_set = False

        window_time += 1


def parse_record(time_right, time_left,

                 x_acc_right, y_acc_right, z_acc_right,
                 x_gyro_right, y_gyro_right, z_gyro_right,
                 x_gravity_right, y_gravity_right, z_gravity_right,

                 x_acc_left, y_acc_left, z_acc_left,
                 x_gyro_left, y_gyro_left, z_gyro_left,
                 x_gravity_left, y_gravity_left, z_gravity_left,

                 activity, start, end):
    """ Calculates features for current time slice
    Args:
        Raw Feature Data
        Activity Label
        Start Time
        End Time
    Returns:
        dict - maps feature name to value (record in pandas DataFrame)
    """

    # RIGHT HAND FEATURES
    (avg_height_x_acc_right, avg_height_y_acc_right, avg_height_z_acc_right,
     stdev_heights_x_acc_right, stdev_heights_y_acc_right, stdev_heights_z_acc_right,
     x_peaks_acc_right, y_peaks_acc_right, z_peaks_acc_right,
     avg_peaks_x_acc_right, avg_peaks_y_acc_right, avg_peaks_z_acc_right,
     stdev_peaks_x_acc_right, stdev_peaks_y_acc_right, stdev_peaks_z_acc_right,
     x_valleys_acc_right, y_valleys_acc_right, z_valleys_acc_right,
     avg_valleys_x_acc_right, avg_valleys_y_acc_right, avg_valleys_z_acc_right,
     stdev_valleys_x_acc_right, stdev_valleys_y_acc_right, stdev_valleys_z_acc_right,
     axis_overlap_acc_right) = feature_utils(time_right, x_acc_right, y_acc_right, z_acc_right)

    # LEFT HAND FEATURES
    (avg_height_x_acc_left, avg_height_y_acc_left, avg_height_z_acc_left,
     stdev_heights_x_acc_left, stdev_heights_y_acc_left, stdev_heights_z_acc_left,
     x_peaks_acc_left, y_peaks_acc_left, z_peaks_acc_left,
     avg_peaks_x_acc_left, avg_peaks_y_acc_left, avg_peaks_z_acc_left,
     stdev_peaks_x_acc_left, stdev_peaks_y_acc_left, stdev_peaks_z_acc_left,
     x_valleys_acc_left, y_valleys_acc_left, z_valleys_acc_left,
     avg_valleys_x_acc_left, avg_valleys_y_acc_left, avg_valleys_z_acc_left,
     stdev_valleys_x_acc_left, stdev_valleys_y_acc_left, stdev_valleys_z_acc_left,
     axis_overlap_acc_left) = feature_utils(time_left, x_acc_left, y_acc_left, z_acc_left)

    nFFT = int((end - start) / 2)

    x_acc_right_n = abs(fft(np.asarray(x_acc_right) - np.asarray(x_acc_right).mean()))[0:nFFT]
    y_acc_right_n = abs(fft(np.asarray(y_acc_right) - np.asarray(y_acc_right).mean()))[0:nFFT]
    z_acc_right_n = abs(fft(np.asarray(z_acc_right) - np.asarray(z_acc_right).mean()))[0:nFFT]
    x_acc_left_n = abs(fft(np.asarray(x_acc_left) - np.asarray(x_acc_left).mean()))[0:nFFT]
    y_acc_left_n = abs(fft(np.asarray(y_acc_left) - np.asarray(y_acc_left).mean()))[0:nFFT]
    z_acc_left_n = abs(fft(np.asarray(z_acc_left) - np.asarray(z_acc_left).mean()))[0:nFFT]

    x_acc_right_n = x_acc_right_n / len(x_acc_right_n)
    y_acc_right_n = y_acc_right_n / len(y_acc_right_n)
    z_acc_right_n = z_acc_right_n / len(z_acc_right_n)
    x_acc_left_n = x_acc_left_n / len(x_acc_left_n)
    y_acc_left_n = y_acc_left_n / len(y_acc_left_n)
    z_acc_left_n = z_acc_left_n / len(z_acc_left_n)

    cur_features = [avg_jerk(x_acc_right, time_right),
                    avg_jerk(y_acc_right, time_right),
                    avg_jerk(z_acc_right, time_right),
                    avg_height_x_acc_right,
                    avg_height_y_acc_right,
                    avg_height_z_acc_right,
                    stdev_heights_x_acc_right,
                    stdev_heights_y_acc_right,
                    stdev_heights_z_acc_right,
                    energy(x_acc_right),
                    energy(y_acc_right),
                    energy(z_acc_right),
                    entropy(x_acc_right),
                    entropy(y_acc_right),
                    entropy(z_acc_right),
                    average(x_acc_right),
                    average(y_acc_right),
                    average(z_acc_right),
                    stdev(x_acc_right),
                    stdev(y_acc_right),
                    stdev(z_acc_right),
                    rms(x_acc_right),
                    rms(y_acc_right),
                    rms(z_acc_right),
                    len(x_peaks_acc_right),
                    len(y_peaks_acc_right),
                    len(z_peaks_acc_right),
                    avg_peaks_x_acc_right,
                    avg_peaks_y_acc_right,
                    avg_peaks_z_acc_right,
                    stdev_peaks_x_acc_right,
                    stdev_peaks_y_acc_right,
                    stdev_peaks_z_acc_right,
                    len(x_valleys_acc_right),
                    len(y_valleys_acc_right),
                    len(z_valleys_acc_right),
                    avg_valleys_x_acc_right,
                    avg_valleys_y_acc_right,
                    avg_valleys_z_acc_right,
                    stdev_valleys_x_acc_right,
                    stdev_valleys_y_acc_right,
                    stdev_valleys_z_acc_right,
                    axis_overlap_acc_right,
                    fractal_dimension(x_acc_right, y_acc_right, z_acc_right),
                    spectral_centroid(x_acc_right_n),
                    spectral_centroid(y_acc_right_n),
                    spectral_centroid(z_acc_right_n),
                    spectral_spread(x_acc_right),
                    spectral_spread(y_acc_right),
                    spectral_spread(z_acc_right),
                    spectral_rolloff(x_acc_right),
                    spectral_rolloff(y_acc_right),
                    spectral_rolloff(z_acc_right),

                    avg_jerk(x_acc_left, time_left),
                    avg_jerk(y_acc_left, time_left),
                    avg_jerk(z_acc_left, time_left),
                    avg_height_x_acc_left,
                    avg_height_y_acc_left,
                    avg_height_z_acc_left,
                    stdev_heights_x_acc_left,
                    stdev_heights_y_acc_left,
                    stdev_heights_z_acc_left,
                    energy(x_acc_left),
                    energy(y_acc_left),
                    energy(z_acc_left),
                    entropy(x_acc_left),
                    entropy(y_acc_left),
                    entropy(z_acc_left),
                    average(x_acc_left),
                    average(y_acc_left),
                    average(z_acc_left),
                    stdev(x_acc_left),
                    stdev(y_acc_left),
                    stdev(z_acc_left),
                    rms(x_acc_left),
                    rms(y_acc_left),
                    rms(z_acc_left),
                    len(x_peaks_acc_left),
                    len(y_peaks_acc_left),
                    len(z_peaks_acc_left),
                    avg_peaks_x_acc_left,
                    avg_peaks_y_acc_left,
                    avg_peaks_z_acc_left,
                    stdev_peaks_x_acc_left,
                    stdev_peaks_y_acc_left,
                    stdev_peaks_z_acc_left,
                    len(x_valleys_acc_left),
                    len(y_valleys_acc_left),
                    len(z_valleys_acc_left),
                    avg_valleys_x_acc_left,
                    avg_valleys_y_acc_left,
                    avg_valleys_z_acc_left,
                    stdev_valleys_x_acc_left,
                    stdev_valleys_y_acc_left,
                    stdev_valleys_z_acc_left,
                    axis_overlap_acc_left,
                    fractal_dimension(x_acc_left, y_acc_left, z_acc_left),
                    spectral_centroid(x_acc_left_n),
                    spectral_centroid(y_acc_left_n),
                    spectral_centroid(z_acc_left_n),
                    spectral_spread(x_acc_left),
                    spectral_spread(y_acc_left),
                    spectral_spread(z_acc_left),
                    spectral_rolloff(x_acc_left),
                    spectral_rolloff(y_acc_left),
                    spectral_rolloff(z_acc_left),

                    activity_mode(activity),
                    start, end]

    return {settings.features_header[i]: cur_features[i] for i in range(len(cur_features))}


def parse_data(time_right, time_left,

               x_acc_right, y_acc_right, z_acc_right,
               x_gyro_right, y_gyro_right, z_gyro_right,
               x_gravity_right, y_gravity_right, z_gravity_right,

               x_acc_left, y_acc_left, z_acc_left,
               x_gyro_left, y_gyro_left, z_gyro_left,
               x_gravity_left, y_gravity_left, z_gravity_left,

               activity, start, end):
    """ Returns a list of features for a given timeslice
    """

    # RIGHT HAND FEATURES
    (avg_height_x_acc_right, avg_height_y_acc_right, avg_height_z_acc_right,
     stdev_heights_x_acc_right, stdev_heights_y_acc_right, stdev_heights_z_acc_right,
     x_peaks_acc_right, y_peaks_acc_right, z_peaks_acc_right,
     avg_peaks_x_acc_right, avg_peaks_y_acc_right, avg_peaks_z_acc_right,
     stdev_peaks_x_acc_right, stdev_peaks_y_acc_right, stdev_peaks_z_acc_right,
     x_valleys_acc_right, y_valleys_acc_right, z_valleys_acc_right,
     avg_valleys_x_acc_right, avg_valleys_y_acc_right, avg_valleys_z_acc_right,
     stdev_valleys_x_acc_right, stdev_valleys_y_acc_right, stdev_valleys_z_acc_right,
     axis_overlap_acc_right) = feature_utils(time_right, x_acc_right, y_acc_right, z_acc_right)

    # LEFT HAND FEATURES
    (avg_height_x_acc_left, avg_height_y_acc_left, avg_height_z_acc_left,
     stdev_heights_x_acc_left, stdev_heights_y_acc_left, stdev_heights_z_acc_left,
     x_peaks_acc_left, y_peaks_acc_left, z_peaks_acc_left,
     avg_peaks_x_acc_left, avg_peaks_y_acc_left, avg_peaks_z_acc_left,
     stdev_peaks_x_acc_left, stdev_peaks_y_acc_left, stdev_peaks_z_acc_left,
     x_valleys_acc_left, y_valleys_acc_left, z_valleys_acc_left,
     avg_valleys_x_acc_left, avg_valleys_y_acc_left, avg_valleys_z_acc_left,
     stdev_valleys_x_acc_left, stdev_valleys_y_acc_left, stdev_valleys_z_acc_left,
     axis_overlap_acc_left) = feature_utils(time_left, x_acc_left, y_acc_left, z_acc_left)

    cur_features = [avg_jerk(x_acc_right, time_right),
                    avg_jerk(y_acc_right, time_right),
                    avg_jerk(z_acc_right, time_right),
                    avg_height_x_acc_right,
                    avg_height_y_acc_right,
                    avg_height_z_acc_right,
                    stdev_heights_x_acc_right,
                    stdev_heights_y_acc_right,
                    stdev_heights_z_acc_right,
                    energy(x_acc_right),
                    energy(y_acc_right),
                    energy(z_acc_right),
                    entropy(x_acc_right),
                    entropy(y_acc_right),
                    entropy(z_acc_right),
                    average(x_acc_right),
                    average(y_acc_right),
                    average(z_acc_right),
                    stdev(x_acc_right),
                    stdev(y_acc_right),
                    stdev(z_acc_right),
                    sig_corr(x_acc_right, y_acc_right),
                    sig_corr(x_acc_right, z_acc_right),
                    sig_corr(y_acc_right, z_acc_right),
                    rms(x_acc_right),
                    rms(y_acc_right),
                    rms(z_acc_right),
                    len(x_peaks_acc_right),
                    len(y_peaks_acc_right),
                    len(z_peaks_acc_right),
                    avg_peaks_x_acc_right,
                    avg_peaks_y_acc_right,
                    avg_peaks_z_acc_right,
                    stdev_peaks_x_acc_right,
                    stdev_peaks_y_acc_right,
                    stdev_peaks_z_acc_right,
                    len(x_valleys_acc_right),
                    len(y_valleys_acc_right),
                    len(z_valleys_acc_right),
                    avg_valleys_x_acc_right,
                    avg_valleys_y_acc_right,
                    avg_valleys_z_acc_right,
                    stdev_valleys_x_acc_right,
                    stdev_valleys_y_acc_right,
                    stdev_valleys_z_acc_right,
                    axis_overlap_acc_right,
                    fractal_dimension(x_acc_right, y_acc_right, z_acc_right),


                    avg_jerk(x_acc_left, time_left),
                    avg_jerk(y_acc_left, time_left),
                    avg_jerk(z_acc_left, time_left),
                    avg_height_x_acc_left,
                    avg_height_y_acc_left,
                    avg_height_z_acc_left,
                    stdev_heights_x_acc_left,
                    stdev_heights_y_acc_left,
                    stdev_heights_z_acc_left,
                    energy(x_acc_left),
                    energy(y_acc_left),
                    energy(z_acc_left),
                    entropy(x_acc_left),
                    entropy(y_acc_left),
                    entropy(z_acc_left),
                    average(x_acc_left),
                    average(y_acc_left),
                    average(z_acc_left),
                    stdev(x_acc_left),
                    stdev(y_acc_left),
                    stdev(z_acc_left),
                    sig_corr(x_acc_left, y_acc_left),
                    sig_corr(x_acc_left, z_acc_left),
                    sig_corr(y_acc_left, z_acc_left),
                    rms(x_acc_left),
                    rms(y_acc_left),
                    rms(z_acc_left),
                    len(x_peaks_acc_left),
                    len(y_peaks_acc_left),
                    len(z_peaks_acc_left),
                    avg_peaks_x_acc_left,
                    avg_peaks_y_acc_left,
                    avg_peaks_z_acc_left,
                    stdev_peaks_x_acc_left,
                    stdev_peaks_y_acc_left,
                    stdev_peaks_z_acc_left,
                    len(x_valleys_acc_left),
                    len(y_valleys_acc_left),
                    len(z_valleys_acc_left),
                    avg_valleys_x_acc_left,
                    avg_valleys_y_acc_left,
                    avg_valleys_z_acc_left,
                    stdev_valleys_x_acc_left,
                    stdev_valleys_y_acc_left,
                    stdev_valleys_z_acc_left,
                    axis_overlap_acc_left,
                    fractal_dimension(x_acc_left, y_acc_left, z_acc_left),
                    sig_corr(x_acc_right, x_acc_left),
                    sig_corr(x_acc_right, y_acc_left),
                    sig_corr(x_acc_right, z_acc_left),
                    sig_corr(y_acc_right, x_acc_left),
                    sig_corr(y_acc_right, y_acc_left),
                    sig_corr(y_acc_right, z_acc_left),
                    sig_corr(z_acc_right, x_acc_left),
                    sig_corr(z_acc_right, y_acc_left),
                    sig_corr(z_acc_right, z_acc_left),

                    activity_mode(activity),
                    # binary_activity(activity),
                    time_right[0], time_right[len(time_right) - 1]]

    return cur_features
    # writer.writerow(cur_features)
    # w.close()
    # return activity_mode(activity)


'''
****************************************************
**                     Features                   **
****************************************************
     * Spectral Coefficients
     * Most Common Activity
     * Average Jerk
     * Average Distance Between Axes
     * Axis Order
     * Energy
     * Entropy
     * Mean of height of sides
     * StdDev of height of sides
     * Mean of distance from peak/valley to mean
     * StdDev of distance from peak/valley to mean
     * Average
     * Standard Deviation
     * Correlation
     * RMS
     * Peaks
     * Valleys
     * Zero Crossings
     * Fractal Dimension
     * Spectral Centroid
     * Spectral Spread
     * Spectral Rolloff
'''


# def spectral_coeff(X):
#     fig, ax = plt.subplots(1, 1)
#     Pg, freqs, bins, im = ax.specgram(X, NFFT=32, Fs=settings.sampling_rate, noverlap=0)

#     coef = []
#     for row in Pg:
#         coef.append(sum(row) / float(len(row)))
#     plt.close()
#     return coef


def most_common(L):
    # get an iterable of (item, iterable) pairs
    SL = sorted((x, i) for i, x in enumerate(L))
    groups = itertools.groupby(SL, key=operator.itemgetter(0))

    # auxiliary function to get "quality" for an item
    def _auxfun(g):
        item, iterable = g
        count = 0
        min_index = len(L)
        for _, where in iterable:
            count += 1
            min_index = min(min_index, where)
        return count, -min_index
    # pick the highest-count/earliest item
    return max(groups, key=_auxfun)[0]


# Find most common activity
def activity_mode(activities):
    act_types = {}
    for activity in activities:
        if activity in act_types:
            act_types[activity] = act_types[activity] + 1
        else:
            act_types[activity] = 1

    activity = str(max(act_types, key=act_types.get)).lower().strip()

    return activity


# Find most common activity
def binary_activity(activities):
    act_types = {}
    for activity in activities:
        if activity in act_types:
            act_types['inactive'] = act_types[activity] + 1
        else:
            act_types[activity] = 1

    activity = str(max(act_types, key=act_types.get)).lower().strip()
    if 'wash' in activity:
        return activity
    else:
        return 'inactive'


# Find Average Jerk
def avg_jerk(x, time):
    jerk = 0.0
    for i in range(1, len(x)):
        if time[i] - time[i - 1] <= 0:
            print("Jerk bug:", x[i], x[i - 1], "(", time[i], time[i - 1], ")")
        else:
            jerk += (x[i] - x[i - 1]) / (time[i] - time[i - 1])

    # if len(x) == 1:
    return jerk
    # else:
    #    return round(jerk/(len(x)-1))


# Find Average Distance Between Each Value
def avg_diff(x, y):
    diff = 0.0

    for i in range(1, len(x)):
        diff += x[i] - y[i]

    return diff / len(x)


# Finds number of times axis order changes
def axis_order(x, y):
    changes = 0
    xgreatery = None

    for cnt in range(len(x)):

        if (cnt == 0):
            if x[cnt] > y[cnt]:
                xgreatery = True
            elif x[cnt] < y[cnt]:
                xgreatery = None
        else:
            if x[cnt] > y[cnt]:
                if not xgreatery:
                    changes += 1
            elif x[cnt] < y[cnt]:
                if xgreatery:
                    changes += 1

    return changes


# Find Energy
def energy(x):
    energy = 0

    for k in range(len(x)):

        ak = 0.0
        bk = 0.0
        for i in range(len(x)):
            angle = 2 * math.pi * i * k / len(x)
            ak += x[i] * math.cos(angle)
            bk += -x[i] * math.sin(angle)

        energy += (math.pow(ak, 2) + math.pow(bk, 2)) / len(x)

    return energy


# Find Entropy
def entropy(x):
    spectralentropy = 0.0
    for j in range(len(x)):
        ak = 0.0
        bk = 0.0
        aj = 0.0
        bj = 0.0
        mag_j = 0.0
        mag_k = 0.0
        cj = 0.0

        for i in range(len(x)):
            angle = 2 * math.pi * i * j / len(x)
            ak = x[i] * math.cos(angle)   # Real
            bk = -x[i] * math.sin(angle)  # Imaginary
            aj += ak
            bj += bk

            mag_k += math.sqrt(math.pow(ak, 2) + math.pow(bk, 2))

        mag_j = math.sqrt(math.pow(aj, 2) + math.pow(bj, 2))
        if mag_k != 0 and mag_j != 0:
            cj = mag_j / mag_k
            spectralentropy += cj * math.log(cj) / math.log(2)
            return -spectralentropy
        else:
            return 0


# calculates side_height
def side_height(x, time):
    heights = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q1_points = []
    cur_q3_points = []
    peaks_valleys = []

    sorted_x = list(x)
    cur_median = median(sorted_x)

    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0
    for i in range(len(x)):
        cur_x = x[i]
        if i == 0:
            if cur_x > q3:
                cur_q3_points.append(cur_x)
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
            else:
                cur_q1_points.append(cur_x)
        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    if cur_q1_points:
                        peaks_valleys.append(min(cur_q1_points))  # add valley
                    del cur_q1_points[:]
                    moved_to_middle = None
                cur_q3_points.append(cur_x)
            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    if cur_q3_points:
                        peaks_valleys.append(max(cur_q3_points))  # add peak

                    del cur_q3_points[:]
                    moved_to_middle = None

                cur_q1_points.append(cur_x)
                q1_check = None
                q3_check = None

    for i in range(len(peaks_valleys) - 1):
        heights.append(abs(peaks_valleys[i + 1] - peaks_valleys[i]))

    return heights


# calculates the distance from the peak/valley to the mean
def dist_to_mean(x):
    avg = 0.0
    increasing = None
    decreasing = None
    dist = []

    avg = average(x)

    for i in range(len(x)):
        if x[i] > x[i - 1]:
            increasing = True
            if decreasing:
                dist.append(avg - x[i - 1])
                decreasing = None
        elif x[i] < x[i - 1]:
            decreasing = None
            if increasing:
                dist.append(x[i - 1] - avg)
                increasing = None
    return dist


# calculates average
def average(x):
    avg = 0.0
    for cnt in x:
        try:
            cnt = float(cnt)
        except ValueError:
            print(cnt)
        avg += cnt
    return avg / len(x)


# Find Standard Deviation
def stdev(x):
    avg = average(x)
    std = 0.0
    for cur_x in x:
        std += math.pow((cur_x - avg), 2)
    return math.sqrt(std / len(x))


# Find Signal Correlation
def sig_corr(x, y):
    correlation = 0.0
    for cnt in range(min(len(x), len(y))):
        correlation += x[cnt] * y[cnt]

    return correlation / len(x)


# Find Root Mean Square
def rms(x):
    avg = 0.0

    for cnt in x:
        avg += math.pow(cnt, 2)

    return math.sqrt(avg / len(x))


def peaks(x):
    peaks = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q3_points = []

    sorted_x = list(x)
    cur_median = median(sorted_x)
    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0
    for i, cur_x in enumerate(x):
        if i == 0:
            if cur_x > q3:
                cur_q3_points.append(cur_x)
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    moved_to_middle = None
                cur_q3_points.append(cur_x)
            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    if cur_q3_points:
                        peaks.append(max(cur_q3_points))  # add peak

                    del cur_q3_points[:]
                    moved_to_middle = None

                q1_check = None
                q3_check = None

    return peaks


def valleys(x):
    valleys = []

    q1_check = None  # true greater than, false less than
    q3_check = None
    moved_to_middle = None
    cur_q1_points = []

    sorted_x = list(x)
    cur_median = median(sorted_x)
    q1 = min(x) + abs((cur_median - min(x)) / 2)
    q3 = cur_median + abs((max(x) - cur_median) / 2)

    cur_x = 0.0

    for i, cur_x in enumerate(x):
        if i == 0:
            if cur_x > q3:
                q1_check = True
                q3_check = True
            elif cur_x > q1:
                q1_check = True
            else:
                cur_q1_points.append(cur_x)

        else:
            if cur_x > q3:
                q3_check = True
                q1_check = True
                if moved_to_middle:
                    if cur_q1_points:
                        valleys.append(min(cur_q1_points))  # add valley

                    del cur_q1_points[:]
                    moved_to_middle = None

            elif cur_x > q1:
                if (q3_check and q1_check) or (not q3_check and not q1_check):
                    moved_to_middle = True

                q1_check = True
                q3_check = None
            else:
                if moved_to_middle:
                    moved_to_middle = None

                cur_q1_points.append(cur_x)
                q1_check = None
                q3_check = None

    return valleys


# Find Zero Crossings
def z_crossings(x):
    cur_sign = 0
    prev_sign = 0
    sign = 0
    cnt = 0
    crossings = 0

    while prev_sign == 0 and cnt < len(x) - 1:
        prev_sign = math.copysign(1, x[cnt])
        cnt += 1

    if prev_sign == 0:
        return crossings

    while cnt < len(x):
        cur_sign = math.copysign(1, x[cnt])
        while cur_sign == 0 and cnt < len(x) - 1:
            cnt += 1
            cur_sign = math.copysign(1, x[cnt])

        if cur_sign == 0:  # the last value was zero, so no more crossings will occur
            break

        sign = cur_sign - prev_sign

        if sign == 2:  # 1-(-1)
            crossings += 1
            break
        elif sign == 0:  # 1-(+1), -1-(-1)
            break
        elif sign == -2:  # -1-(+1)
            crossings += 1
            break

        prev_sign = cur_sign
        cnt += 1

    return crossings


def boxcount(bitmap, k):
    """Counts number of boxes in the bitmap

    Returns the number of boxes of size k in the bitmap.
    From https://github.com/rougier/numpy-100 (#87)

    Args:
        bitmap (list): bitmap where 1s represent the scanpath
        k (int): box size

    Returns:
        int: number of boxes in the bitmap
    """

    boxes = []
    for x, y, z in bitmap:

        x_pos = math.floor(int(round(x * 100)) / k)
        y_pos = math.floor(int(round(y * 100)) / k)
        z_pos = math.floor(int(round(z * 100)) / k)

        if [x_pos, y_pos, z_pos] not in boxes:
            boxes.append([x_pos, y_pos, z_pos])

    return len(boxes)


def fractal_dimension(x, y, z):
    """Calculates the Minkowski-Bouligand dimension (Fractal Dimension)

    Calculates the Minkowski-Bouligand dimension (Fractal Dimension), which counts the number of boxes as the size required to cover the scanpath as the size of the box decreases

    From https://gist.github.com/rougier/e5eafc276a4e54f516ed5559df4242c0

    Args:
        x (list): x-axis sensor data
        y (list): y-axis sensor data
        z (list): z-axis sensor data

    Returns:
        int: Minkowski-Bouligand dimension
    """

    # Minimal dimension of image
    p = min(int(round(max(x) * 100)) - int(round(min(x) * 100)),
            int(round(max(y) * 100)) - int(round(min(y) * 100)),
            int(round(max(z) * 100)) - int(round(min(z) * 100)))

    # Greatest power of 2 less than or equal to p
    n = 2**np.floor(np.log(p) / np.log(2))

    # Extract the exponent
    try:
        n = int(np.log(n) / np.log(2))
    except:
        print("p =", p)
        print("n =", n)
        # if n == 0, then no boxes
        return 0

    # Build successive box sizes (from 2**n down to 2**1)
    sizes = 2**np.arange(n, 1, -1)

    # Actual box counting with decreasing size
    counts = []
    for size in sizes:
        counts.append(boxcount(zip(*[x, y, z]), size))

    if counts:
        # Fit the successive log(sizes) with log (counts)
        # RankWarning: Polyfit may be poorly conditioned
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]
    else:
        return 0


def spectral_centroid(x):
    """Calculates the spectral centroid

    Calculates the spectral centroid, which is the weighted mean of the frequencies present in the signal

    From https://github.com/tyiannak/recognizeFitExercise/blob/master/accelerometer.py

    Args:
        x (list): x-axis sensor data

    Returns:
        float: Spectral centroid
    """

    ind = (np.arange(1, len(x) + 1)) * (settings.sampling_rate / (2.0 * len(x)))

    Xt = x.copy()
    Xt = Xt / (Xt.max() + eps)
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(np.sum(((ind - C) ** 2) * Xt) / DEN)

    # Normalize:
    C = C / (settings.sampling_rate / 2.0)

    return C


def spectral_spread(x):
    """Calculates the spectral spread

    Calculates the spectral spread, which is the average deviation from the centroid

    From https://github.com/tyiannak/recognizeFitExercise/blob/master/accelerometer.py

    Args:
        x (list): x-axis sensor data

    Returns:
        float: Spectral spread
    """

    ind = (np.arange(1, len(x) + 1)) * (settings.sampling_rate / (2.0 * len(x)))

    Xt = x.copy()
    Xt = Xt / (Xt.max() + eps)
    NUM = np.sum(ind * Xt)
    DEN = np.sum(Xt) + eps

    # Centroid:
    C = (NUM / DEN)

    # Spread:
    S = np.sqrt(abs(np.sum(((ind - C) ** 2) * Xt) / DEN))

    # Normalize:
    S = S / (settings.sampling_rate / 2.0)

    return S


def spectral_rolloff(x):
    """Calculates the spectral centroid

    Calculates the spectral rolloff, which is the frequency below which c% of the total spectral energy is concentrated

    From https://github.com/tyiannak/recognizeFitExercise/blob/master/accelerometer.py

    Args:
        x (list): x-axis sensor data

    Returns:
        float: Spectral rolloff
    """

    c = 0.85
    totalEnergy = np.sum(x ** 2)
    fftLength = len(x)
    Thres = c * totalEnergy

    # Find the spectral rolloff as the frequency position where the respective spectral energy is equal to c*totalEnergy
    CumSum = np.cumsum(x ** 2) + eps
    [a, ] = np.nonzero(CumSum > Thres)

    if len(a) > 0:
        mC = np.float64(a[0]) / (float(fftLength))
    else:
        mC = 0.0

    return (mC)
