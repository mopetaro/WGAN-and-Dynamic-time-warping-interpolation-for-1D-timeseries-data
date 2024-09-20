#This was a one-time use code for me so it needs to be polished for re-using.

pip install fastdtw

pip install imbalanced-learn


import numpy as np
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import heapq
import time
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from scipy.io import savemat

# import the datasets (Disregards the code in this section since they were only applicable to my use. However, I
# kept them for reference.)

import GAN_functions

datasets = GAN_functions.GAN_preprocessing()

case_0_i_z = datasets['case_0_i_z']
case_1_i_z = datasets['case_1_i_z']
case_3_i_z = datasets['case_3_i_z']
case_4_i_z = datasets['case_4_i_z']
case_6_i_z = datasets['case_6_i_z']
case_7_i_z = datasets['case_7_i_z']
case_8_i_z = datasets['case_8_i_z']
case_9_i_z = datasets['case_9_i_z']
case_10_i_z = datasets['case_10_i_z']
case_11_i_z = datasets['case_11_i_z']
case_12_i_z = datasets['case_12_i_z']

all_cases = np.vstack((case_0_i_z, case_1_i_z, case_3_i_z, case_4_i_z, case_6_i_z, case_7_i_z, case_8_i_z, case_9_i_z, case_10_i_z, case_11_i_z, case_12_i_z))


# DTW:


def one_to_one_interpolation(series1, series2, dtw_path):
    interpolated_series = np.zeros_like(series1)

    for idx1, idx2 in dtw_path:
        # Calculate the mean for the current aligned points
        interpolated_point = (series1[idx1] + series2[idx2]) / 2
        interpolated_series[idx1] = interpolated_point

    return interpolated_series

some_threshold = 800
beta = 1 # Degree of adjustment
n_neighbors = 9 # 9 interpolations per timeseries data so we will have 500 timeseries per class in the end including the original ones
class_size = case_1_i_z.shape[0]

N = np.zeros((13, 1)) # Total number of fault samples needing to be generated

N[1] = (case_0_i_z.shape[0] - case_1_i_z.shape[0])*beta
N[3] = (case_0_i_z.shape[0] - case_3_i_z.shape[0])*beta
N[4] = (case_0_i_z.shape[0] - case_4_i_z.shape[0])*beta
N[6] = (case_0_i_z.shape[0] - case_6_i_z.shape[0])*beta
N[7] = (case_0_i_z.shape[0] - case_7_i_z.shape[0])*beta
N[8] = (case_0_i_z.shape[0] - case_8_i_z.shape[0])*beta
N[9] = (case_0_i_z.shape[0] - case_9_i_z.shape[0])*beta
N[10] = (case_0_i_z.shape[0] - case_10_i_z.shape[0])*beta
N[11] = (case_0_i_z.shape[0] - case_11_i_z.shape[0])*beta
N[12] = (case_0_i_z.shape[0] - case_12_i_z.shape[0])*beta


# This code block can be re-iterated for dynamic time warping between class pairs other than <Class #1> and <Any class expect Class #1>
####################################################
start_time = time.time()
shortest_dists_all = np.zeros((1, n_neighbors))
indices_all_1 = np.zeros((1, n_neighbors)) # This was a one-time use code for me, but if you want to use this code later, try not add this zero row to the beginning of arrays
r_case_1 = []
for i in range(case_1_i_z.shape[0]):
  start_time = time.time()
  k_1 = 0
  dists = []
  for j in range(case_0_i_z.shape[0], case_0_i_z.shape[0] + case_1_i_z.shape[0]):
    if j == i + case_0_i_z.shape[0]:
      print(j)
      continue
    else:
      distance, _ = fastdtw(case_1_i_z[i].reshape(-1, 1), all_cases[j].reshape(-1, 1), dist=euclidean)
      dists.append(distance)
  shortest_dists = heapq.nsmallest(n_neighbors, dists) # Get the n smallest values
  indices = [dists.index(value) for value in shortest_dists] # Get indices of the n smallest values
  for w in range(len(indices)):
    if indices[w] >= i:
      indices[w] += 1

  for w in range(len(indices)):
    if indices[w] < i:
      if indices[w] >= 0:
        k_1 += 1
    else:
      if indices[w] < case_1_i_z.shape[0]:
        k_1 += 1
  r_case_1.append(k_1/n_neighbors)
  indices_all_1 = np.vstack((indices_all_1, indices))
  shortest_dists_all = np.vstack((shortest_dists_all, shortest_dists))
  print('indices_all_1:')
  print(indices_all_1)
  print('shortest_dists_all:')
  print(shortest_dists_all)
  print('----------------------------------')
  print(i)
  # print('r_case_1:') # used for oversampling between different classes. I am not doing that so I will not use it.
  # print(r_case_1)

indices_all_1 = indices_all_1.astype(int)
end_time = time.time()  # Record the end time
elapsed_time = end_time - start_time  # Calculate elapsed time
elapsed_minutes = elapsed_time/60
print(f"Distance measurement took {elapsed_minutes:.2f} minutes")
####################################################



for dataset in [case_1_i_z, case_3_i_z, case_4_i_z, case_6_i_z, case_7_i_z, case_8_i_z, case_9_i_z, case_10_i_z, case_11_i_z, case_12_i_z]:

#--------------------------------------------------------------
  if np.array_equal(dataset, case_1_i_z):
    mode = 1
    indices_all = indices_all_1
  elif np.array_equal(dataset, case_3_i_z):
    mode = 3
    indices_all = indices_all_3
  elif np.array_equal(dataset, case_4_i_z):
    mode = 4
    indices_all = indices_all_4
  elif np.array_equal(dataset, case_6_i_z):
    mode = 6
    indices_all = indices_all_6
  elif np.array_equal(dataset, case_7_i_z):
    mode = 7
    indices_all = indices_all_7
  elif np.array_equal(dataset, case_8_i_z):
    mode = 8
    indices_all = indices_all_8
  elif np.array_equal(dataset, case_9_i_z):
    mode = 9
    indices_all = indices_all_9
  elif np.array_equal(dataset, case_10_i_z):
    mode = 10
    indices_all = indices_all_10
  elif np.array_equal(dataset, case_11_i_z):
    mode = 11
    indices_all = indices_all_11
  elif np.array_equal(dataset, case_12_i_z):
    mode = 12
    indices_all = indices_all_12
#--------------------------------------------------------------
  start_time = time.time()
  all_paths = []
  for j in range(class_size):
    paths = []
    for i in range(n_neighbors):
      _, p =fastdtw(dataset[j].reshape(-1, 1), dataset[indices_all[j+1][i]].reshape(-1, 1), dist=euclidean)
      paths.append(p)
    all_paths.append(paths)

  for j in range(class_size):
   for i in range(n_neighbors):
      inter = one_to_one_interpolation(dataset[j], dataset[indices_all[j+1][i]], all_paths[j][i])
      dataset = np.vstack((dataset, inter))
  end_time = time.time()  # Record the end time
  elapsed_time = end_time - start_time  # Calculate elapsed time
  elapsed_minutes = elapsed_time/60
  print(f"Interpolation took {elapsed_minutes:.2f} minutes")
  
  # Save the array to a .mat file

  if mode == 0:
    savemat('case_0_i_z_dtw_resampled.mat', {'case_0_i_z_dtw_resampled': dataset})
  elif mode == 1:
    savemat('case_1_i_z_dtw_resampled.mat', {'case_1_i_z_dtw_resampled': dataset})
  elif mode == 3:
    savemat('case_3_i_z_dtw_resampled.mat', {'case_3_i_z_dtw_resampled': dataset})
  elif mode == 4:
    savemat('case_4_i_z_dtw_resampled.mat', {'case_4_i_z_dtw_resampled': dataset})
  elif mode == 6:
    savemat('case_6_i_z_dtw_resampled.mat', {'case_6_i_z_dtw_resampled': dataset})
  elif mode == 7:
    savemat('case_7_i_z_dtw_resampled.mat', {'case_7_i_z_dtw_resampled': dataset})
  elif mode == 8:
    savemat('case_8_i_z_dtw_resampled.mat', {'case_8_i_z_dtw_resampled': dataset})
  elif mode == 9:
    savemat('case_9_i_z_dtw_resampled.mat', {'case_9_i_z_dtw_resampled': dataset})
  elif mode == 10:
    savemat('case_10_i_z_dtw_resampled.mat', {'case_10_i_z_dtw_resampled': dataset})
  elif mode == 11:
    savemat('case_11_i_z_dtw_resampled.mat', {'case_11_i_z_dtw_resampled': dataset})
  elif mode == 12:
    savemat('case_12_i_z_dtw_resampled.mat', {'case_12_i_z_dtw_resampled': dataset})


###  Other interpolation techniques that can be useful:

# # Example usage
# series1 = np.array([...])  # Your first time series
# series2 = np.array([...])  # Your second time series
# dtw_path = [...]           # Your DTW path

# interpolated_series = linear_interpolation(series1, series2, dtw_path)

def weighted_interpolation(series1, series2, dtw_path):
    interpolated_series = np.zeros_like(series1)
    prev_idx1, prev_idx2 = -1, -1

    for idx1, idx2 in dtw_path:
        # Calculate the number of points in the current segment for each series
        num_points1 = idx1 - prev_idx1
        num_points2 = idx2 - prev_idx2

        # Calculate the mean for each segment
        if num_points1 > 0:
            series1_mean = np.mean(series1[prev_idx1+1:idx1+1])
        else:
            series1_mean = 0  # No new points from series1

        if num_points2 > 0:
            series2_mean = np.mean(series2[prev_idx2+1:idx2+1])
        else:
            series2_mean = 0  # No new points from series2

        # Adjust weights based on the number of points
        total_points = max(num_points1, 1) + max(num_points2, 1)
        weight1 = max(num_points2, 1) / total_points
        weight2 = max(num_points1, 1) / total_points
        interpolated_point = (series1_mean * weight1 + series2_mean * weight2)

        interpolated_series[idx1] = interpolated_point
        prev_idx1, prev_idx2 = idx1, idx2

    return interpolated_series

def enhanced_weighted_interpolation(series1, series2, dtw_path, variation_threshold):
    interpolated_series = np.zeros_like(series1)
    # Initialize cumulative counts for one-to-many and many-to-one relationships
    cum_count1, cum_count2 = 0, 0
    prev_idx1, prev_idx2 = -1, -1

    for idx1, idx2 in dtw_path:
        num_points1 = idx1 - prev_idx1
        num_points2 = idx2 - prev_idx2

        # Check if we are in a continuous one-to-many or many-to-one relationship
        if num_points1 == 1 and num_points2 > 1:  # One-to-Many
            cum_count1 += 1
            cum_count2 = 0  # Reset the counter for series2
        elif num_points1 > 1 and num_points2 == 1:  # Many-to-One
            cum_count2 += 1
            cum_count1 = 0  # Reset the counter for series1
        else:  # One-to-One or change in pattern
            cum_count1, cum_count2 = 0, 0  # Reset counters

        # Calculate the mean for each segment
        series1_mean = np.mean(series1[max(0, prev_idx1+1):idx1+1]) if num_points1 > 0 else 0
        series2_mean = np.mean(series2[max(0, prev_idx2+1):idx2+1]) if num_points2 > 0 else 0

        # Introduce threshold for minor variations
        if abs(series1_mean - series2_mean) < variation_threshold:
            interpolated_point = (series1[idx1] + series2[idx2]) / 2
        else:
            # Adjust weights based on the cumulative counts
            total_weight = max(cum_count1, 1) + max(cum_count2, 1)
            weight1 = max(cum_count2, 1) / total_weight
            weight2 = max(cum_count1, 1) / total_weight
            interpolated_point = (series1_mean * weight1 + series2_mean * weight2)

        interpolated_series[idx1] = interpolated_point
        prev_idx1, prev_idx2 = idx1, idx2

    return interpolated_series

def sinusoidal_weighted_interpolation(series1, series2, dtw_path, variation_threshold):
    interpolated_series = np.zeros_like(series1)
    prev_idx1, prev_idx2 = -1, -1

    for idx1, idx2 in dtw_path:
        num_points1 = idx1 - prev_idx1
        num_points2 = idx2 - prev_idx2

        # Calculate the mean for each segment
        series1_mean = np.mean(series1[max(0, prev_idx1+1):idx1+1]) if num_points1 > 0 else 0
        series2_mean = np.mean(series2[max(0, prev_idx2+1):idx2+1]) if num_points2 > 0 else 0

        # Check if the points are within the variation threshold
        if abs(series1_mean - series2_mean) < variation_threshold:
            # If the variation is small, use a simple average
            interpolated_point = (series1[idx1] + series2[idx2]) / 2
        else:
            # If the variation is large, adjust the weights based on the proximity to peaks
            # Peaks in sinusoidal data are at -1 and 1
            proximity_to_peak = min(abs(series1_mean - 1), abs(series1_mean + 1), abs(series2_mean - 1), abs(series2_mean + 1))
            weight_factor = (1 - proximity_to_peak)  # Weight factor increases closer to peaks

            # Calculate weights for each series, incorporating the weight_factor
            total_weight = (num_points1 + num_points2) * weight_factor
            weight1 = (num_points2 / total_weight) * weight_factor
            weight2 = (num_points1 / total_weight) * weight_factor

            interpolated_point = (series1_mean * weight1 + series2_mean * weight2)

        interpolated_series[idx1] = interpolated_point
        prev_idx1, prev_idx2 = idx1, idx2

    return interpolated_series

def linear_interpolation(series1, series2, dtw_path):
    interpolated_series = np.zeros_like(series1)
    prev_idx1, prev_idx2 = -1, -1

    for idx1, idx2 in dtw_path:
        # One-to-many or many-to-one cases
        if idx1 == prev_idx1 or idx2 == prev_idx2:
            series1_mean = np.mean(series1[prev_idx1:idx1+1])
            series2_mean = np.mean(series2[prev_idx2:idx2+1])
            interpolated_point = (series1_mean + series2_mean) / 2
        else:  # One-to-one case
            interpolated_point = (series1[idx1] + series2[idx2]) / 2

        interpolated_series[idx1] = interpolated_point
        prev_idx1, prev_idx2 = idx1, idx2

    return interpolated_series