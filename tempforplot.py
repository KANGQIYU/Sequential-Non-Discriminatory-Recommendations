from sklearn import decomposition
import numpy as np
from sklearn.cluster import KMeans
import argparse
from numpy.linalg import inv
from numpy.random import rand
from numpy.random import randn
from numpy import eye
from numpy import argmax
import time
import matplotlib.pyplot as plt
import scipy.io



sumr_t_pro_seq_runs = np.load('sumr_t_pro_seq_runs.npy')
sumr_t_full_seq_runs = np.load('sumr_t_full_seq_runs.npy')
sumr_t_unprotect_seq_runs = np.load('sumr_t_unprotect_seq_runs.npy')
gender_pro_seq_runs = np.load('gender_pro_seq_runs.npy')
gender_full_seq_runs = np.load('gender_full_seq_runs.npy')
gender_unprotect_seq_runs = np.load('gender_unprotect_seq_runs.npy')

runtimes = gender_full_seq_runs.shape[0]
r_full = np.sum(np.sum(sumr_t_full_seq_runs, axis=2), axis=0)/runtimes
r_pro= np.sum(np.sum(sumr_t_pro_seq_runs, axis=2), axis=0)/runtimes
r_unprotect = np.sum(np.sum(sumr_t_unprotect_seq_runs, axis=2), axis=0)/runtimes

temp_full = gender_full_seq_runs[:, 1:, [0,1]]/gender_full_seq_runs[:, 1:, [2,3]]
female_full = np.sum(temp_full[:,:,0], axis=0)/runtimes
male_full = np.sum(temp_full[:,:,1], axis=0)/runtimes

temp_pro = gender_pro_seq_runs[:, 1:, [0, 1]] / gender_pro_seq_runs[:, 1:, [2, 3]]
female_pro = np.sum(temp_pro[:, :, 0], axis=0) /runtimes
male_pro = np.sum(temp_pro[:, :, 1], axis=0) / runtimes

temp_unprotect = gender_unprotect_seq_runs[:, 1:, [0, 1]] / gender_unprotect_seq_runs[:, 1:, [2, 3]]
female_unprotect = np.sum(temp_unprotect[:, :, 0], axis=0) / runtimes
male_unprotect = np.sum(temp_unprotect[:, :, 1], axis=0) / runtimes




plt.figure(1)
plt.title('rating')
plt.plot(r_full, 'r-', r_pro, 'g--', r_unprotect, 'b-.')
plt.figure(2)
plt.title('gender_differece')
plt.plot(female_full, 'r-', male_full, 'r-.', female_pro, 'g-', male_pro, 'g-.',\
         female_unprotect, 'b-', male_unprotect, 'b-.')
plt.show()

print(1)