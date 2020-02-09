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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scipy.io
import math
import multiprocessing

def main():

    n_trial = 10000
    recording_time = 100
    result = np.load('./result/wine/result_0.npy')
    for i in range(1, 8):
        result = np.concatenate((result, np.load('./result/wine/result_'+str(i)+'.npy')),axis=0)
    sumr_t_pro_seq_runs = result[:, :, 0]
    sumr_t_full_seq_runs = result[:, :, 1]
    sumr_t_unprotect_seq_runs = result[:, :, 2]
    sumr_t_ground_seq_runs = result[:, :, 3]
    regret_t_pro_seq_runs = result[:, :, 4]
    regret_t_full_seq_runs = result[:, :, 5]
    regret_t_unprotect_seq_runs = result[:, :, 6]
    regret_t_ground_seq_runs = result[:, :, 7]
    n_best_t_pro_seq_runs = result[:, :, 8]
    n_best_t_full_seq_runs = result[:, :, 9]
    n_best_t_unprotect_seq_runs =result[:, :, 10]
    n_best_t_ground_seq_runs =result[:, :, 11]
    gender_pro_seq_runs = result[:, :, 12:18]
    gender_full_seq_runs = result[:, :, 18:24]
    gender_unprotect_seq_runs = result[:, :, 24:30]
    gender_ground_seq_runs = result[:, :, 30:]

    n_runtime = sumr_t_full_seq_runs.shape[0]

    sumr_full = np.sum(sumr_t_full_seq_runs, axis=0)/n_runtime
    sumr_pro = np.sum(sumr_t_pro_seq_runs, axis=0)/n_runtime
    sumr_unprotect = np.sum(sumr_t_unprotect_seq_runs, axis=0)/n_runtime
    sumr_ground = np.sum(sumr_t_ground_seq_runs, axis=0)/n_runtime

    regret_pro = np.sum(regret_t_pro_seq_runs, axis=0)/n_runtime
    regret_full = np.sum(regret_t_full_seq_runs, axis=0)/n_runtime
    regret_unprotect = np.sum(regret_t_unprotect_seq_runs, axis=0)/n_runtime
    regret_ground = np.sum(regret_t_ground_seq_runs, axis=0)/n_runtime

    n_best_pro = np.sum(n_best_t_pro_seq_runs, axis=0)/n_runtime
    n_best_full = np.sum(n_best_t_full_seq_runs, axis=0)/n_runtime
    n_best_unprotect = np.sum(n_best_t_unprotect_seq_runs, axis=0)/n_runtime
    n_best_ground = np.sum(n_best_t_ground_seq_runs, axis=0)/n_runtime


    female_whole_pro = np.sum(gender_pro_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_pro = np.sum(gender_pro_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_full = np.sum(gender_full_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_full = np.sum(gender_full_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_unprotect = np.sum(gender_unprotect_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_unprotect = np.sum(gender_unprotect_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_ground = np.sum(gender_ground_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_ground = np.sum(gender_ground_seq_runs[:,:,3],axis=0)/n_runtime


    plt.figure(1)
    plt.title('reward')
    plt.plot(sumr_full, 'r-', sumr_pro, 'g-', sumr_unprotect, 'b-', sumr_ground, 'k-')
    plt.savefig('./reward_wine.pdf', bbox_inches='tight')
    plt.figure(2)
    plt.title('regret')
    plt.plot(regret_full, 'r-', regret_pro, 'g-', regret_unprotect, 'b-', regret_ground, 'k-')
    plt.savefig('./regret_wine.pdf', bbox_inches='tight')
    plt.figure(3)
    plt.title('n_best')
    plt.plot(n_best_full, 'r-', n_best_pro, 'g-', n_best_unprotect, 'b-', n_best_ground, 'k-')
    plt.show()
    plt.savefig('./n_best_wine.pdf', bbox_inches='tight')


    plt.figure(4)
    plt.title('discrimation')
    plt.plot(
             female_whole_full, 'r-', male_whole_full, 'r-.',
             female_whole_pro, 'g-', male_whole_pro, 'g-.',
             female_whole_unprotect, 'b-', male_whole_unprotect, 'b-.',
             female_whole_ground, 'k-', male_whole_ground, 'k-.')
    plt.show()
    plt.savefig('./discrimation_wine.pdf', bbox_inches='tight')

    plt.figure(5)
    n_groups = 4
    female_n = (female_whole_pro[-1], female_whole_full[-1], female_whole_unprotect[-1], female_whole_ground[-1])
    male_n = (male_whole_pro[-1], male_whole_full[-1], male_whole_unprotect[-1], male_whole_ground[-1])
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, female_n, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Female')

    rects2 = plt.bar(index + bar_width, male_n, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Male')
    plt.xlabel('method')
    plt.ylabel('n_best')
    plt.title('full sequence')
    plt.xticks(index + bar_width, ('pro', 'full', 'unprotect', 'ground'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('./full_sequence_wine.pdf', bbox_inches='tight')

    plt.figure(6)
    n_groups = 4
    female_n = (female_whole_pro[-1]-female_whole_pro[n_trial//(2*recording_time)+1],
                female_whole_full[-1]-female_whole_full[n_trial//(2*recording_time)+1],
                female_whole_unprotect[-1]-female_whole_unprotect[n_trial//(2*recording_time)+1],
                female_whole_ground[-1]-female_whole_ground[n_trial//(2*recording_time)+1])
    male_n = (male_whole_pro[-1]-male_whole_pro[n_trial//(2*recording_time)+1],
                male_whole_full[-1]-male_whole_full[n_trial//(2*recording_time)+1],
                male_whole_unprotect[-1]-male_whole_unprotect[n_trial//(2*recording_time)+1],
                male_whole_ground[-1]-male_whole_ground[n_trial//(2*recording_time)+1])
    # create plot
    fig, ax = plt.subplots()
    index = np.arange(n_groups)
    bar_width = 0.35
    opacity = 0.8
    rects1 = plt.bar(index, female_n, bar_width,
                     alpha=opacity,
                     color='b',
                     label='Female')

    rects2 = plt.bar(index + bar_width, male_n, bar_width,
                     alpha=opacity,
                     color='g',
                     label='Male')
    plt.xlabel('method')
    plt.ylabel('n_best')
    plt.title('full sequence')
    plt.xticks(index + bar_width, ('pro', 'full', 'unprotect', 'ground'))
    plt.legend()

    plt.tight_layout()
    plt.show()
    plt.savefig('./lasthalf_wine.pdf', bbox_inches='tight')


    print(1)


if __name__ == '__main__':
    main()









