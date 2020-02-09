import numpy as np
from numpy.linalg import inv
from numpy import zeros
from numpy.random import rand
from numpy.random import randn
from numpy import eye
from numpy import argmax
import argparse
from ReadWine import getWine
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
from numpy.linalg import inv
from numpy import zeros
from numpy.random import rand
from numpy.random import randn
from numpy import eye
from numpy import argmax
import time
import matplotlib.pyplot as plt
import scipy.io
import math


def runone(run_id, args, group_cluster, n_unproctect_dim, n_feat_dim, wine, quality, true_quality, typeindicator, Lambda, Projection, functiondic_pro, functiondic_unprotect, functiondic_full, functiondic_ground):
    ooo = 0
    decrease_fun = args.decrease_fun
    initsample_user = group_cluster.apply(lambda x: x.sample(1))

    #projection initial
    X_t_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_pro = np.zeros([1, args.n_clusters])
    V_t_pro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_pro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])

    # full initial
    X_t_full = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_full = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_full = np.zeros([1, args.n_clusters])
    V_t_full = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_full = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_full = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])



    #only unproctect feature initial
    X_t_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect = np.zeros([1, args.n_clusters])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])


    # ground truth initial
    X_t_ground = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_ground = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_ground = np.zeros([1, args.n_clusters])
    V_t_ground = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_ground = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_ground = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_ground_seq = np.zeros([args.n_trial // args.recording_time + 1, args.n_clusters])




    gender_pro_sumr_t = np.zeros([1, 2])
    gender_pro_n_rate = np.zeros([1, 2])
    gender_full_sumr_t = np.zeros([1, 2])
    gender_full_n_rate = np.zeros([1, 2])
    gender_unprotect_sumr_t = np.zeros([1, 2])
    gender_unprotect_n_rate = np.zeros([1, 2])
    gender_ground_sumr_t = np.zeros([1, 2])
    gender_ground_n_rate = np.zeros([1,2])

    gender_pro_true_reward = np.zeros([1, 2])
    gender_full_true_reward = np.zeros([1, 2])
    gender_unprotect_true_reward = np.zeros([1, 2])
    gender_ground_true_reward = np.zeros([1, 2])

    ##  first two column gender_sum_r_t, middle column number, final true sum reward
    gender_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_full_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 6])


    n_best_pro = np.zeros([1, args.n_clusters])
    n_best_full = np.zeros([1, args.n_clusters])
    n_best_unprotect = np.zeros([1, args.n_clusters])
    n_best_ground = np.zeros([1, args.n_clusters])
    regret_true_pro = np.zeros([1, args.n_clusters])
    regret_true_full = np.zeros([1, args.n_clusters])
    regret_true_unprotect = np.zeros([1, args.n_clusters])
    regret_true_ground = np.zeros([1, args.n_clusters])
    reward_true_pro = np.zeros([1, args.n_clusters])
    reward_true_full = np.zeros([1, args.n_clusters])
    reward_true_unprotect = np.zeros([1, args.n_clusters])
    reward_true_ground = np.zeros([1, args.n_clusters])

    n_best_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_ground_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_ground_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_ground_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])


    # initial
    for i in range(args.n_clusters):
        cluster_id = i
        if len(wine[wine['cluster_id'] == cluster_id]) >= args.n_Arms:
            wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        else:
            wine_t = wine[wine['cluster_id'] == cluster_id]
        # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        rewards_arm = true_quality.loc[wine_t.index]
        rewards_arm = np.asarray(rewards_arm)
        best_reward = argmax(rewards_arm)
        Arms = np.asarray(wine_t.iloc[:, 0:-1]).T
        ### random select one arm, same for all methods
        SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
        SelectArm_id_pro = wine_t.index[SelectArm_pro]


        #### 0 is white, 1 is red
        gender = 0 if wine_t.iloc[SelectArm_pro]['type'] < 0.1 else 1

        ##################   initial projection arms AND  full arms, they are the same######
        X_t_0 = Arms[:, [SelectArm_pro]]
        r_t = quality.iloc[SelectArm_id_pro, 0]
        X_t_pro[:, [i]] = X_t_0
        V_t_pro[:, :, i] = X_t_0 @ X_t_0.T + Lambda * eye(n_feat_dim)
        V_t_inv_pro[:, :, i] = inv(V_t_pro[:, :, i])

        sumrX_pro[:, [i]] = r_t * X_t_0
        sumr_t_pro[:, i] = r_t

        # 0 is female, 1 is male

        # compare if this is the best reward
        r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        if r_t_true == best_reward:
            n_best_pro+=1
        reward_true_pro[:, i] = r_t_true
        regret_true_pro[:, i] = best_reward-r_t_true

        gender_pro_n_rate[:, gender] += 1
        gender_pro_sumr_t[:, gender] += r_t
        gender_pro_true_reward[:, gender] += r_t_true

        #############################   initial unproctect feature arms #########################
        X_t_0_unproctect = Projection@X_t_0
        X_t_unprotect[:, [i]] = X_t_0_unproctect
        V_t_unprotect[:, :, i] = X_t_0_unproctect @ X_t_0_unproctect.T + Lambda * eye(n_feat_dim)
        V_t_inv_unprotect[:, :, i] = inv(V_t_unprotect[:, :, i])
        r_t = quality.iloc[SelectArm_id_pro, 0]
        sumrX_unprotect[:, [i]] = r_t * X_t_0_unproctect
        sumr_t_unprotect[:, i] = r_t

        # compare if this is the best reward
        r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        if r_t_true == best_reward:
            n_best_unprotect += 1
        reward_true_unprotect[:, i] = r_t_true
        regret_true_unprotect[:, i] = best_reward-r_t_true

        gender_unprotect_n_rate[:, gender] += 1
        gender_unprotect_sumr_t[:, gender] += r_t
        gender_unprotect_true_reward[:, gender] += r_t_true

        #############################   initial ground feature arms #########################
        X_t_0_ground = Projection@X_t_0
        X_t_ground[:, [i]] = X_t_0_ground
        V_t_ground[:, :, i] = X_t_0_ground @ X_t_0_ground.T + Lambda * eye(n_feat_dim)
        V_t_inv_ground[:, :, i] = inv(V_t_ground[:, :, i])
        r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        sumrX_ground[:, [i]] = r_t_true * X_t_0_ground
        sumr_t_ground[:, i] = r_t_true
        r_t = quality.iloc[SelectArm_id_pro, 0]

        # compare if this is the best reward
        if r_t_true == best_reward:
            n_best_ground += 1
        reward_true_ground[:, i] = r_t_true
        regret_true_ground[:, i] = best_reward-r_t_true

        gender_ground_n_rate[:, gender] += 1
        gender_ground_sumr_t[:, gender] += r_t
        gender_ground_true_reward[:, gender] += r_t_true


    ####################    initial full arms, they are the same as pro#######################
    X_t_full = X_t_pro.copy()
    V_t_full = V_t_pro.copy()
    V_t_inv_full = V_t_inv_pro.copy()
    sumrX_full = sumrX_pro.copy()
    sumr_t_full = sumr_t_pro.copy()
    ##
    gender_full_n_rate = gender_pro_n_rate.copy()
    gender_full_sumr_t = gender_pro_sumr_t.copy()
    gender_full_true_reward =  gender_pro_true_reward.copy()
    ##
    n_best_full = n_best_pro.copy()
    reward_true_full = reward_true_pro.copy()
    regret_true_full = regret_true_pro.copy()

    for rtime in range(args.n_trial):
        cluster_id = wine.sample(1).iloc[0, -1]
        if len(wine[wine['cluster_id'] == cluster_id]) >= args.n_Arms:
            wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        else:
            wine_t = wine[wine['cluster_id'] == cluster_id]
        # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        rewards_arm = true_quality.loc[wine_t.index]
        rewards_arm = np.asarray(rewards_arm)
        best_reward = argmax(rewards_arm)
        Arms = np.asarray(wine_t.iloc[:, 0:-1]).T

        ###############################  Projection  ##############################
        # ProArms = Projection @ Arms
        EstTheta_pro[:, [cluster_id]] = V_t_inv_pro[:, :, cluster_id] @ sumrX_pro[:, [cluster_id]]
        if rand(1) > functiondic_pro[decrease_fun](rtime + 1):
            SelectArm_pro = argmax(Arms.T @ (Projection@EstTheta_pro[:, [cluster_id]]))
        else:
            SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
        SelectArm_id_pro = wine_t.index[SelectArm_pro]
        X_t_pro = Arms[:, [SelectArm_pro]]
        r_t_pro = quality.iloc[SelectArm_id_pro, 0]
        sumr_t_pro[:, cluster_id] = sumr_t_pro[:, cluster_id] + r_t_pro

        gender = 0 if wine_t.iloc[SelectArm_pro]['type'] < 0.1 else 1

        V_t_pro[:, :, cluster_id] = V_t_pro[:, :, cluster_id] + X_t_pro @ X_t_pro.T
        V_t_inv_ = V_t_inv_pro[:, :, cluster_id]
        V_t_inv_pro[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_pro @ X_t_pro.T @ V_t_inv_ / (1 + X_t_pro.T @ V_t_inv_ @ X_t_pro)
        sumrX_pro[:, [cluster_id]] += r_t_pro * X_t_pro
        # compare if this is the best reward
        r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        if r_t_true == best_reward:
            n_best_pro += 1
        reward_true_pro[:, cluster_id] += r_t_true
        regret_true_pro[:, cluster_id] += best_reward - r_t_true

        gender_pro_n_rate[:, gender] += 1
        gender_pro_true_reward[:, gender] += r_t_true
        gender_pro_sumr_t[:, gender] += r_t_pro

        #####################################  Full   ###########################################
        EstTheta_full[:, [cluster_id]] = V_t_inv_full[:, :, cluster_id] @ sumrX_full[:, [cluster_id]]
        if rand(1) > functiondic_full[decrease_fun](rtime + 1):
            SelectArm_full = argmax(Arms.T @ EstTheta_full[:, [cluster_id]])
        else:
            SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]

        SelectArm_id_full = wine_t.index[SelectArm_full]
        X_t_full = Arms[:, [SelectArm_full]]
        r_t_full = quality.iloc[SelectArm_id_full, 0]
        sumr_t_full[:, cluster_id] = sumr_t_full[:, cluster_id] + r_t_full

        #
        gender = 0 if wine_t.iloc[SelectArm_full]['type'] < 0.1 else 1


        V_t_full[:, :, cluster_id] = V_t_full[:, :, cluster_id] + X_t_full @ X_t_full.T
        V_t_inv_ = V_t_inv_full[:, :, cluster_id]
        V_t_inv_full[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_full @ X_t_full.T @ V_t_inv_ / (
                    1 + X_t_full.T @ V_t_inv_ @ X_t_full)
        sumrX_full[:, [cluster_id]] += r_t_full * X_t_full
        # compare if this is the best reward
        r_t_true = true_quality.iloc[SelectArm_id_full, 0]
        if r_t_true == best_reward:
            n_best_full += 1
        reward_true_full[:, cluster_id] += r_t_true
        regret_true_full[:, cluster_id] += best_reward - r_t_true

        gender_full_n_rate[:, gender] += 1
        gender_full_sumr_t[:, gender] += r_t_full
        gender_full_true_reward[:, gender] += r_t_true

        ################################### Only unproctect #############################################
        # Arms_movieonly = infor_movie.T
        EstTheta_unprotect[:, [cluster_id]] = V_t_inv_unprotect[:, :, cluster_id] @ sumrX_unprotect[:, [cluster_id]]
        if rand(1) > functiondic_unprotect[decrease_fun](rtime + 1):
            SelectArm_unprotect = argmax(Arms.T @ (Projection@EstTheta_unprotect[:, [cluster_id]]))
        else:
            SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]

        SelectArm_id_unprotect = wine_t.index[SelectArm_unprotect]
        X_t_unprotect = Projection@Arms[:, [SelectArm_unprotect]]
        r_t_unprotect = quality.iloc[SelectArm_id_unprotect, 0]
        sumr_t_unprotect[:, cluster_id] = sumr_t_unprotect[:, cluster_id] + r_t_unprotect

        gender = 0 if wine_t.iloc[SelectArm_unprotect]['type'] < 0.1 else 1


        V_t_unprotect[:, :, cluster_id] = V_t_unprotect[:, :, cluster_id] + X_t_unprotect @ X_t_unprotect.T
        V_t_inv_ = V_t_inv_unprotect[:, :, cluster_id]
        V_t_inv_unprotect[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_unprotect @ X_t_unprotect.T @ V_t_inv_ / (
                    1 + X_t_unprotect.T @ V_t_inv_ @ X_t_unprotect)
        sumrX_unprotect[:, [cluster_id]] += r_t_unprotect * X_t_unprotect
        # compare if this is the best reward
        r_t_true = true_quality.iloc[SelectArm_id_unprotect, 0]
        if r_t_true == best_reward:
            n_best_unprotect += 1
        reward_true_unprotect[:, i] += r_t_true
        regret_true_unprotect[:, i] += best_reward - r_t_true
        gender_unprotect_n_rate[:, gender] += 1
        gender_unprotect_sumr_t[:, gender] += r_t_unprotect
        gender_unprotect_true_reward[:, gender] += r_t_true



    ###############################    ground truth     ################################
        # Arms_movieonly = infor_movie.T
        EstTheta_ground[:, [cluster_id]] = V_t_inv_ground[:, :, cluster_id] @ sumrX_ground[:, [cluster_id]]
        if rand(1) > functiondic_ground[decrease_fun](rtime + 1):
            SelectArm_ground = argmax(Arms.T @ (Projection@EstTheta_ground[:, [cluster_id]]))
        else:
            SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]

        SelectArm_id_ground = wine_t.index[SelectArm_ground]
        X_t_ground = Projection@Arms[:, [SelectArm_ground]]
        r_t_true = true_quality.iloc[SelectArm_id_ground, 0]
        sumr_t_ground[:, cluster_id] = sumr_t_ground[:, cluster_id] + r_t_true

        gender = 0 if wine_t.iloc[SelectArm_ground]['type'] < 0.1 else 1


        V_t_ground[:, :, cluster_id] = V_t_ground[:, :, cluster_id] + X_t_ground @ X_t_ground.T
        V_t_inv_ = V_t_inv_ground[:, :, cluster_id]
        V_t_inv_ground[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_ground @ X_t_ground.T @ V_t_inv_ / (
                    1 + X_t_ground.T @ V_t_inv_ @ X_t_ground)
        sumrX_ground[:, [cluster_id]] += r_t_true * X_t_ground
        # compare if this is the best reward
        r_t_ground = quality.iloc[SelectArm_id_ground, 0]
        if r_t_true == best_reward:
            n_best_ground += 1
        reward_true_ground[:, i] += r_t_true
        regret_true_ground[:, i] += best_reward - r_t_true
        gender_ground_n_rate[:, gender] += 1
        gender_ground_sumr_t[:, gender] += r_t_ground
        gender_ground_true_reward[:, gender] += r_t_true


    ##################################   store #######################################
        if (rtime + 1) % args.recording_time == 0:
            sumr_t_pro_seq[ooo+1, :] = sumr_t_pro.flatten()
            sumr_t_full_seq[ooo + 1, :] = sumr_t_full.flatten()
            sumr_t_unprotect_seq[ooo + 1, :] = sumr_t_unprotect.flatten()
            sumr_t_ground_seq[ooo + 1, :] = sumr_t_ground.flatten()

            gender_pro_seq[ooo + 1, :] = np.hstack((gender_pro_sumr_t, gender_pro_n_rate, gender_pro_true_reward)).flatten()
            gender_full_seq[ooo + 1, :] = np.hstack((gender_full_sumr_t, gender_full_n_rate, gender_full_true_reward)).flatten()
            gender_unprotect_seq[ooo + 1, :] = np.hstack((gender_unprotect_sumr_t, gender_unprotect_n_rate, gender_unprotect_true_reward)).flatten()
            gender_ground_seq[ooo + 1, :] = np.hstack((gender_ground_sumr_t, gender_ground_n_rate, gender_ground_true_reward)).flatten()

            n_best_pro_seq[ooo + 1, :] = n_best_pro
            n_best_full_seq[ooo + 1, :] = n_best_full
            n_best_unprotect_seq[ooo + 1, :] = n_best_unprotect
            n_best_ground_seq[ooo + 1, :] = n_best_ground

            regret_true_pro_seq[ooo + 1, :] = regret_true_pro
            regret_true_full_seq[ooo + 1, :] = regret_true_full
            regret_true_unprotect_seq[ooo + 1, :] = regret_true_unprotect
            regret_true_ground_seq[ooo + 1, :] = regret_true_ground

            reward_true_pro_seq[ooo + 1, :] = reward_true_pro
            reward_true_full_seq[ooo + 1, :] = reward_true_full
            reward_true_unprotect_seq[ooo + 1, :] = reward_true_unprotect
            reward_true_ground_seq[ooo + 1, :] = reward_true_ground



            ooo += 1
            print(rtime)


    # return np.sum(sumr_t_pro, axis=1)
    return sumr_t_pro_seq, sumr_t_full_seq, sumr_t_unprotect_seq, sumr_t_ground_seq,\
           gender_pro_seq, gender_full_seq, gender_unprotect_seq, gender_ground_seq,\
           n_best_pro_seq, n_best_full_seq, n_best_unprotect_seq, n_best_ground_seq,\
           regret_true_pro_seq, regret_true_full_seq, regret_true_unprotect_seq, regret_true_ground_seq, \
           reward_true_pro_seq, reward_true_full_seq, reward_true_unprotect_seq, reward_true_ground_seq\



def main():
# Training settings


    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=1, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                        help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                        help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=3, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_Arms', type=int, default=100, metavar='N',
                        help='set number of arms (default: 100)')
    parser.add_argument('--n_protect_dim', type=int, default=3, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_unprotect_dim', type=int, default=3, metavar='N',
                        help='set number of arms (default: 100)')

    parser.add_argument('--alpha_pro', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_unprotect', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_full', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_ground', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')

    parser.add_argument('--lambda_pro', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--sigma_noise', type=float, default=0.1, metavar='M',
                        help='set noise variance')

    parser.add_argument('--decrease_fun', type=str, default='f', metavar='M',
                        help='set decreasing function')

    # parser.add_argument('--levels', nargs='+', type=int, help='<Required> Set flag', required=True)
    parser.add_argument('--run_unprotect', action='store_true', default=False,
                        help='if specify, we will run unprotected version in the simulation')
    parser.add_argument('--infinite', action='store_true', default=False,
                        help='if specify, we will generate new arms every time')
    parser.add_argument('--plot', action='store_true', default=False,
                        help='if specify, we will plot')
    parser.add_argument('--rand_pro', action='store_true', default=False,
                        help='if specify, we will plot')
    args = parser.parse_args()
    Lambda = 1


    wine, quality, typeindicator = getWine(args)
    true_quality = quality.copy()
    whileindex = (typeindicator == 0).nonzero()[0]
    redindex = (typeindicator == 1).nonzero()[0]
    sample = np.random.choice(whileindex, math.floor(len(whileindex) * 0.7),
                              replace=False).tolist()
    quality.iloc[sample] = quality.iloc[sample] + 1




    #We currently choosely randomly a movie that has a quality for the specific user each time when choosing it.
    #But we also construct D_k below which may be used.

    # n_protect_dim = len(list(users))-1
    n_protect_dim = 1
    n_unproctect_dim = len(list(wine))-2
    n_feat_dim = n_protect_dim + n_unproctect_dim

    # please define the projection operator here.
    # here we first define it as all user information
    Projection = np.diag(np.hstack([np.ones(n_unproctect_dim), np.zeros(n_protect_dim)]))


    # D_k = np.zeros([n_users, 2])
    D_k = np.zeros([4,2])
    n_Dk = D_k.shape[0]
    epsilon_f_pro = lambda x: min(1, args.alpha_pro*n_Dk/x)
    epsilon_i_pro = lambda x: min(1, args.alpha_pro*n_Dk/x**(1.0/3))
    epsilon_o_pro = lambda x: min(1, args.alpha_pro*n_Dk/x**(1.0/2))
    functiondic_pro = {'i': epsilon_i_pro, 'f': epsilon_f_pro, 'o': epsilon_o_pro}

    epsilon_f_unprotect = lambda x: min(1, args.alpha_unprotect*n_Dk/x)
    epsilon_i_unprotect = lambda x: min(1, args.alpha_unprotect*n_Dk/x**(1.0/3))
    epsilon_o_unprotect = lambda x: min(1, args.alpha_unprotect*n_Dk/x**(1.0/2))
    functiondic_unprotect = {'i': epsilon_i_unprotect, 'f': epsilon_f_unprotect, 'o': epsilon_o_unprotect}

    epsilon_f_full = lambda x: min(1, args.alpha_full*n_Dk/x)
    epsilon_i_full = lambda x: min(1, args.alpha_full*n_Dk/x**(1.0/3))
    epsilon_o_full = lambda x: min(1, args.alpha_full*n_Dk/x**(1.0/2))
    functiondic_full = {'i': epsilon_i_full, 'f': epsilon_f_full, 'o': epsilon_o_full}


    epsilon_f_ground = lambda x: min(1, args.alpha_ground * n_Dk / x)
    epsilon_i_ground = lambda x: min(1, args.alpha_ground * n_Dk / x ** (1.0 / 3))
    epsilon_o_ground = lambda x: min(1, args.alpha_ground * n_Dk / x ** (1.0 / 2))
    functiondic_ground = {'i': epsilon_i_ground, 'f': epsilon_f_ground, 'o': epsilon_o_ground}

# for i in users.index:
    #     D_k[i-1, 0] = i
    #     D_k[i-1, 1] = quality.loc[i].sample(1).index[0]

    #to speed up the simulation, we can run different cluster at the same time
    #First get the size for each cluster, sorry, may be implement this later.
    group_cluster = wine.groupby(by=['cluster_id'])
    clusterasize = np.asarray(group_cluster.count().iloc[:, 0])


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if there is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters
    sumr_t_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    sumr_t_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    sumr_t_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    sumr_t_ground_seq_runs = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, args.n_clusters])

    n_best_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_ground_seq_runs = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, args.n_clusters])

    regret_true_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_ground_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])

    reward_true_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_ground_seq_runs = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, args.n_clusters])

    gender_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 6])
    gender_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 6])
    gender_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 6])
    gender_ground_seq_runs = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, 6])

    starttime = time.time()
    for run_id in range(args.runtimes):
        sumr_t_pro_seq_runs[run_id, :, :], sumr_t_full_seq_runs[run_id, :, :],sumr_t_unprotect_seq_runs[run_id, :, :],sumr_t_ground_seq_runs[run_id, :, :],\
        gender_pro_seq_runs[run_id, :, :], gender_full_seq_runs[run_id, :, :], gender_unprotect_seq_runs[run_id, :, :],gender_ground_seq_runs[run_id, :, :], \
        n_best_pro_seq_runs[run_id, :, :], n_best_full_seq_runs[run_id, :, :], n_best_unprotect_seq_runs[run_id, :, :],n_best_ground_seq_runs[run_id, :, :], \
        regret_true_pro_seq_runs[run_id, :, :], regret_true_full_seq_runs[run_id, :, :], regret_true_unprotect_seq_runs[run_id, :, :],regret_true_ground_seq_runs[run_id, :, :], \
        reward_true_pro_seq_runs[run_id, :, :], reward_true_full_seq_runs[run_id, :, :], reward_true_unprotect_seq_runs[run_id, :, :],reward_true_ground_seq_runs[run_id, :, :], \
         = runone(run_id, args, group_cluster, n_unproctect_dim, n_feat_dim, wine, quality, true_quality, typeindicator,
           Lambda, Projection, functiondic_pro, functiondic_unprotect, functiondic_full, functiondic_ground)
    print('That took {} seconds'.format(time.time() - starttime))

    r_full = np.sum(np.sum(sumr_t_full_seq_runs, axis=2), axis=0)/args.runtimes
    r_pro= np.sum(np.sum(sumr_t_pro_seq_runs, axis=2), axis=0)/args.runtimes
    r_unprotect = np.sum(np.sum(sumr_t_unprotect_seq_runs, axis=2), axis=0)/args.runtimes
    r_ground = np.sum(np.sum(sumr_t_ground_seq_runs, axis=2), axis=0) / args.runtimes

    reward_true_full = np.sum(np.sum(reward_true_full_seq_runs, axis=2), axis=0)/args.runtimes
    reward_true_pro = np.sum(np.sum(reward_true_pro_seq_runs, axis=2), axis=0)/args.runtimes
    reward_true_unprotect = np.sum(np.sum(reward_true_unprotect_seq_runs, axis=2), axis=0)/args.runtimes
    reward_true_ground = np.sum(np.sum(reward_true_ground_seq_runs, axis=2), axis=0) / args.runtimes

    regret_true_full = np.sum(np.sum(regret_true_full_seq_runs, axis=2), axis=0) / args.runtimes
    regret_true_pro = np.sum(np.sum(regret_true_pro_seq_runs, axis=2), axis=0) / args.runtimes
    regret_true_unprotect = np.sum(np.sum(regret_true_unprotect_seq_runs, axis=2), axis=0) / args.runtimes
    regret_true_ground = np.sum(np.sum(regret_true_ground_seq_runs, axis=2), axis=0) / args.runtimes

    n_best_true_full = np.sum(np.sum(n_best_full_seq_runs, axis=2), axis=0) / args.runtimes
    n_best_true_pro = np.sum(np.sum(n_best_pro_seq_runs, axis=2), axis=0) / args.runtimes
    n_best_true_unprotect = np.sum(np.sum(n_best_unprotect_seq_runs, axis=2), axis=0) / args.runtimes
    n_best_true_ground = np.sum(np.sum(n_best_ground_seq_runs, axis=2), axis=0) / args.runtimes

    temp_full_fake = gender_full_seq_runs[:, 1:, [0,1]]/gender_full_seq_runs[:, 1:, [2,3]]
    female_full_fake = np.sum(temp_full_fake[:,:,0], axis=0)/args.runtimes
    male_full_fake = np.sum(temp_full_fake[:,:,1], axis=0)/args.runtimes


    temp_pro_fake = gender_pro_seq_runs[:, 1:, [0, 1]] / gender_pro_seq_runs[:, 1:, [2, 3]]
    female_pro_fake = np.sum(temp_pro_fake[:, :, 0], axis=0) / args.runtimes
    male_pro_fake = np.sum(temp_pro_fake[:, :, 1], axis=0) / args.runtimes

    temp_unprotect_fake = gender_unprotect_seq_runs[:, 1:, [0, 1]] / gender_unprotect_seq_runs[:, 1:, [2, 3]]
    female_unprotect_fake = np.sum(temp_unprotect_fake[:, :, 0], axis=0) / args.runtimes
    male_unprotect_fake = np.sum(temp_unprotect_fake[:, :, 1], axis=0) / args.runtimes

    temp_ground_fake = gender_ground_seq_runs[:, 1:, [0, 1]] / gender_ground_seq_runs[:, 1:, [2, 3]]
    female_ground_fake = np.sum(temp_ground_fake[:, :, 0], axis=0) / args.runtimes
    male_ground_fake = np.sum(temp_ground_fake[:, :, 1], axis=0) / args.runtimes


    temp_full_true = gender_full_seq_runs[:, 1:, [4, 5]] / gender_full_seq_runs[:, 1:, [2, 3]]
    female_full_true = np.sum(temp_full_true[:, :, 0], axis=0) / args.runtimes
    male_full_true = np.sum(temp_full_true[:, :, 1], axis=0) / args.runtimes

    temp_pro_true = gender_pro_seq_runs[:, 1:, [4, 5]] / gender_pro_seq_runs[:, 1:, [2, 3]]
    female_pro_true = np.sum(temp_pro_true[:, :, 0], axis=0) / args.runtimes
    male_pro_true = np.sum(temp_pro_true[:, :, 1], axis=0) / args.runtimes

    temp_unprotect_true = gender_unprotect_seq_runs[:, 1:, [4, 5]] / gender_unprotect_seq_runs[:, 1:, [2, 3]]
    female_unprotect_true = np.sum(temp_unprotect_true[:, :, 0], axis=0) / args.runtimes
    male_unprotect_true = np.sum(temp_unprotect_true[:, :, 1], axis=0) / args.runtimes

    temp_ground_true = gender_ground_seq_runs[:, 1:, [4, 5]] / gender_ground_seq_runs[:, 1:, [2, 3]]
    female_ground_true = np.sum(temp_ground_true[:, :, 0], axis=0) / args.runtimes
    male_ground_true = np.sum(temp_ground_true[:, :, 1], axis=0) / args.runtimes


    ###count number from each type(gender)


    scipy.io.savemat('./result/wine/result_matlab_real.mat',
                     mdict={'r_full': r_full,
                            'r_pro': r_pro,
                            'r_unprotect': r_unprotect,
                            'r_ground': r_ground,
                'female_full_fake': female_full_fake,
                'male_full_fake': male_full_fake,
                'female_pro_fake': female_pro_fake,
                'male_pro_fake': male_pro_fake,
                'female_unprotect_fake': female_unprotect_fake,
                'male_unprotect_fake': male_unprotect_fake,
                'female_ground_fake' : female_ground_fake,
                'male_ground_fake' : male_ground_fake,
                'female_full_true': female_full_true,
                'male_full_true': male_full_true,
                'female_pro_true': female_pro_true,
                'male_pro_true': male_pro_true,
                'female_unprotect_true': female_unprotect_true,
                'male_unprotect_true': male_unprotect_true,
                'female_ground_true' : female_ground_true,
                'male_ground_true' : male_ground_true,


                'reward_true_full':reward_true_full,
                'reward_true_pro':reward_true_pro,
                'reward_true_unprotect':reward_true_unprotect,
                'reward_true_ground': reward_true_ground,

                'regret_true_full':regret_true_full,
                'regret_true_pro':regret_true_pro,
                'regret_true_unprotect':regret_true_unprotect,
                'regret_true_ground' : regret_true_ground,

                'n_best_true_full':n_best_true_full,
                'n_best_true_pro':n_best_true_pro,
                'n_best_true_unprotect':n_best_true_unprotect,
                'n_best_true_ground': n_best_true_ground,
                })
    scipy.io.savemat('./result/wine/result_matlab_full_real.mat',\
                     mdict={'sumr_t_pro_seq_runs': sumr_t_pro_seq_runs,
                            'sumr_t_full_seq_runs': sumr_t_full_seq_runs,
                            'sumr_t_unprotect_seq_runs': sumr_t_unprotect_seq_runs,
                            'sumr_t_ground_seq_runs':sumr_t_ground_seq_runs,
                            'gender_pro_seq_runs': gender_pro_seq_runs,
                            'gender_full_seq_runs': gender_full_seq_runs,
                            'gender_unprotect_seq_runs': gender_unprotect_seq_runs,
                            'gender_ground_seq_runs': gender_ground_seq_runs,
                            'reward_true_pro_seq_runs': reward_true_pro_seq_runs,
                            'reward_true_full_seq_runs': reward_true_full_seq_runs,
                            'reward_true_unprotect_seq_runs': reward_true_unprotect_seq_runs,
                            'reward_true_ground_seq_runs': reward_true_ground_seq_runs,
                            'regret_true_pro_seq_runs': regret_true_pro_seq_runs,
                            'regret_true_full_seq_runs': regret_true_full_seq_runs,
                            'regret_true_unprotect_seq_runs': reward_true_unprotect_seq_runs,
                            'regret_true_ground_seq_runs': regret_true_ground_seq_runs,
                            'n_best_pro_seq_runs': n_best_pro_seq_runs,
                            'n_best_full_seq_runs': n_best_full_seq_runs,
                            'n_best_unprotect_seq_runs': n_best_unprotect_seq_runs,
                            'n_best_ground_seq_runs': n_best_ground_seq_runs

                            })
    np.save('./result/wine/sumr_t_pro_seq_runs.npy', sumr_t_pro_seq_runs)
    np.save('./result/wine/sumr_t_full_seq_runs.npy', sumr_t_full_seq_runs)
    np.save('./result/wine/sumr_t_unprotect_seq_runs.npy', sumr_t_unprotect_seq_runs)
    np.save('./result/wine/sumr_t_ground_seq_runs.npy', sumr_t_ground_seq_runs)

    np.save('./result/wine/gender_pro_seq_runs.npy', gender_pro_seq_runs)
    np.save('./result/wine/gender_full_seq_runs.npy', gender_full_seq_runs)
    np.save('./result/wine/gender_unprotect_seq_runs.npy', gender_unprotect_seq_runs)
    np.save('./result/wine/gender_ground_seq_runs.npy', gender_ground_seq_runs)

    np.save('./result/wine/reward_true_pro_seq_runs', reward_true_pro_seq_runs)
    np.save('./result/wine/reward_true_full_seq_runs', reward_true_full_seq_runs)
    np.save('./result/wine/regret_true_unprotect_seq_runs', reward_true_unprotect_seq_runs)
    np.save('./result/wine/regret_true_ground_seq_runs', reward_true_ground_seq_runs)

    np.save('./result/wine/regret_true_pro_seq_runs', regret_true_pro_seq_runs)
    np.save('./result/wine/regret_true_full_seq_runs', regret_true_full_seq_runs)
    np.save('./result/wine/regret_true_unprotect_seq_runs', regret_true_unprotect_seq_runs)
    np.save('./result/wine/regret_true_ground_seq_runs', regret_true_ground_seq_runs)

    np.save('./result/wine/n_best_full_seq_runs', n_best_full_seq_runs)
    np.save('./result/wine/n_best_pro_seq_runs', n_best_pro_seq_runs)
    np.save('./result/wine/n_best_unprotect_seq_runs', n_best_unprotect_seq_runs)
    np.save('./result/wine/n_best_ground_seq_runs', n_best_ground_seq_runs)


    if args.plot:
        plt.figure(1)
        plt.title('quality')
        plt.plot(r_full, 'r-', r_pro, 'g--', r_unprotect, 'b-.', r_ground, 'k-')
        plt.figure(2)
        plt.title('type_difference_fake')
        plt.plot(female_full_fake, 'r-', male_full_fake, 'r-.',
                 female_pro_fake, 'g-', male_pro_fake, 'g-.',
                 female_unprotect_fake, 'b-', male_unprotect_fake, 'b-.',
                 female_ground_fake, 'k-', male_ground_fake, 'b-.')
        plt.figure(3)
        plt.title('type_difference_true')
        plt.plot(female_full_true, 'r-', male_full_true, 'r-.',
                 female_pro_true, 'g-', male_pro_true, 'g-.',
                 female_unprotect_true, 'b-', male_unprotect_true, 'b-.',
                 female_ground_true, 'k-', male_ground_true, 'b-.')

        plt.figure(3)
        plt.title('reward')
        plt.plot(reward_true_full, 'r-', reward_true_pro, 'g--', reward_true_unprotect, 'b-.')
        plt.figure(4)
        plt.title('regret')
        plt.plot(regret_true_full, 'r-', regret_true_pro, 'g--', regret_true_unprotect, 'b-.')
        plt.figure(5)
        plt.title('n_best')
        plt.plot(n_best_true_full, 'r-', n_best_true_pro, 'g--', n_best_true_unprotect, 'b-.')

        plt.show()

    print(1)

if __name__ == '__main__':
    main()



