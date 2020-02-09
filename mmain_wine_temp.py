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
from mkdirr import mkdirp
import multiprocessing
args = 0
wine_o = 0
n_feat_dim = 3
# quality = pd.DataFrame([])
true_quality_o = pd.DataFrame([])
Projection = 0
group_cluster = 0


def runone(run_id):

    # quality = true_quality.copy()
    corruption = 3
    # global wine
    # global true_quality


    samples = []
    for i in range(4, 9):
        a = (true_quality_o['quality'] == i).nonzero()[0].tolist()
        if a:
            sampleids = np.random.choice(a, min(args.n_Arms//5, len(a)), replace=False).tolist()
            samples = samples + sampleids

    # samples = np.random.choice(np.asarray(samples), len(samples), replace=False).tolist()
    wine = wine_o.iloc[samples, :]
    true_quality = true_quality_o.iloc[samples, :]

    num_arms = len(true_quality)
    tempid = np.random.choice(num_arms, math.floor(num_arms/5), replace=False)
    armslastrow = np.zeros([1, num_arms])
    armslastrow[0, tempid] = 1
    # quality.iloc[tempid.tolist()] = quality.iloc[tempid.tolist()] + 1


    epsilon_f_pro = lambda x: min(1, args.alpha_pro*args.n_Dk/x)
    epsilon_i_pro = lambda x: min(1, args.alpha_pro*args.n_Dk/x**(1.0/3))
    epsilon_o_pro = lambda x: min(1, args.alpha_pro*args.n_Dk/x**(1.0/2))
    functiondic_pro = {'i': epsilon_i_pro, 'f': epsilon_f_pro, 'o': epsilon_o_pro}

    epsilon_f_unprotect = lambda x: min(1, args.alpha_unprotect*args.n_Dk/x)
    epsilon_i_unprotect = lambda x: min(1, args.alpha_unprotect*args.n_Dk/x**(1.0/3))
    epsilon_o_unprotect = lambda x: min(1, args.alpha_unprotect*args.n_Dk/x**(1.0/2))
    functiondic_unprotect = {'i': epsilon_i_unprotect, 'f': epsilon_f_unprotect, 'o': epsilon_o_unprotect}

    epsilon_f_full = lambda x: min(1, args.alpha_full*args.n_Dk/x)
    epsilon_i_full = lambda x: min(1, args.alpha_full*args.n_Dk/x**(1.0/3))
    epsilon_o_full = lambda x: min(1, args.alpha_full*args.n_Dk/x**(1.0/2))
    functiondic_full = {'i': epsilon_i_full, 'f': epsilon_f_full, 'o': epsilon_o_full}

    epsilon_f_ground = lambda x: min(1, args.alpha_ground*args.n_Dk/x)
    epsilon_i_ground = lambda x: min(1, args.alpha_ground*args.n_Dk/x**(1.0/3))
    epsilon_o_ground = lambda x: min(1, args.alpha_ground*args.n_Dk/x**(1.0/2))
    functiondic_ground = {'i': epsilon_i_ground, 'f': epsilon_f_ground, 'o': epsilon_o_ground}

    print(run_id)
    ooo = 0
    decrease_fun = args.decrease_fun
    # initsample_user = group_cluster.apply(lambda x: x.sample(1))

    #projection initial
    # X_t_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_pro = np.zeros([1, args.n_clusters])
    V_t_pro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_pro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_pro = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    # full initial
    # X_t_full = np.zeros([n_feat_dim, 1])
    sumrX_full = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_full = np.zeros([1, args.n_clusters])
    V_t_full = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_full = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_full = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])



    #only moive initial
    # X_t_unprotect = np.zeros([n_feat_dim, 1])
    sumrX_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect = np.zeros([1, args.n_clusters])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    # X_t_ground = np.zeros([n_feat_dim, 1])
    sumrX_ground = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_ground = np.zeros([1, args.n_clusters])
    V_t_ground = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_ground = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_ground = np.zeros([n_feat_dim, args.n_clusters])
    # sumr_t_ground_seq = np.zeros([args.n_trial // args.recording_time + 1, args.n_clusters])
    sumr_t_ground_seq = np.zeros([args.n_trial // args.recording_time + 1, 1])

    sumrX_linucb = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_linucb = np.zeros([1, args.n_clusters])
    V_t_linucb = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_linucb = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_linucb = np.zeros([n_feat_dim, args.n_clusters])
    # sumr_t_ground_seq = np.zeros([args.n_trial // args.recording_time + 1, args.n_clusters])
    sumr_t_linucb_seq = np.zeros([args.n_trial // args.recording_time + 1, 1])

    sumrX_linucbpro = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_linucbpro = np.zeros([1, args.n_clusters])
    V_t_linucbpro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_linucbpro = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_linucbpro = np.zeros([n_feat_dim, args.n_clusters])
    # sumr_t_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, args.n_clusters])
    sumr_t_linucbpro_seq = np.zeros([args.n_trial // args.recording_time + 1, 1])

    gender_pro_sumr_t = np.zeros([1, 2])
    gender_pro_n_rate = np.zeros([1, 2])
    gender_full_sumr_t = np.zeros([1, 2])
    gender_full_n_rate = np.zeros([1, 2])
    gender_unprotect_sumr_t = np.zeros([1, 2])
    gender_unprotect_n_rate = np.zeros([1, 2])
    gender_ground_sumr_t = np.zeros([1, 2])
    gender_ground_n_rate = np.zeros([1, 2])
    gender_linucb_sumr_t = np.zeros([1, 2])
    gender_linucb_n_rate = np.zeros([1, 2])
    gender_linucbpro_sumr_t = np.zeros([1, 2])
    gender_linucbpro_n_rate = np.zeros([1, 2])

    gender_pro_true_reward = np.zeros([1, 2])
    gender_full_true_reward = np.zeros([1, 2])
    gender_unprotect_true_reward = np.zeros([1, 2])
    gender_ground_true_reward = np.zeros([1, 2])
    gender_linucb_true_reward = np.zeros([1, 2])
    gender_linucbpro_true_reward = np.zeros([1, 2])

    ##  first two column gender_sum_r_t, middle column number, final true sum reward
    gender_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_full_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 6])

    n_best_pro = np.zeros([1])
    n_best_full = np.zeros([1])
    n_best_unprotect = np.zeros([1])
    n_best_ground = np.zeros([1])
    n_best_linucb = np.zeros([1])
    n_best_linucbpro = np.zeros([1])

    regret_true_pro = np.zeros([1, args.n_clusters])
    regret_true_full = np.zeros([1, args.n_clusters])
    regret_true_unprotect = np.zeros([1, args.n_clusters])
    regret_true_ground = np.zeros([1, args.n_clusters])
    regret_true_linucb = np.zeros([1, args.n_clusters])
    regret_true_linucbpro = np.zeros([1, args.n_clusters])

    reward_true_pro = np.zeros([1, args.n_clusters])
    reward_true_full = np.zeros([1, args.n_clusters])
    reward_true_unprotect = np.zeros([1, args.n_clusters])
    reward_true_ground = np.zeros([1, args.n_clusters])
    reward_true_linucb = np.zeros([1, args.n_clusters])
    reward_true_linucbpro = np.zeros([1, args.n_clusters])

    n_best_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    regret_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_true_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_true_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_true_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    reward_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    reward_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    reward_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    reward_true_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    reward_true_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    reward_true_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    cluster = np.ones((num_arms,args.n_clusters), dtype=bool)
    tempwine = []
    templastrow = []


    # initial
    for i in range(args.n_clusters):
        cluster_id = i
        cluster[:, cluster_id] = wine['cluster_id'] == cluster_id
        tempwine.append(wine[cluster[:, cluster_id]])
        templastrow.append(armslastrow[:, cluster[:, cluster_id]])
        lenwine = len(tempwine[cluster_id])
        if lenwine >= args.n_Arms:
            sampleid = np.random.choice(lenwine, args.n_Arms, replace=False)
            wine_t = tempwine[cluster_id].iloc[sampleid.tolist()]
            tlastrow = templastrow[cluster_id][:, sampleid]
        else:
            wine_t = tempwine[cluster_id]
            tlastrow = templastrow[cluster_id]
        # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        rewards_true = true_quality.loc[wine_t.index]
        rewards_true = np.asarray(rewards_true)
        best_reward = np.max(rewards_true)
        Arms = np.asarray(wine_t.iloc[:, 0:-1]).T
        Arms = np.vstack((Arms, tlastrow))
        ## random select one arm, same for all methods
        SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
        # rewards_fake = np.asarray(quality.loc[wine_t.index])


        #### 0 is white, 1 is red
        # gender = 0 if wine_t.iloc[SelectArm_pro]['type'] < 0.1 else 1
        gender = int(tlastrow[:, SelectArm_pro])
        ##################   initial projection arms AND  full arms, they are the same######
        X_t_0 = Arms[:, [SelectArm_pro]]
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        # r_t_pro = rewards_true[SelectArm_pro, 0]
        # X_t_pro[:, [i]] = X_t_0
        V_t_pro[:, :, i] = X_t_0 @ X_t_0.T + args.lambda_pro * eye(n_feat_dim)
        V_t_inv_pro[:, :, i] = inv(V_t_pro[:, :, i])

        sumrX_pro[:, [i]] = r_t * X_t_0
        sumr_t_pro[:, i] = r_t

        # 0 is female, 1 is male

        # compare if this is the best reward
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
        if r_t_true == best_reward:
            n_best_pro+=1
        reward_true_pro[:, i] = r_t_true
        regret_true_pro[:, i] = best_reward-r_t_true

        gender_pro_n_rate[:, gender] += 1
        gender_pro_sumr_t[:, gender] += r_t
        gender_pro_true_reward[:, gender] += r_t_true

        #############################   initial unproctect feature arms #########################
        X_t_0_unproctect = Projection@X_t_0
        # X_t_unprotect[:, [i]] = X_t_0_unproctect
        V_t_unprotect[:, :, i] = X_t_0_unproctect @ X_t_0_unproctect.T + args.lambda_pro * eye(n_feat_dim)
        V_t_inv_unprotect[:, :, i] = inv(V_t_unprotect[:, :, i])
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        # r_t = rewards_fake[SelectArm_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        sumrX_unprotect[:, [i]] = r_t * X_t_0_unproctect
        sumr_t_unprotect[:, i] = r_t

        # compare if this is the best reward
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
        if r_t_true == best_reward:
            n_best_unprotect += 1
        reward_true_unprotect[:, i] = r_t_true
        regret_true_unprotect[:, i] = best_reward-r_t_true

        gender_unprotect_n_rate[:, gender] += 1
        gender_unprotect_sumr_t[:, gender] += r_t
        gender_unprotect_true_reward[:, gender] += r_t_true

        #############################   initial ground feature arms #########################
        X_t_0_ground = Projection@X_t_0
        # X_t_ground[:, [i]] = X_t_0_ground
        V_t_ground[:, :, i] = X_t_0_ground @ X_t_0_ground.T + args.lambda_pro * eye(n_feat_dim)
        V_t_inv_ground[:, :, i] = inv(V_t_ground[:, :, i])
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
        sumrX_ground[:, [i]] = r_t_true * X_t_0_ground
        sumr_t_ground[:, i] = r_t_true
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        # r_t = rewards_fake[SelectArm_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        # compare if this is the best reward
        if r_t_true == best_reward:
            n_best_ground += 1
        reward_true_ground[:, i] = r_t_true
        regret_true_ground[:, i] = best_reward-r_t_true

        gender_ground_n_rate[:, gender] += 1
        gender_ground_sumr_t[:, gender] += r_t
        gender_ground_true_reward[:, gender] += r_t_true
        #############################   initial linucb arms #########################
        X_t_0_linucb = X_t_0
        V_t_linucb[:, :, i] = X_t_0_linucb @ X_t_0_linucb.T + args.lambda_pro * eye(n_feat_dim)
        V_t_inv_linucb[:, :, i] = inv(V_t_linucb[:, :, i])
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
        sumrX_linucb[:, [i]] = r_t_true * X_t_0_linucb
        sumr_t_linucb[:, i] = r_t_true
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        # r_t = rewards_fake[SelectArm_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1 - args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        # compare if this is the best reward
        if r_t_true == best_reward:
            n_best_linucb += 1
        reward_true_linucb[:, i] = r_t_true
        regret_true_linucb[:, i] = best_reward - r_t_true

        gender_linucb_n_rate[:, gender] += 1
        gender_linucb_sumr_t[:, gender] += r_t
        gender_linucb_true_reward[:, gender] += r_t_true

        #############################   initial linucbpro arms #########################
        X_t_0_linucbpro = X_t_0
        V_t_linucbpro[:, :, i] = X_t_0_linucbpro @ X_t_0_linucbpro.T + args.lambda_pro * eye(n_feat_dim)
        V_t_inv_linucbpro[:, :, i] = inv(V_t_linucbpro[:, :, i])
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
        sumrX_linucbpro[:, [i]] = r_t_true * X_t_0_linucbpro
        sumr_t_linucbpro[:, i] = r_t_true
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        # r_t = rewards_fake[SelectArm_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1 - args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        # compare if this is the best reward
        if r_t_true == best_reward:
            n_best_linucbpro += 1
        reward_true_linucbpro[:, i] = r_t_true
        regret_true_linucbpro[:, i] = best_reward - r_t_true

        gender_linucbpro_n_rate[:, gender] += 1
        gender_linucbpro_sumr_t[:, gender] += r_t
        gender_linucbpro_true_reward[:, gender] += r_t_true


    ####################    initial full arms, they are the same as pro#######################
    # X_t_full = X_t_pro.copy()
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


    cluster_id = wine.sample(1).iloc[0, -1]
    lenwine = len(tempwine[cluster_id])
    if lenwine >= args.n_Arms:
        sampleid = np.random.choice(lenwine, args.n_Arms, replace=False)
        wine_t = tempwine[cluster_id].iloc[sampleid.tolist()]
        tlastrow = templastrow[cluster_id][:, sampleid]
    else:
        wine_t = tempwine[cluster_id]
        tlastrow = templastrow[cluster_id]
    # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
    rewards_true = true_quality.loc[wine_t.index]
    rewards_true = np.asarray(rewards_true)
    best_reward = np.max(rewards_true)
    Arms = np.asarray(wine_t.iloc[:, 0:-1]).T
    Arms = np.vstack((Arms, tlastrow))

    for rtime in range(args.n_trial):
        # cluster_id = wine.sample(1).iloc[0, -1]
        # lenwine = len(tempwine[cluster_id])
        # if lenwine >= args.n_Arms:
        #     sampleid = np.random.choice(lenwine, args.n_Arms, replace=False)
        #     wine_t = tempwine[cluster_id].iloc[sampleid.tolist()]
        #     tlastrow = templastrow[cluster_id][:, sampleid]
        # else:
        #     wine_t = tempwine[cluster_id]
        #     tlastrow = templastrow[cluster_id]
        # # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        # rewards_true = true_quality.loc[wine_t.index]
        # rewards_true = np.asarray(rewards_true)
        # best_reward = np.max(rewards_true)
        # Arms = np.asarray(wine_t.iloc[:, 0:-1]).T
        # Arms = np.vstack((Arms, tlastrow))
        # rewards_fake = np.asarray(quality.loc[wine_t.index])

        ###############################  Projection  ##############################
        # ProArms = Projection @ Arms
        EstTheta_pro[:, [cluster_id]] = V_t_inv_pro[:, :, cluster_id] @ sumrX_pro[:, [cluster_id]]
        if rand(1) > functiondic_pro[decrease_fun](rtime + 1):
            SelectArm_pro = argmax(Arms.T @ (Projection@EstTheta_pro[:, [cluster_id]]))
        else:
            # SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_pro = np.random.randint((Projection.shape[1]), size=1)[0]

        gender = int(tlastrow[:, SelectArm_pro])

        # SelectArm_id_pro = wine_t.index[SelectArm_pro]
        X_t_pro = Arms[:, [SelectArm_pro]]
        # r_t_pro = quality.iloc[SelectArm_id_pro, 0]
        # r_t_pro = rewards_fake[SelectArm_pro, 0]
        if gender == 1:
            r_t_pro = rewards_true[SelectArm_pro, 0] + corruption*np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t_pro = rewards_true[SelectArm_pro, 0]
        sumr_t_pro[:, cluster_id] = sumr_t_pro[:, cluster_id] + r_t_pro

        # gender = 0 if wine_t.iloc[SelectArm_pro]['type'] < 0.1 else 1

        V_t_pro[:, :, cluster_id] = V_t_pro[:, :, cluster_id] + X_t_pro @ X_t_pro.T
        V_t_inv_ = V_t_inv_pro[:, :, cluster_id]
        V_t_inv_pro[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_pro @ X_t_pro.T @ V_t_inv_ / (1 + X_t_pro.T @ V_t_inv_ @ X_t_pro)
        sumrX_pro[:, [cluster_id]] += r_t_pro * X_t_pro
        # compare if this is the best reward
        # r_t_true = true_quality.iloc[SelectArm_id_pro, 0]
        r_t_true = rewards_true[SelectArm_pro, 0]
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
            # SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_full = np.random.randint((Projection.shape[1]), size=1)[0]

        gender = int(tlastrow[:, SelectArm_full])

        # SelectArm_id_full = wine_t.index[SelectArm_full]
        # r_t_full = rewards_fake[SelectArm_full, 0]
        if gender == 1:
            r_t_full = rewards_true[SelectArm_full, 0] + corruption*np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t_full = rewards_true[SelectArm_full, 0]
        X_t_full = Arms[:, [SelectArm_full]]
        # r_t_full = quality.iloc[SelectArm_id_full, 0]
        sumr_t_full[:, cluster_id] = sumr_t_full[:, cluster_id] + r_t_full

        #
        # gender = 0 if wine_t.iloc[SelectArm_full]['type'] < 0.1 else 1


        V_t_full[:, :, cluster_id] = V_t_full[:, :, cluster_id] + X_t_full @ X_t_full.T
        V_t_inv_ = V_t_inv_full[:, :, cluster_id]
        V_t_inv_full[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_full @ X_t_full.T @ V_t_inv_ / (
                    1 + X_t_full.T @ V_t_inv_ @ X_t_full)
        sumrX_full[:, [cluster_id]] += r_t_full * X_t_full
        # compare if this is the best reward
        # r_t_true = true_quality.iloc[SelectArm_id_full, 0]
        r_t_true = rewards_true[SelectArm_full, 0]
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
            # SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_unprotect = np.random.randint((Projection.shape[1]), size=1)[0]
        gender = int(tlastrow[:, SelectArm_unprotect])
        # SelectArm_id_unprotect = wine_t.index[SelectArm_unprotect]
        X_t_unprotect = Projection@Arms[:, [SelectArm_unprotect]]
        if gender == 1:
            r_t_unprotect = rewards_true[SelectArm_unprotect, 0] + corruption*np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t_unprotect = rewards_true[SelectArm_unprotect, 0]
        # r_t_unprotect = quality.iloc[SelectArm_id_unprotect, 0]
        sumr_t_unprotect[:, cluster_id] = sumr_t_unprotect[:, cluster_id] + r_t_unprotect

        # gender = 0 if wine_t.iloc[SelectArm_unprotect]['type'] < 0.1 else 1

        V_t_unprotect[:, :, cluster_id] = V_t_unprotect[:, :, cluster_id] + X_t_unprotect @ X_t_unprotect.T
        V_t_inv_ = V_t_inv_unprotect[:, :, cluster_id]
        V_t_inv_unprotect[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_unprotect @ X_t_unprotect.T @ V_t_inv_ / (
                    1 + X_t_unprotect.T @ V_t_inv_ @ X_t_unprotect)
        sumrX_unprotect[:, [cluster_id]] += r_t_unprotect * X_t_unprotect
        r_t_true = rewards_true[SelectArm_unprotect, 0]
        # compare if this is the best reward
        # r_t_true = true_quality.iloc[SelectArm_id_unprotect, 0]
        if r_t_true == best_reward:
            n_best_unprotect += 1
        reward_true_unprotect[:, cluster_id] += r_t_true
        regret_true_unprotect[:, cluster_id] += best_reward - r_t_true
        gender_unprotect_n_rate[:, gender] += 1
        gender_unprotect_sumr_t[:, gender] += r_t_unprotect
        gender_unprotect_true_reward[:, gender] += r_t_true



    # ###############################    ground truth     ################################
    #     # Arms_movieonly = infor_movie.T
    #     EstTheta_ground[:, [cluster_id]] = V_t_inv_ground[:, :, cluster_id] @ sumrX_ground[:, [cluster_id]]
    #     if rand(1) > functiondic_ground[decrease_fun](rtime + 1):
    #         SelectArm_ground = argmax(Arms.T @ (Projection@EstTheta_ground[:, [cluster_id]]))
    #     else:
    #         SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]
    #     gender = int(tlastrow[:, SelectArm_ground])
    #     if gender == 1:
    #         r_t_ground = rewards_true[SelectArm_ground, 0] + np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
    #     else:
    #         r_t_ground = rewards_true[SelectArm_ground, 0]
    #
    #     # SelectArm_id_ground = wine_t.index[SelectArm_ground]
    #     X_t_ground = Projection@Arms[:, [SelectArm_ground]]
    #     # r_t_true = true_quality.iloc[SelectArm_id_ground, 0]
    #     r_t_true = rewards_true[SelectArm_ground, 0]
    #     sumr_t_ground[:, cluster_id] = sumr_t_ground[:, cluster_id] + r_t_ground
    #
    #     # gender = 0 if wine_t.iloc[SelectArm_ground]['type'] < 0.1 else 1
    #
    #
    #     V_t_ground[:, :, cluster_id] = V_t_ground[:, :, cluster_id] + X_t_ground @ X_t_ground.T
    #     V_t_inv_ = V_t_inv_ground[:, :, cluster_id]
    #     V_t_inv_ground[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_ground @ X_t_ground.T @ V_t_inv_ / (
    #                 1 + X_t_ground.T @ V_t_inv_ @ X_t_ground)
    #     sumrX_ground[:, [cluster_id]] += r_t_true * X_t_ground
    #     # compare if this is the best reward
    #     # r_t_ground = quality.iloc[SelectArm_id_ground, 0]
    #
    #     if r_t_true == best_reward:
    #         n_best_ground += 1
    #     reward_true_ground[:, cluster_id] += r_t_true
    #     regret_true_ground[:, cluster_id] += best_reward - r_t_true
    #     gender_ground_n_rate[:, gender] += 1
    #     gender_ground_sumr_t[:, gender] += r_t_ground
    #     gender_ground_true_reward[:, gender] += r_t_true

        ###############################    linucb        ################################
        # EstTheta_linucb[:, [cluster_id]] = V_t_inv_linucb[:, :, cluster_id] @ sumrX_linucb[:, [cluster_id]]
        # upperbound = args.alpha_linucb * np.sqrt(np.sum(Arms * (V_t_inv_linucb[:, :, cluster_id] @ Arms), axis=0)).reshape(-1, 1)
        # SelectArm_linucb = argmax(Arms.T @ (EstTheta_linucb[:, [cluster_id]]) + upperbound)
        #
        # gender = int(tlastrow[:, SelectArm_linucb])
        # if gender == 1:
        #     r_t_linucb = rewards_true[SelectArm_linucb, 0] + np.random.choice([0, 1], 1, p=[1 - args.pr, args.pr])
        # else:
        #     r_t_linucb = rewards_true[SelectArm_linucb, 0]
        #
        # X_t_linucb = Arms[:, [SelectArm_linucb]]
        # r_t_true = rewards_true[SelectArm_linucb, 0]
        # sumr_t_linucb[:, cluster_id] = sumr_t_linucb[:, cluster_id] + r_t_linucb
        #
        # V_t_linucb[:, :, cluster_id] = V_t_linucb[:, :, cluster_id] + X_t_linucb @ X_t_linucb.T
        # V_t_inv_ = V_t_inv_linucb[:, :, cluster_id]
        # V_t_inv_linucb[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_linucb @ X_t_linucb.T @ V_t_inv_ / (
        #         1 + X_t_linucb.T @ V_t_inv_ @ X_t_linucb)
        # sumrX_linucb[:, [cluster_id]] += r_t_linucb * X_t_linucb
        #
        # if r_t_true == best_reward:
        #     n_best_linucb += 1
        # reward_true_linucb[:, cluster_id] += r_t_true
        # regret_true_linucb[:, cluster_id] += best_reward - r_t_true
        # gender_linucb_n_rate[:, gender] += 1
        # gender_linucb_sumr_t[:, gender] += r_t_linucb
        # gender_linucb_true_reward[:, gender] += r_t_true

        ###############################    linucbpro        ################################
        EstTheta_linucbpro[:, [cluster_id]] = V_t_inv_linucbpro[:, :, cluster_id] @ sumrX_linucbpro[:, [cluster_id]]
        upperbound = args.alpha_linucbpro * np.sqrt(np.sum(Arms * (V_t_inv_linucbpro[:, :, cluster_id] @ Arms), axis=0)).reshape(-1, 1)
        # SelectArm_linucbpro = argmax(Arms.T @ (Projection @ EstTheta_linucbpro[:, [cluster_id]]) + upperbound)
        SelectArm_linucbpro = argmax(Arms.T @ (EstTheta_linucbpro[:, [cluster_id]]) + upperbound)

        gender = int(tlastrow[:, SelectArm_linucbpro])
        if gender == 1:
            r_t_linucbpro = rewards_true[SelectArm_linucbpro, 0] + corruption*np.random.choice([0, 1], 1, p=[1 - args.pr, args.pr])
        else:
            r_t_linucbpro = rewards_true[SelectArm_linucbpro, 0]

        X_t_linucbpro = Arms[:, [SelectArm_linucbpro]]
        r_t_true = rewards_true[SelectArm_linucbpro, 0]
        sumr_t_linucbpro[:, cluster_id] = sumr_t_linucbpro[:, cluster_id] + r_t_linucbpro

        V_t_linucbpro[:, :, cluster_id] = V_t_linucbpro[:, :, cluster_id] + X_t_linucbpro @ X_t_linucbpro.T
        V_t_inv_ = V_t_inv_linucbpro[:, :, cluster_id]
        V_t_inv_linucbpro[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t_linucbpro @ X_t_linucbpro.T @ V_t_inv_ / (
                1 + X_t_linucbpro.T @ V_t_inv_ @ X_t_linucbpro)
        sumrX_linucbpro[:, [cluster_id]] += r_t_linucbpro * X_t_linucbpro

        if r_t_true == best_reward:
            n_best_linucbpro += 1
        reward_true_linucbpro[:, cluster_id] += r_t_true
        regret_true_linucbpro[:, cluster_id] += best_reward - r_t_true
        gender_linucbpro_n_rate[:, gender] += 1
        gender_linucbpro_sumr_t[:, gender] += r_t_linucbpro
        gender_linucbpro_true_reward[:, gender] += r_t_true
        ##################################   store #######################################
        # if (rtime + 1) % args.recording_time == 0:
        #     sumr_t_pro_seq[ooo+1, :] = sumr_t_pro.flatten()
        #     sumr_t_full_seq[ooo + 1, :] = sumr_t_full.flatten()
        #     sumr_t_unprotect_seq[ooo + 1, :] = sumr_t_unprotect.flatten()
        #     sumr_t_ground_seq[ooo + 1, :] = sumr_t_ground.flatten()
        #
        #     gender_pro_seq[ooo + 1, :] = np.hstack((gender_pro_sumr_t, gender_pro_n_rate, gender_pro_true_reward)).flatten()
        #     gender_full_seq[ooo + 1, :] = np.hstack((gender_full_sumr_t, gender_full_n_rate, gender_full_true_reward)).flatten()
        #     gender_unprotect_seq[ooo + 1, :] = np.hstack((gender_unprotect_sumr_t, gender_unprotect_n_rate, gender_unprotect_true_reward)).flatten()
        #     gender_ground_seq[ooo + 1, :] = np.hstack((gender_ground_sumr_t, gender_ground_n_rate, gender_ground_true_reward)).flatten()
        #
        #     n_best_pro_seq[ooo + 1, :] = n_best_pro
        #     n_best_full_seq[ooo + 1, :] = n_best_full
        #     n_best_unprotect_seq[ooo + 1, :] = n_best_unprotect
        #     n_best_ground_seq[ooo + 1, :] = n_best_ground
        #
        #     regret_true_pro_seq[ooo + 1, :] = regret_true_pro
        #     regret_true_full_seq[ooo + 1, :] = regret_true_full
        #     regret_true_unprotect_seq[ooo + 1, :] = regret_true_unprotect
        #     regret_true_ground_seq[ooo + 1, :] = regret_true_ground
        #
        #     reward_true_pro_seq[ooo + 1, :] = reward_true_pro
        #     reward_true_full_seq[ooo + 1, :] = reward_true_full
        #     reward_true_unprotect_seq[ooo + 1, :] = reward_true_unprotect
        #     reward_true_ground_seq[ooo + 1, :] = reward_true_ground
        #     ooo += 1

        if (rtime + 1) % args.recording_time == 0:
            sumr_t_pro_seq[ooo + 1, :] = np.sum(sumr_t_pro.flatten())
            sumr_t_full_seq[ooo + 1, :] = np.sum(sumr_t_full.flatten())
            sumr_t_unprotect_seq[ooo + 1, :] = np.sum(sumr_t_unprotect.flatten())
            sumr_t_ground_seq[ooo + 1, :] = np.sum(sumr_t_ground.flatten())
            sumr_t_linucb_seq[ooo + 1, :] = np.sum(sumr_t_linucb.flatten())
            sumr_t_linucbpro_seq[ooo + 1, :] = np.sum(sumr_t_linucbpro.flatten())

            gender_pro_seq[ooo + 1, :] = np.hstack(
                (gender_pro_sumr_t, gender_pro_n_rate, gender_pro_true_reward)).flatten()
            gender_full_seq[ooo + 1, :] = np.hstack(
                (gender_full_sumr_t, gender_full_n_rate, gender_full_true_reward)).flatten()
            gender_unprotect_seq[ooo + 1, :] = np.hstack(
                (gender_unprotect_sumr_t, gender_unprotect_n_rate, gender_unprotect_true_reward)).flatten()
            gender_ground_seq[ooo + 1, :] = np.hstack(
                (gender_ground_sumr_t, gender_ground_n_rate, gender_ground_true_reward)).flatten()
            gender_linucb_seq[ooo + 1, :] = np.hstack(
                (gender_linucb_sumr_t, gender_linucb_n_rate, gender_linucb_true_reward)).flatten()
            gender_linucbpro_seq[ooo + 1, :] = np.hstack(
                (gender_linucbpro_sumr_t, gender_linucbpro_n_rate, gender_linucbpro_true_reward)).flatten()

            n_best_pro_seq[ooo + 1, :] = np.sum(n_best_pro)
            n_best_full_seq[ooo + 1, :] = np.sum(n_best_full)
            n_best_unprotect_seq[ooo + 1, :] = np.sum(n_best_unprotect)
            n_best_ground_seq[ooo + 1, :] = np.sum(n_best_ground)
            n_best_linucb_seq[ooo + 1, :] = np.sum(n_best_linucb)
            n_best_linucbpro_seq[ooo + 1, :] = np.sum(n_best_linucbpro)

            regret_true_pro_seq[ooo + 1, :] = np.sum(regret_true_pro)
            regret_true_full_seq[ooo + 1, :] = np.sum(regret_true_full)
            regret_true_unprotect_seq[ooo + 1, :] = np.sum(regret_true_unprotect)
            regret_true_ground_seq[ooo + 1, :] = np.sum(regret_true_ground)
            regret_true_linucb_seq[ooo + 1, :] = np.sum(regret_true_linucb)
            regret_true_linucbpro_seq[ooo + 1, :] = np.sum(regret_true_linucbpro)

            reward_true_pro_seq[ooo + 1, :] = np.sum(reward_true_pro)
            reward_true_full_seq[ooo + 1, :] = np.sum(reward_true_full)
            reward_true_unprotect_seq[ooo + 1, :] = np.sum(reward_true_unprotect)
            reward_true_ground_seq[ooo + 1, :] = np.sum(reward_true_ground)
            reward_true_linucb_seq[ooo + 1, :] = np.sum(reward_true_linucb)
            reward_true_linucbpro_seq[ooo + 1, :] = np.sum(reward_true_linucbpro)
            ooo += 1

    # return np.sum(sumr_t_pro, axis=1)
    return np.hstack((sumr_t_pro_seq,
                      sumr_t_full_seq,
                      sumr_t_unprotect_seq,
                      sumr_t_ground_seq,
                      sumr_t_linucb_seq,
                      sumr_t_linucbpro_seq,

                      regret_true_pro_seq,
                      regret_true_full_seq,
                      regret_true_unprotect_seq,
                      regret_true_ground_seq,
                      regret_true_linucb_seq,
                      regret_true_linucbpro_seq,

                      n_best_pro_seq,
                      n_best_full_seq,
                      n_best_unprotect_seq,
                      n_best_ground_seq,
                      n_best_linucb_seq,
                      n_best_linucbpro_seq,

                      gender_pro_seq,
                      gender_full_seq,
                      gender_unprotect_seq,
                      gender_ground_seq,
                      gender_linucb_seq,
                      gender_linucbpro_seq,

                      reward_true_pro_seq,
                      reward_true_full_seq,
                      reward_true_unprotect_seq,
                      reward_true_ground_seq,
                      reward_true_linucb_seq,
                      reward_true_linucbpro_seq,

                      ))




def main():
# Training settings
    global args
    global args
    global wine_o
    global n_feat_dim
    # global quality
    global true_quality_o
    global Projection
    global group_cluster

    mkdirp('./result/wine')
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=1, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                        help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                        help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=100, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_Arms', type=int, default=100, metavar='N',
                        help='set number of arms (default: 100)')
    parser.add_argument('--n_protect_dim', type=int, default=3, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_unprotect_dim', type=int, default=3, metavar='N',
                        help='set number of arms (default: 100)')
    parser.add_argument('--n_Dk', type=int, default=0, metavar='N',
                    help='set number of arms (default: 100)')


    parser.add_argument('--alpha_pro', type=float, default=1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_unprotect', type=float, default=1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_full', type=float, default=1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_ground', type=float, default=1, metavar='M',
                    help='set parameter alpha')
    parser.add_argument('--alpha_linucb', type=float, default=1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_linucbpro', type=float, default=1, metavar='M',
                        help='set parameter alpha')


    parser.add_argument('--lambda_pro', type=float, default=1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--sigma_noise', type=float, default=1, metavar='M',
                        help='set noise variance')
    parser.add_argument('--pr', type=float, default=1, metavar='M',
                    help='set noise variance')

    parser.add_argument('--decrease_fun', type=str, default='i', metavar='M',
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
    parser.add_argument('--not_mul', action='store_true', default=False,
                    help='if specify, we will plot')

    args = parser.parse_args()
    args.lambda_pro = 1


    wine_o, quality, typeindicator = getWine(args)
    true_quality_o = quality.copy()
    # whileindex = (typeindicator == 0).nonzero()[0]
    # redindex = (typeindicator == 1).nonzero()[0]
    # sample = np.random.choice(whileindex, math.floor(len(whileindex) * 0.7),
    #                           replace=False).tolist()
    # quality.iloc[sample] = quality.iloc[sample] + 1





    #We currently choosely randomly a movie that has a quality for the specific user each time when choosing it.
    #But we also construct D_k below which may be used.

    # n_protect_dim = len(list(users))-1
    n_protect_dim = 1
    n_unproctect_dim = len(list(wine_o))-1
    #### last column is cluster indicator
    n_feat_dim = n_protect_dim + n_unproctect_dim

    # please define the projection operator here.
    # here we first define it as all user information
    Projection = np.diag(np.hstack([np.ones(n_unproctect_dim), np.zeros(n_protect_dim)]))




    # D_k = np.zeros([n_users, 2])
    D_k = np.zeros([n_feat_dim,2])
    args.n_Dk = D_k.shape[0]

    n_clusters = args.n_clusters

# for i in users.index:
    #     D_k[i-1, 0] = i
    #     D_k[i-1, 1] = quality.loc[i].sample(1).index[0]

    #to speed up the simulation, we can run different cluster at the same time
    #First get the size for each cluster, sorry, may be implement this later.
    # group_cluster = wine.groupby(by=['cluster_id'])
    # clusterasize = np.asarray(group_cluster.count().iloc[:, 0])


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if there is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters

    starttime = time.time()



    #
#
    if not args.not_mul:
        with multiprocessing.Pool() as pool:
            result = pool.map(runone, range(0, args.runtimes))
            pool.close()
            pool.join()
            result = np.asarray(result)
    else:
        result = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, 60])
        for run_id in range(args.runtimes):
            result[run_id, :] = runone(run_id)


    sumr_t_pro_seq_runs = result[:, :, 0]
    sumr_t_full_seq_runs = result[:, :, 1]
    sumr_t_unprotect_seq_runs = result[:, :, 2]
    sumr_t_ground_seq_runs = result[:, :, 3]
    sumr_t_linucb_seq_runs = result[:, :, 4]
    sumr_t_linucbpro_seq_runs = result[:, :, 5]

    regret_true_pro_seq_runs = result[:, :, 6]
    regret_true_full_seq_runs = result[:, :, 7]
    regret_true_unprotect_seq_runs = result[:, :, 8]
    regret_true_ground_seq_runs = result[:, :, 9]
    regret_true_linucb_seq_runs = result[:, :, 10]
    regret_true_linucbpro_seq_runs = result[:, :, 11]

    n_best_true_pro_seq_runs = result[:, :, 12]
    n_best_true_full_seq_runs = result[:, :, 13]
    n_best_true_unprotect_seq_runs = result[:, :, 14]
    n_best_true_ground_seq_runs = result[:, :, 15]
    n_best_true_linucb_seq_runs = result[:, :, 16]
    n_best_true_linucbpro_seq_runs = result[:, :, 17]

    gender_pro_seq_runs = result[:, :, 18:18 + 6]
    gender_full_seq_runs = result[:, :, 18 + 6:18 + 2 * 6]
    gender_unprotect_seq_runs = result[:, :, 18 + 2 * 6:18 + 3 * 6]
    gender_ground_seq_runs = result[:, :, 18 + 3 * 6:18 + 4 * 6]
    gender_linucb_seq_runs = result[:, :, 18 + 4 * 6:18 + 5 * 6]
    gender_linucbpro_seq_runs = result[:, :, 18 + 5 * 6:18 + 6 * 6]

    reward_true_pro_seq_runs = result[:, :, 18 + 6 * 6]
    reward_true_full_seq_runs = result[:, :, 18 + 6 * 6 + 1]
    reward_true_unprotect_seq_runs = result[:, :, 18 + 6 * 6 + 2]
    reward_true_ground_seq_runs = result[:, :, 18 + 6 * 6 + 3]
    reward_true_linucb_seq_runs = result[:, :, 18 + 6 * 6 + 4]
    reward_true_linucbpro_seq_runs = result[:, :, 18 + 6 * 6 + 5]

    print('That took {} seconds'.format(time.time() - starttime))

    ###count number from each type(gender)


    scipy.io.savemat('./result/wine/result_matlab.mat', \
                     mdict={'sumr_t_pro_seq_runs': sumr_t_pro_seq_runs,
                            'sumr_t_full_seq_runs': sumr_t_full_seq_runs,
                            'sumr_t_unprotect_seq_runs': sumr_t_unprotect_seq_runs,
                            'sumr_t_ground_seq_runs': sumr_t_ground_seq_runs,
                            'sumr_t_linucb_seq_runs': sumr_t_linucb_seq_runs,
                            'sumr_t_linucbpro_seq_runs': sumr_t_linucbpro_seq_runs,

                            'gender_pro_seq_runs': gender_pro_seq_runs,
                            'gender_full_seq_runs': gender_full_seq_runs,
                            'gender_unprotect_seq_runs': gender_unprotect_seq_runs,
                            'gender_ground_seq_runs': gender_ground_seq_runs,
                            'gender_linucb_seq_runs': gender_linucb_seq_runs,
                            'gender_linucbpro_seq_runs': gender_linucbpro_seq_runs,

                            'reward_true_pro_seq_runs': reward_true_pro_seq_runs,
                            'reward_true_full_seq_runs': reward_true_full_seq_runs,
                            'reward_true_unprotect_seq_runs': reward_true_unprotect_seq_runs,
                            'reward_true_ground_seq_runs': reward_true_ground_seq_runs,
                            'reward_true_linucb_seq_runs': reward_true_linucb_seq_runs,
                            'reward_true_linucbpro_seq_runs': reward_true_linucbpro_seq_runs,

                            'regret_true_pro_seq_runs': regret_true_pro_seq_runs,
                            'regret_true_full_seq_runs': regret_true_full_seq_runs,
                            'regret_true_unprotect_seq_runs': reward_true_unprotect_seq_runs,
                            'regret_true_ground_seq_runs': regret_true_ground_seq_runs,
                            'regret_true_linucb_seq_runs': regret_true_linucb_seq_runs,
                            'regret_true_linucbpro_seq_runs': regret_true_linucbpro_seq_runs,

                            'n_best_pro_seq_runs': n_best_true_pro_seq_runs,
                            'n_best_full_seq_runs': n_best_true_full_seq_runs,
                            'n_best_unprotect_seq_runs': n_best_true_unprotect_seq_runs,
                            'n_best_ground_seq_runs': n_best_true_ground_seq_runs,
                            'n_best_linucb_seq_runs': n_best_true_linucb_seq_runs,
                            'n_best_linucbpro_seq_runs': n_best_true_linucbpro_seq_runs,

                            })
    np.save('./result/wine/sumr_t_pro_seq_runs.npy', sumr_t_pro_seq_runs)
    np.save('./result/wine/sumr_t_full_seq_runs.npy', sumr_t_full_seq_runs)
    np.save('./result/wine/sumr_t_unprotect_seq_runs.npy', sumr_t_unprotect_seq_runs)
    np.save('./result/wine/sumr_t_ground_seq_runs.npy', sumr_t_ground_seq_runs)
    np.save('./result/wine/sumr_t_linucb_seq_runs.npy', sumr_t_linucb_seq_runs)
    np.save('./result/wine/sumr_t_linucbpro_seq_runs.npy', sumr_t_linucbpro_seq_runs)

    np.save('./result/wine/gender_pro_seq_runs.npy', gender_pro_seq_runs)
    np.save('./result/wine/gender_full_seq_runs.npy', gender_full_seq_runs)
    np.save('./result/wine/gender_unprotect_seq_runs.npy', gender_unprotect_seq_runs)
    np.save('./result/wine/gender_ground_seq_runs.npy', gender_ground_seq_runs)
    np.save('./result/wine/gender_linucb_seq_runs.npy', gender_linucb_seq_runs)
    np.save('./result/wine/gender_linucbpro_seq_runs.npy', gender_linucbpro_seq_runs)

    np.save('./result/wine/reward_true_pro_seq_runs', reward_true_pro_seq_runs)
    np.save('./result/wine/reward_true_full_seq_runs', reward_true_full_seq_runs)
    np.save('./result/wine/regret_true_unprotect_seq_runs', reward_true_unprotect_seq_runs)
    np.save('./result/wine/regret_true_ground_seq_runs', reward_true_ground_seq_runs)
    np.save('./result/wine/regret_true_linucb_seq_runs', reward_true_linucb_seq_runs)
    np.save('./result/wine/regret_true_linucbpro_seq_runs', reward_true_linucbpro_seq_runs)

    np.save('./result/wine/regret_true_pro_seq_runs', regret_true_pro_seq_runs)
    np.save('./result/wine/regret_true_full_seq_runs', regret_true_full_seq_runs)
    np.save('./result/wine/regret_true_unprotect_seq_runs', regret_true_unprotect_seq_runs)
    np.save('./result/wine/regret_true_ground_seq_runs', regret_true_ground_seq_runs)
    np.save('./result/wine/regret_true_linucb_seq_runs', regret_true_linucb_seq_runs)
    np.save('./result/wine/regret_true_linucbpro_seq_runs', regret_true_linucbpro_seq_runs)

    np.save('./result/wine/n_best_full_seq_runs', n_best_true_full_seq_runs)
    np.save('./result/wine/n_best_pro_seq_runs', n_best_true_pro_seq_runs)
    np.save('./result/wine/n_best_unprotect_seq_runs', n_best_true_unprotect_seq_runs)
    np.save('./result/wine/n_best_ground_seq_runs', n_best_true_ground_seq_runs)
    np.save('./result/wine/n_best_linucb_seq_runs', n_best_true_linucb_seq_runs)
    np.save('./result/wine/n_best_linucbpro_seq_runs', n_best_true_linucbpro_seq_runs)

    np.save('./result/wine/result.npy', result)
    scipy.io.savemat('./result/wine/result.mat',
                     mdict={'result': result})

    n_runtime = sumr_t_full_seq_runs.shape[0]
    sumr_full = np.sum(sumr_t_full_seq_runs, axis=0) / n_runtime
    sumr_pro = np.sum(sumr_t_pro_seq_runs, axis=0) / n_runtime
    sumr_unprotect = np.sum(sumr_t_unprotect_seq_runs, axis=0) / n_runtime
    sumr_ground = np.sum(sumr_t_ground_seq_runs, axis=0) / n_runtime
    sumr_linucb = np.sum(sumr_t_linucb_seq_runs, axis=0) / n_runtime
    sumr_linucbpro = np.sum(sumr_t_linucbpro_seq_runs, axis=0) / n_runtime

    regret_pro = np.sum(regret_true_pro_seq_runs, axis=0) / n_runtime
    regret_full = np.sum(regret_true_full_seq_runs, axis=0) / n_runtime
    regret_unprotect = np.sum(regret_true_unprotect_seq_runs, axis=0) / n_runtime
    regret_ground = np.sum(regret_true_ground_seq_runs, axis=0) / n_runtime
    regret_linucb = np.sum(regret_true_linucb_seq_runs, axis=0) / n_runtime
    regret_linucbpro = np.sum(regret_true_linucbpro_seq_runs, axis=0) / n_runtime

    n_best_pro = np.sum(n_best_true_pro_seq_runs, axis=0) / n_runtime
    n_best_full = np.sum(n_best_true_full_seq_runs, axis=0) / n_runtime
    n_best_unprotect = np.sum(n_best_true_unprotect_seq_runs, axis=0) / n_runtime
    n_best_ground = np.sum(n_best_true_ground_seq_runs, axis=0) / n_runtime
    n_best_linucb = np.sum(n_best_true_linucb_seq_runs, axis=0) / n_runtime
    n_best_linucbpro = np.sum(n_best_true_linucbpro_seq_runs, axis=0) / n_runtime

    female_whole_pro = np.sum(gender_pro_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_pro = np.sum(gender_pro_seq_runs[:, :, 3], axis=0) / n_runtime
    female_whole_full = np.sum(gender_full_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_full = np.sum(gender_full_seq_runs[:, :, 3], axis=0) / n_runtime
    female_whole_unprotect = np.sum(gender_unprotect_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_unprotect = np.sum(gender_unprotect_seq_runs[:, :, 3], axis=0) / n_runtime
    female_whole_ground = np.sum(gender_ground_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_ground = np.sum(gender_ground_seq_runs[:, :, 3], axis=0) / n_runtime
    female_whole_linucb = np.sum(gender_linucb_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_linucb = np.sum(gender_linucb_seq_runs[:, :, 3], axis=0) / n_runtime
    female_whole_linucbpro = np.sum(gender_linucbpro_seq_runs[:, :, 2], axis=0) / n_runtime
    male_whole_linucbpro = np.sum(gender_linucbpro_seq_runs[:, :, 3], axis=0) / n_runtime
    if args.plot:
        plt.figure(1)
        plt.title('reward')
        plt.plot(sumr_full, 'r-', sumr_pro, 'g-', sumr_unprotect, 'b-', sumr_ground, 'k-')
        plt.savefig('./reward.pdf', bbox_inches='tight')
        plt.figure(2)
        plt.title('regret')
        plt.plot(regret_full, 'r-', regret_pro, 'g-', regret_unprotect, 'b-', regret_ground, 'k-')
        plt.savefig('./regret.pdf', bbox_inches='tight')
        plt.figure(3)
        plt.title('n_best')
        plt.plot(n_best_full, 'r-', n_best_pro, 'g-', n_best_unprotect, 'b-', n_best_ground, 'k-')
        plt.show()
        plt.savefig('./n_best.pdf', bbox_inches='tight')


        plt.figure(4)
        plt.title('discrimation')
        plt.plot(female_whole_pro, 'g-', male_whole_pro, 'g-.',
                 female_whole_full, 'r-', male_whole_full, 'r-.',
                 female_whole_unprotect, 'b-', male_whole_unprotect, 'b-.',
                 female_whole_ground, 'k-', male_whole_ground, 'k-.')
        plt.show()
        plt.savefig('./discrimation.pdf', bbox_inches='tight')

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
        plt.savefig('./full_sequence.pdf', bbox_inches='tight')

        plt.figure(6)
        n_groups = 4
        female_n = (female_whole_pro[-1]-female_whole_pro[args.n_trial//(2*args.recording_time)+1],
                    female_whole_full[-1]-female_whole_full[args.n_trial//(2*args.recording_time)+1],
                    female_whole_unprotect[-1]-female_whole_unprotect[args.n_trial//(2*args.recording_time)+1],
                    female_whole_ground[-1]-female_whole_ground[args.n_trial//(2*args.recording_time)+1])
        male_n = (male_whole_pro[-1]-male_whole_pro[args.n_trial//(2*args.recording_time)+1],
                    male_whole_full[-1]-male_whole_full[args.n_trial//(2*args.recording_time)+1],
                    male_whole_unprotect[-1]-male_whole_unprotect[args.n_trial//(2*args.recording_time)+1],
                    male_whole_ground[-1]-male_whole_ground[args.n_trial//(2*args.recording_time)+1])
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
        plt.savefig('./lasthalf.pdf', bbox_inches='tight')


    print(1)

if __name__ == '__main__':
    main()



