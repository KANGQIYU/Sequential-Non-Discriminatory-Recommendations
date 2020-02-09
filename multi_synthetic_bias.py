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
from mkdirr import mkdirp

# Theta = np.zeros(1)
# n_feat_dim = 1
args = 0
# Arms = np.zeros(1)
# Projection = np.zeros(1)
# D_k = np.zeros(1)
# n_Dk = 1
# rewards_true = np.zeros(1)
# rewards_fake = np.zeros(1)


def runone(run_id):
    # global Arms
    n_protect_dim = args.n_protect_dim
    n_unprotect_dim = args.n_unprotect_dim
    n_feat_dim = n_protect_dim + n_unprotect_dim
    # Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
    # Arms = Arms/np.power(np.sum(Arms*Arms, axis=0), 1/2)
    # pca1 = decomposition.PCA(n_components=n_feat_dim)
    # pca1.fit(Arms.T)
    # Arms = pca1.transform(Arms.T).T
    # tempid = np.random.choice(Arms.shape[1], math.floor(Arms.shape[1]/2), replace=False)
    # templastrow = np.zeros([1, Arms.shape[1]])
    # templastrow[0, tempid] = 1
    # Arms = np.vstack((Arms, np.ones([1, Arms.shape[1]]), templastrow))
    # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # n_feat_dim = n_feat_dim+2
    Arms = 2 * rand(n_feat_dim-1, args.n_Arms) - 1
    Theta = 2 * rand(n_feat_dim, 1) - 1
    Projection = eye(n_feat_dim)
    Projection[-1, -1] = 0
    tempid = np.random.choice(Arms.shape[1], math.floor(Arms.shape[1]/2), replace=False)
    templastrow = np.zeros([1, Arms.shape[1]])
    templastrow[0, tempid] = 1
    Arms = np.vstack((Arms, templastrow))
    # D_k = eye(n_feat_dim)
    D_k_id = np.random.randint(args.n_Arms, size=n_feat_dim)
    D_k = Arms[:, D_k_id]
    n_Dk = D_k.shape[1]

    rewards_true = Arms.T@(Projection@Theta)
    bias = np.zeros([args.n_Arms, 1])
    bias[tempid, 0] = 1
    # bias = np.ones([args.n_Arms, 1])
    # bias[tempid, 0] = 0

    rewards_fake = rewards_true+bias

    print(run_id)
    ooo = 0
    sigma_noise = args.sigma_noise
    # ProArms = Projection @ Arms
    decrease_fun = args.decrease_fun

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

    epsilon_f_ground = lambda x: min(1, args.alpha_ground*n_Dk/x)
    epsilon_i_ground = lambda x: min(1, args.alpha_ground*n_Dk/x**(1.0/3))
    epsilon_o_ground = lambda x: min(1, args.alpha_ground*n_Dk/x**(1.0/2))
    functiondic_ground = {'i': epsilon_i_ground, 'f': epsilon_f_ground, 'o': epsilon_o_ground}

    #projection initial
    X_t_pro = np.zeros([n_feat_dim, 1])
    sumrX_pro = np.zeros([n_feat_dim, 1])
    sumr_t_pro = np.zeros([1])
    V_t_pro = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_pro = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_pro = np.zeros([n_feat_dim, 1])
    sumr_t_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_pro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    # full initial
    X_t_full = np.zeros([n_feat_dim, 1])
    sumrX_full = np.zeros([n_feat_dim, 1])
    sumr_t_full = np.zeros([1])
    V_t_full = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_full = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_full = np.zeros([n_feat_dim, 1])
    sumr_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    #only unprotect initial
    X_t_unprotect = np.zeros([n_feat_dim, 1])
    sumrX_unprotect = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect = np.zeros([1])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_unprotect = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    #ground initial
    X_t_ground = np.zeros([n_feat_dim, 1])
    sumrX_ground = np.zeros([n_feat_dim, 1])
    sumr_t_ground = np.zeros([1])
    V_t_ground = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_ground = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_ground = np.zeros([n_feat_dim, 1])
    sumr_t_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 1])


    #linucb initial
    X_t_linucb = np.zeros([n_feat_dim, 1])
    sumrX_linucb = np.zeros([n_feat_dim, 1])
    sumr_t_linucb = np.zeros([1])
    V_t_linucb = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_linucb = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_linucb = np.zeros([n_feat_dim, 1])
    sumr_t_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    #linucbpro initial
    X_t_linucbpro = np.zeros([n_feat_dim, 1])
    sumrX_linucbpro = np.zeros([n_feat_dim, 1])
    sumr_t_linucbpro = np.zeros([1])
    V_t_linucbpro = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_linucbpro = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_linucbpro = np.zeros([n_feat_dim, 1])
    sumr_t_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    ##################   initial projection ########################
    SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_pro]]
    X_t_pro = X_t
    V_t_pro = X_t_pro @ X_t_pro.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_pro = inv(V_t_pro)
    r_t_pro = rewards_fake[SelectArm_pro, 0] + randn(1)*sigma_noise
    sumrX_pro = r_t_pro * X_t_pro
    sumr_t_pro = r_t_pro
    ####################    initial full arms, they are the same as pro#######################
    X_t_full = X_t_pro.copy()
    V_t_full = V_t_pro.copy()
    V_t_inv_full = V_t_inv_pro.copy()
    sumrX_full = sumrX_pro.copy()
    sumr_t_full = sumr_t_pro.copy()


    #############################   initial unprotected feature arms #########################
    SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_unprotect]]
    X_t_unprotect = Projection@X_t
    V_t_unprotect = X_t_unprotect @ X_t_unprotect.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_unprotect = inv(V_t_unprotect)
    r_t_unprotect = rewards_fake[SelectArm_unprotect, 0] + randn(1, 1)*sigma_noise
    sumrX_unprotect = r_t_unprotect * X_t_unprotect
    sumr_t_unprotect = r_t_unprotect

    #############################   initial ground feature arms #########################
    SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_ground]]
    X_t_ground = Projection@X_t
    V_t_ground = X_t_ground @ X_t_ground.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_ground = inv(V_t_ground)
    r_t_ground = rewards_true[SelectArm_ground, 0] + randn(1, 1)*sigma_noise
    sumrX_ground = r_t_ground * X_t_ground
    sumr_t_ground = r_t_ground

    #############################   initial linucb #########################
    SelectArm_linucb = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_linucb]]
    X_t_linucb = X_t
    V_t_linucb = X_t_linucb @ X_t_linucb.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_linucb = inv(V_t_linucb)
    r_t_linucb = rewards_true[SelectArm_linucb, 0] + randn(1, 1)*sigma_noise
    sumrX_linucb = r_t_linucb * X_t_linucb
    sumr_t_linucb = r_t_linucb

    #############################   initial linucbpro #########################
    SelectArm_linucbpro = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_linucbpro]]
    X_t_linucbpro = Projection@X_t
    V_t_linucbpro = X_t_linucbpro @ X_t_linucbpro.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_linucbpro = inv(V_t_linucbpro)
    r_t_linucbpro = rewards_true[SelectArm_linucbpro, 0] + randn(1, 1)*sigma_noise
    sumrX_linucbpro = r_t_linucbpro * X_t_linucbpro
    sumr_t_linucbpro = r_t_linucbpro

    # reward = Arms.T @ (Projection @ Theta)
    best_arm = argmax(rewards_true)
    best_reward = rewards_true.ravel()[best_arm]

    n_best_pro = np.ones([1, 1])
    n_best_full = np.ones([1, 1])
    n_best_unprotect = np.ones([1, 1])
    n_best_ground = np.ones([1, 1])
    n_best_linucb = np.ones([1, 1])
    n_best_linucbpro = np.ones([1, 1])

    regret_pro = np.ones([1, 1])
    regret_full = np.ones([1, 1])
    regret_unprotect = np.ones([1, 1])
    regret_ground = np.ones([1, 1])
    regret_linucb = np.ones([1, 1])
    regret_linucbpro = np.ones([1, 1])

    gender_pro_sumr_t = np.zeros([1, 2])
    gender_pro_n_rate = np.zeros([1, 2])
    gender_pro_true_reward = np.zeros([1, 2])

    gender_full_sumr_t = np.zeros([1, 2])
    gender_full_n_rate = np.zeros([1, 2])
    gender_full_true_reward = np.zeros([1, 2])

    gender_unprotect_sumr_t = np.zeros([1, 2])
    gender_unprotect_n_rate = np.zeros([1, 2])
    gender_unprotect_true_reward = np.zeros([1, 2])

    gender_ground_sumr_t = np.zeros([1, 2])
    gender_ground_n_rate = np.zeros([1,2])
    gender_ground_true_reward = np.zeros([1, 2])

    gender_linucb_sumr_t = np.zeros([1, 2])
    gender_linucb_n_rate = np.zeros([1, 2])
    gender_linucb_true_reward = np.zeros([1, 2])

    gender_linucbpro_sumr_t = np.zeros([1,2])
    gender_linucbpro_n_rate = np.zeros([1, 2])
    gender_linucbpro_true_reward = np.zeros([1, 2])







    gender_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_full_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_linucb_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_linucbpro_seq = np.zeros([args.n_trial//args.recording_time+1, 6])

    n_feat_dim = n_feat_dim - 2

    for rtime in range(args.n_trial):
        # if at each time, arms are update.
        # ProArms = Projection @ Arms
        if args.infinite:
            # Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
            # Arms = Arms / np.power(np.sum(Arms * Arms, axis=0), 1 / 2)
            # reward = Arms.T@(Projection@Theta)
            # best_arm = argmax(reward)
            # best_reward = reward.ravel()[best_arm]
            Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
            Arms = Arms / np.power(np.sum(Arms * Arms, axis=0), 1 / 2)
            pca1 = decomposition.PCA(n_components=n_feat_dim)
            pca1.fit(Arms.T)
            Arms = pca1.transform(Arms.T).T
            tempid = np.random.choice(Arms.shape[1], math.floor(Arms.shape[1] / 2), replace=False)
            templastrow = np.zeros([1, Arms.shape[1]])
            templastrow[0, tempid] = 1
            Arms = np.vstack((Arms, np.ones([1, Arms.shape[1]]), templastrow))
            rewards_true = Arms.T @ Theta
            bias = np.zeros([args.n_Arms, 1])
            bias[tempid, 0] = 1
            # bias = np.ones([args.n_Arms, 1])
            # bias[tempid, 0] = 0
            rewards_fake = rewards_true + bias
            best_arm = argmax(rewards_true)
            best_reward = rewards_true.ravel()[best_arm]

        ###############################  Projection  ##############################
        EstTheta_pro = V_t_inv_pro @ sumrX_pro
        if rand(1) > functiondic_pro[decrease_fun](rtime + 1):
            # SelectArm_pro = argmax(Arms.T @ (Projection@EstTheta_pro))
            SelectArm_pro = argmax(Arms.T @ np.vstack((EstTheta_pro[0:-1, :], np.zeros([1, 1]))))
        else:
            SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
            # SelectArm_pro = np.random.choice(D_k, 1)

        X_t_pro = Arms[:, [SelectArm_pro]]
        # r_t_pro = X_t_pro.T @ Theta + randn(1, 1)*sigma_noise
        r_t_pro = rewards_fake[SelectArm_pro, 0] + randn(1)*sigma_noise
        r_t_true = rewards_true[SelectArm_pro, 0]
        sumr_t_pro = sumr_t_pro + r_t_pro
        n_best_pro += (SelectArm_pro == best_arm)
        # regret_pro += best_reward-Arms[:, [SelectArm_pro]].T@ (Projection@Theta)
        regret_pro += best_reward-r_t_true

        V_t_pro = V_t_pro + X_t_pro @ X_t_pro.T
        V_t_inv_ = V_t_inv_pro
        V_t_inv_pro = V_t_inv_ - V_t_inv_ @ X_t_pro @ X_t_pro.T @ V_t_inv_ / (1 + X_t_pro.T @ V_t_inv_ @ X_t_pro)
        sumrX_pro += r_t_pro * X_t_pro

        gender = int(Arms[-1, SelectArm_pro])

        gender_pro_n_rate[:, gender] += 1
        gender_pro_true_reward[:, gender] += r_t_true
        gender_pro_sumr_t[:, gender] += r_t_pro

        #####################################  Full   ###########################################
        EstTheta_full = V_t_inv_full @ sumrX_full
        if rand(1) > functiondic_full[decrease_fun](rtime + 1):
            SelectArm_full = argmax(Arms.T @ EstTheta_full)
        else:
            SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]
            # SelectArm_full = np.random.choice(D_k, 1)

        X_t_full = Arms[:, [SelectArm_full]]
        # r_t_full = X_t_full.T@Theta + randn(1, 1)*sigma_noise
        r_t_full = rewards_fake[SelectArm_full, 0] + randn(1)*sigma_noise
        r_t_true = rewards_true[SelectArm_full, 0]
        sumr_t_full = sumr_t_full + r_t_full
        n_best_full += (SelectArm_full == best_arm)
        # regret_full += best_reward - Arms[:, [SelectArm_full]].T @ (Projection @ Theta)
        regret_full += best_reward - r_t_true
        V_t_full = V_t_full + X_t_full @ X_t_full.T
        V_t_inv_ = V_t_inv_full
        V_t_inv_full = V_t_inv_ - V_t_inv_ @ X_t_full @ X_t_full.T @ V_t_inv_ / (
                    1 + X_t_full.T @ V_t_inv_ @ X_t_full)
        sumrX_full += r_t_full * X_t_full

        gender = int(Arms[-1, SelectArm_full])
        gender_full_n_rate[:, gender] += 1
        gender_full_sumr_t[:, gender] += r_t_full
        gender_full_true_reward[:, gender] += r_t_true

        ################################### Only unprotect #############################################
        EstTheta_unprotect = V_t_inv_unprotect @ sumrX_unprotect
        if rand(1) > functiondic_unprotect[decrease_fun](rtime + 1):
            # SelectArm_unprotect = argmax(Arms.T @ (Projection@EstTheta_unprotect))
            SelectArm_unprotect = argmax(Arms.T @ np.vstack((EstTheta_unprotect[0:-1, :], np.zeros([1, 1]))))
        else:
            SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
            # SelectArm_unprotect = np.random.choice(D_k, 1)

        X_t = Arms[:, [SelectArm_unprotect]]
        X_t_unprotect = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        # X_t_unprotect = Projection@X_t
        # r_t_unprotect = X_t.T@Theta + randn(1, 1)*sigma_noise
        r_t_unprotect = rewards_fake[SelectArm_unprotect, 0] + randn(1) * sigma_noise
        r_t_true = rewards_true[SelectArm_unprotect, 0]
        sumr_t_unprotect = sumr_t_unprotect + r_t_unprotect
        n_best_unprotect += (SelectArm_unprotect == best_arm)
        # regret_unprotect += best_reward - Arms[:, [SelectArm_unprotect]].T @ (Projection @ Theta)
        regret_unprotect += best_reward - r_t_true

        V_t_unprotect = V_t_unprotect+ X_t_unprotect @ X_t_unprotect.T
        V_t_inv_ = V_t_inv_unprotect
        V_t_inv_unprotect = V_t_inv_ - V_t_inv_ @ X_t_unprotect @ X_t_unprotect.T @ V_t_inv_ / (
                    1 + X_t_unprotect.T @ V_t_inv_ @ X_t_unprotect)
        sumrX_unprotect += r_t_unprotect * X_t_unprotect

        gender = int(Arms[-1, SelectArm_unprotect])
        gender_unprotect_n_rate[:, gender] += 1
        gender_unprotect_sumr_t[:, gender] += r_t_unprotect
        gender_unprotect_true_reward[:, gender] += r_t_true

        ###################################  ground truth  #############################################
        # EstTheta_ground = V_t_inv_ground @ sumrX_ground
        # if rand(1) > functiondic_ground[decrease_fun](rtime + 1):
        #     SelectArm_ground = argmax(Arms.T @ (Projection@EstTheta_ground))
        # else:
        #     SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]
        #     # SelectArm_unprotect = np.random.choice(D_k, 1)
        #
        # X_t = Arms[:, [SelectArm_ground]]
        # # X_t_unprotect = Projection@X_t
        # X_t_ground = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        # r_t_true = rewards_true[SelectArm_ground, 0]
        # noise = randn(1) * sigma_noise
        # r_t_ground = rewards_fake[SelectArm_ground, 0] + noise
        # sumr_t_ground = sumr_t_ground + r_t_ground
        # n_best_ground += (SelectArm_ground == best_arm)
        # regret_ground += best_reward - r_t_true
        #
        # V_t_ground = V_t_ground + X_t_ground @ X_t_ground.T
        # V_t_inv_ = V_t_inv_ground
        # V_t_inv_ground = V_t_inv_ - V_t_inv_ @ X_t_ground @ X_t_ground.T @ V_t_inv_ / (
        #             1 + X_t_ground.T @ V_t_inv_ @ X_t_ground)
        # sumrX_ground += (r_t_true + noise) * X_t_ground
        #
        # gender = int(Arms[-1, SelectArm_ground])
        # gender_ground_n_rate[:, gender] += 1
        # gender_ground_sumr_t[:, gender] += r_t_ground
        # gender_ground_true_reward[:, gender] += r_t_true


        ###############################    linucb        ################################
        # EstTheta_linucb = V_t_inv_linucb @ sumrX_linucb
        # upperbound = args.alpha_linucb*np.sqrt(np.sum(Arms*(V_t_inv_linucb@Arms), axis=0)).reshape(-1, 1)
        # SelectArm_linucb = argmax(Arms.T @ (EstTheta_linucb)+upperbound)
        #
        # X_t_linucb = Arms[:, [SelectArm_ground]]
        # r_t_linucb = rewards_fake[SelectArm_linucb, 0] + randn(1) * sigma_noise
        # r_t_true = rewards_true[SelectArm_linucb, 0]
        # sumr_t_linucb = sumr_t_linucb + r_t_linucb
        # n_best_linucb += (SelectArm_linucb == best_arm)
        # # regret_linucb += best_reward - Arms[:, [SelectArm_linucb]].T @ (Projection @ Theta)
        # regret_linucb += best_reward - r_t_true
        # V_t_linucb = V_t_linucb + X_t_linucb @ X_t_linucb.T
        # V_t_inv_ = V_t_inv_linucb
        # V_t_inv_linucb = V_t_inv_ - V_t_inv_ @ X_t_linucb @ X_t_linucb.T @ V_t_inv_ / (
        #         1 + X_t_linucb.T @ V_t_inv_ @ X_t_linucb)
        # sumrX_linucb += r_t_linucb * X_t_linucb
        #
        # gender = int(Arms[-1, SelectArm_linucb])
        # gender_linucb_n_rate[:, gender] += 1
        # gender_linucb_sumr_t[:, gender] += r_t_linucb
        # gender_linucb_true_reward[:, gender] += r_t_true

        ###############################    linucbpro        ################################
        EstTheta_linucbpro = V_t_inv_linucbpro @ sumrX_linucbpro
        upperbound = args.alpha_linucbpro*np.sqrt(np.sum(Arms*(V_t_inv_linucbpro@Arms), axis=0)).reshape(-1, 1)
        SelectArm_linucbpro = argmax(Arms.T @ (Projection@EstTheta_linucbpro)+upperbound)

        X_t_linucbpro = Arms[:, [SelectArm_linucbpro]]
        # X_t_linucbpro = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        r_t_linucbpro = rewards_fake[SelectArm_linucbpro, 0] + randn(1) * sigma_noise
        r_t_true = rewards_true[SelectArm_linucbpro, 0]
        sumr_t_linucbpro = sumr_t_linucbpro + r_t_linucbpro
        n_best_linucbpro += (SelectArm_linucbpro == best_arm)
        # regret_linucbpro += best_reward - Arms[:, [SelectArm_linucbpro]].T @ (Projection @ Theta)
        regret_linucbpro += best_reward - r_t_true
        V_t_linucbpro = V_t_linucbpro + X_t_linucbpro @ X_t_linucbpro.T
        V_t_inv_ = V_t_inv_linucbpro
        V_t_inv_linucbpro = V_t_inv_ - V_t_inv_ @ X_t_linucbpro @ X_t_linucbpro.T @ V_t_inv_ / (
                1 + X_t_linucbpro.T @ V_t_inv_ @ X_t_linucbpro)
        sumrX_linucbpro += r_t_linucbpro * X_t_linucbpro

        gender = int(Arms[-1, SelectArm_linucbpro])
        gender_linucbpro_n_rate[:, gender] += 1
        gender_linucbpro_sumr_t[:, gender] += r_t_linucbpro
        gender_linucbpro_true_reward[:, gender] += r_t_true
        ############################   store #######################################
        if (rtime + 1) % args.recording_time == 0:
            sumr_t_pro_seq[ooo+1, :] = sumr_t_pro.flatten()
            regret_t_pro_seq[ooo+1, :] = regret_pro.flatten()
            n_best_t_pro_seq[ooo+1, :] = n_best_pro.flatten()

            ##why flatten() #
            # for future discrimination compare like male female
            sumr_t_full_seq[ooo + 1, :] = sumr_t_full.flatten()
            regret_t_full_seq[ooo + 1,:] =regret_full.flatten()
            n_best_t_full_seq[ooo+1, :] = n_best_full.flatten()

            sumr_t_unprotect_seq[ooo + 1, :] = sumr_t_unprotect.flatten()
            regret_t_unprotect_seq[ooo + 1, :] = regret_unprotect.flatten()
            n_best_t_unprotect_seq[ooo+1, :] = n_best_unprotect.flatten()

            sumr_t_ground_seq[ooo + 1, :] = sumr_t_ground.flatten()
            regret_t_ground_seq[ooo + 1, :] = regret_ground.flatten()
            n_best_t_ground_seq[ooo + 1, :] = n_best_ground.flatten()

            sumr_t_linucb_seq[ooo + 1, :] = sumr_t_linucb.flatten()
            regret_t_linucb_seq[ooo + 1, :] = regret_linucb.flatten()
            n_best_t_linucb_seq[ooo + 1, :] = n_best_linucb.flatten()

            sumr_t_linucbpro_seq[ooo + 1, :] = sumr_t_linucbpro.flatten()
            regret_t_linucbpro_seq[ooo + 1, :] = regret_linucbpro.flatten()
            n_best_t_linucbpro_seq[ooo + 1, :] = n_best_linucbpro.flatten()

            gender_pro_seq[ooo + 1, :] = np.hstack((gender_pro_sumr_t, gender_pro_n_rate, gender_pro_true_reward)).flatten()
            gender_full_seq[ooo + 1, :] = np.hstack((gender_full_sumr_t, gender_full_n_rate, gender_full_true_reward)).flatten()
            gender_unprotect_seq[ooo + 1, :] = np.hstack((gender_unprotect_sumr_t, gender_unprotect_n_rate, gender_unprotect_true_reward)).flatten()
            gender_ground_seq[ooo + 1, :] = np.hstack((gender_ground_sumr_t, gender_ground_n_rate, gender_ground_true_reward)).flatten()
            gender_linucb_seq[ooo + 1, :] = np.hstack((gender_linucb_sumr_t, gender_linucb_n_rate, gender_linucb_true_reward)).flatten()
            gender_linucbpro_seq[ooo + 1, :] = np.hstack((gender_linucbpro_sumr_t, gender_linucbpro_n_rate, gender_linucbpro_true_reward)).flatten()

            ooo += 1

    # return np.sum(sumr_t_pro, axis=1)
    return np.hstack((sumr_t_pro_seq,
                      sumr_t_full_seq,
                      sumr_t_unprotect_seq,
                      sumr_t_ground_seq,
                      sumr_t_linucb_seq,
                      sumr_t_linucbpro_seq,

                      regret_t_pro_seq,
                      regret_t_full_seq,
                      regret_t_unprotect_seq,
                      regret_t_ground_seq,
                      regret_t_linucb_seq,
                      regret_t_linucbpro_seq,

                      n_best_t_pro_seq,
                      n_best_t_full_seq,
                      n_best_t_unprotect_seq,
                      n_best_t_ground_seq,
                      n_best_t_linucb_seq,
                      n_best_t_linucbpro_seq,

                      gender_pro_seq,
                      gender_full_seq,
                      gender_unprotect_seq,
                      gender_ground_seq,
                      gender_linucb_seq,
                      gender_linucbpro_seq
                      ))

def main():
    # Training settings
    # global n_feat_dim
    global args
    # global Arms
    # global Projection
    # global D_k
    # global n_Dk
    # global Theta
    # global rewards_true
    # global rewards_fake
    mkdirp('./result/syn_bias')
    whichset = 'syn_bias'
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=5, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                         help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                         help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=200, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_Arms', type=int, default=100, metavar='N',
                        help='set number of arms (default: 100)')

    parser.add_argument('--n_protect_dim', type=int, default=4, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_unprotect_dim', type=int, default=1, metavar='N',
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

    parser.add_argument('--lambda_pro', type=float, default=0.5, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--sigma_noise', type=float, default=1, metavar='M',
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
                            help='if specify, we will generate rand projection instead of diag one')
    parser.add_argument('--not_mul', action='store_true', default=False,
                        help='if specify, not use multiprocessor')
    # # parser.add_argument('--which_dataset', nargs='+',
    # #                     help='Please input which dataset: MNIST, FashionMNIST, CIFAR10', required=True)
    args = parser.parse_args()
    # args.lambda_pro = args.lambda_pro

    #We currently choosely randomly a movie that has a rating for the specific user when each time when choosing it.
    #But we also construct D_k below which may be used.

    # n_protect_dim = args.n_protect_dim
    # n_unprotect_dim = args.n_unprotect_dim
    # n_feat_dim = n_protect_dim + n_unprotect_dim
    #
    # Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
    # Arms = Arms/np.power(np.sum(Arms*Arms, axis=0), 1/2)
    # pca1 = decomposition.PCA(n_components=n_feat_dim)
    # pca1.fit(Arms.T)
    # Arms = pca1.transform(Arms.T).T
    # tempid = np.random.choice(Arms.shape[1], math.floor(Arms.shape[1]/2), replace=False)
    # templastrow = np.zeros([1, Arms.shape[1]])
    # templastrow[0, tempid] = 1
    # Arms = np.vstack((Arms, np.ones([1, Arms.shape[1]]), templastrow))
    #
    # # Theta = 2 * rand(n_feat_dim, 1) - 1
    # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # n_feat_dim = n_feat_dim+2
    # Projection = eye(n_feat_dim)
    # Projection[-1, -1] = 0
    # # please define the projection operator here.
    # # D_k = eye(n_feat_dim)
    # D_k_id = np.random.randint(args.n_Arms, size=n_feat_dim)
    # D_k = Arms[:, D_k_id]
    # n_Dk = D_k.shape[1]
    #
    # rewards_true = Arms.T@Theta
    # bias = np.zeros([args.n_Arms, 1])
    # bias[tempid, 0] = 1
    # # bias = np.ones([args.n_Arms, 1])
    # # bias[tempid, 0] = 0
    #
    # rewards_fake = rewards_true+bias


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if t1here is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters




    starttime = time.time()


    if not args.not_mul:
        with multiprocessing.Pool() as pool:
            result = pool.map(runone, range(0, args.runtimes))
            pool.close()
            pool.join()
            result = np.asarray(result)
    else:
        result = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, 18+6*6])
        for run_id in range(args.runtimes):
            result[run_id, :] = runone(run_id)

    # result = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 36])
    # for run_id in range(args.runtimes):
    #     result[run_id, :] = runone(run_id)
    #
    # print('That took {} seconds'.format(time.time() - starttime))

    sumr_t_pro_seq_runs = result[:, :, 0]
    sumr_t_full_seq_runs = result[:, :, 1]
    sumr_t_unprotect_seq_runs = result[:, :, 2]
    sumr_t_ground_seq_runs = result[:, :, 3]
    sumr_t_linucb_seq_runs = result[:, :, 4]
    sumr_t_linucbpro_seq_runs = result[:, :, 5]

    regret_t_pro_seq_runs = result[:, :, 6]
    regret_t_full_seq_runs = result[:, :, 7]
    regret_t_unprotect_seq_runs = result[:, :, 8]
    regret_t_ground_seq_runs = result[:, :, 9]
    regret_t_linucb_seq_runs = result[:, :, 10]
    regret_t_linucbpro_seq_runs = result[:, :, 11]

    n_best_t_pro_seq_runs = result[:, :, 12]
    n_best_t_full_seq_runs = result[:, :, 13]
    n_best_t_unprotect_seq_runs =result[:, :, 14]
    n_best_t_ground_seq_runs =result[:, :, 15]
    n_best_t_linucb_seq_runs =result[:, :, 16]
    n_best_t_linucbpro_seq_runs =result[:, :, 17]

    gender_pro_seq_runs = result[:, :, 18:24]
    gender_full_seq_runs = result[:, :, 24:30]
    gender_unprotect_seq_runs = result[:, :, 30:36]
    gender_ground_seq_runs = result[:, :, 36:40]
    gender_linucb_seq_runs = result[:, :, 40:46]
    gender_linucbpro_seq_runs = result[:, :, 46:51]

    n_runtime = sumr_t_full_seq_runs.shape[0]

    sumr_full = np.sum(sumr_t_full_seq_runs, axis=0)/n_runtime
    sumr_pro = np.sum(sumr_t_pro_seq_runs, axis=0)/n_runtime
    sumr_unprotect = np.sum(sumr_t_unprotect_seq_runs, axis=0)/n_runtime
    sumr_ground = np.sum(sumr_t_ground_seq_runs, axis=0)/n_runtime
    sumr_linucb = np.sum(sumr_t_linucb_seq_runs, axis=0)/n_runtime
    sumr_linucbpro = np.sum(sumr_t_linucbpro_seq_runs, axis=0)/n_runtime

    regret_pro = np.sum(regret_t_pro_seq_runs, axis=0)/n_runtime
    regret_full = np.sum(regret_t_full_seq_runs, axis=0)/n_runtime
    regret_unprotect = np.sum(regret_t_unprotect_seq_runs, axis=0)/n_runtime
    regret_ground = np.sum(regret_t_ground_seq_runs, axis=0)/n_runtime
    regret_linucb = np.sum(regret_t_linucb_seq_runs, axis=0)/n_runtime
    regret_linucbpro = np.sum(regret_t_linucbpro_seq_runs, axis=0)/n_runtime

    n_best_pro = np.sum(n_best_t_pro_seq_runs, axis=0)/n_runtime
    n_best_full = np.sum(n_best_t_full_seq_runs, axis=0)/n_runtime
    n_best_unprotect = np.sum(n_best_t_unprotect_seq_runs, axis=0)/n_runtime
    n_best_ground = np.sum(n_best_t_ground_seq_runs, axis=0)/n_runtime
    n_best_linucb = np.sum(n_best_t_linucb_seq_runs, axis=0)/n_runtime
    n_best_linucbpro = np.sum(n_best_t_linucbpro_seq_runs, axis=0)/n_runtime

    scipy.io.savemat('./result/'+whichset+'/result_matlab_full_'+whichset+'.mat',
                     mdict={'sumr_t_pro_seq_runs': sumr_t_pro_seq_runs,
                            'sumr_t_full_seq_runs': sumr_t_full_seq_runs,
                            'sumr_t_unprotect_seq_runs': sumr_t_unprotect_seq_runs,
                            'sumr_t_ground_seq_runs': sumr_t_ground_seq_runs,
                            'sumr_t_linucb_seq_runs': sumr_t_linucb_seq_runs,
                            'sumr_t_linucbpro_seq_runs': sumr_t_linucbpro_seq_runs,

                            'regret_t_pro_seq_runs': regret_t_pro_seq_runs,
                            'regret_t_full_seq_runs': regret_t_full_seq_runs,
                            'regret_t_unprotect_seq_runs': regret_t_unprotect_seq_runs,
                            'regret_t_ground_seq_runs': regret_t_ground_seq_runs,
                            'regret_t_linucb_seq_runs': regret_t_linucb_seq_runs,
                            'regret_t_linucbpro_seq_runs': regret_t_linucbpro_seq_runs,

                            'n_best_t_pro_seq_runs': n_best_t_pro_seq_runs,
                            'n_best_t_full_seq_runs': n_best_t_full_seq_runs,
                            'n_best_t_unprotect_seq_runs': n_best_t_unprotect_seq_runs,
                            'n_best_t_ground_seq_runs': n_best_t_ground_seq_runs,
                            'n_best_t_linucb_seq_runs': n_best_t_linucb_seq_runs,
                            'n_best_t_linucbpro_seq_runs': n_best_t_linucbpro_seq_runs,

                            'gender_pro_seq_runs': gender_pro_seq_runs,
                            'gender_full_seq_runs': gender_full_seq_runs,
                            'gender_unprotect_seq_runs': gender_unprotect_seq_runs,
                            'gender_ground_seq_runs': gender_ground_seq_runs,
                            'gender_linucb_seq_runs': gender_linucb_seq_runs,
                            'gender_linucbpro_seq_runs': gender_linucbpro_seq_runs

    })

    np.save('./result/'+whichset+'/sumr_t_pro_seq_runs.npy', sumr_t_pro_seq_runs)
    np.save('./result/'+whichset+'/sumr_t_full_seq_runs.npy', sumr_t_full_seq_runs)
    np.save('./result/'+whichset+'/sumr_t_unprotect_seq_runs.npy', sumr_t_unprotect_seq_runs)
    np.save('./result/'+whichset+'/sumr_t_ground_seq_runs.npy', sumr_t_ground_seq_runs)
    np.save('./result/'+whichset+'/sumr_t_linucb_seq_runs.npy', sumr_t_linucb_seq_runs)
    np.save('./result/'+whichset+'/sumr_t_linucbpro_seq_runs.npy', sumr_t_linucbpro_seq_runs)

    np.save('./result/'+whichset+'/regret_t_pro_seq_runs.npy', regret_t_pro_seq_runs)
    np.save('./result/'+whichset+'/regret_t_full_seq_runs.npy', regret_t_full_seq_runs)
    np.save('./result/'+whichset+'/regret_t_unprotect_seq_runs.npy', regret_t_unprotect_seq_runs)
    np.save('./result/'+whichset+'/regret_t_ground_seq_runs.npy', regret_t_ground_seq_runs)
    np.save('./result/'+whichset+'/regret_t_linucb_seq_runs.npy', regret_t_linucb_seq_runs)
    np.save('./result/'+whichset+'/regret_t_linucbpro_seq_runs.npy', regret_t_linucbpro_seq_runs)

    np.save('./result/'+whichset+'/n_best_t_pro_seq_runs.npy', n_best_t_pro_seq_runs)
    np.save('./result/'+whichset+'/n_best_t_full_seq_runs.npy', n_best_t_full_seq_runs)
    np.save('./result/'+whichset+'/n_best_t_unprotect_seq_runs.npy', n_best_t_unprotect_seq_runs)
    np.save('./result/'+whichset+'/n_best_t_ground_seq_runs.npy', n_best_t_ground_seq_runs)
    np.save('./result/'+whichset+'/n_best_t_linucb_seq_runs.npy', n_best_t_linucb_seq_runs)
    np.save('./result/'+whichset+'/n_best_t_linucbpro_seq_runs.npy', n_best_t_linucbpro_seq_runs)

    np.save('./result/'+whichset+'/gender_pro_seq_runs', gender_pro_seq_runs)
    np.save('./result/'+whichset+'/gender_full_seq_runs', gender_full_seq_runs)
    np.save('./result/'+whichset+'/gender_unprotect_seq_runs', gender_unprotect_seq_runs)
    np.save('./result/'+whichset+'/gender_ground_seq_runs', gender_ground_seq_runs)
    np.save('./result/'+whichset+'/gender_linucb_seq_runs', gender_linucb_seq_runs)
    np.save('./result/'+whichset+'/gender_linucbpro_seq_runs', gender_linucbpro_seq_runs)

    np.save('./result/'+whichset+'/result.npy', result)
    scipy.io.savemat('./result/'+whichset+'/result.mat',
                     mdict={'result': result})

    female_whole_pro = np.sum(gender_pro_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_pro = np.sum(gender_pro_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_full = np.sum(gender_full_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_full = np.sum(gender_full_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_unprotect = np.sum(gender_unprotect_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_unprotect = np.sum(gender_unprotect_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_ground = np.sum(gender_ground_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_ground = np.sum(gender_ground_seq_runs[:,:,3],axis=0)/n_runtime

    female_whole_linucb = np.sum(gender_linucb_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_linucb= np.sum(gender_linucb_seq_runs[:,:,3],axis=0)/n_runtime
    female_whole_linucbpro = np.sum(gender_linucbpro_seq_runs[:,:,2],axis=0)/n_runtime
    male_whole_linucbpro = np.sum(gender_linucbpro_seq_runs[:,:,3],axis=0)/n_runtime
#############the following plot is not correct, should be revised later#########
    if args.plot:
        n_trial = args.n_trial
        recording_time = args.recording_time
        plt.figure(1)
        plt.title('reward')
        plt.plot(sumr_full, 'r-', label='full')
        plt.plot(sumr_pro, 'g-', label='pro')
        plt.plot(sumr_unprotect, 'b-', label='unprotect')
        plt.plot(sumr_ground, 'k-', label='ground')
        plt.plot(sumr_linucb, '-', c='xkcd:plum', label='linucb')
        plt.plot(sumr_linucbpro, '-', c='xkcd:cyan', label='linucbpro')
        plt.savefig('./reward_' + whichset + '.pdf', bbox_inches='tight')
        plt.legend()
        plt.grid(True)

        plt.figure(2)
        plt.title('regret')
        plt.plot(regret_full, 'r-', label='full')
        plt.plot(regret_pro, 'g-', label='pro')
        plt.plot(regret_unprotect, 'b-', label='unprotect')
        plt.plot(regret_ground, 'k-', label='ground')
        plt.plot(regret_linucb, '-', c='xkcd:plum', label='linucb')
        plt.plot(regret_linucbpro, '-', c='xkcd:cyan', label='linucbpro')
        plt.savefig('./regret_' + whichset + '.pdf', bbox_inches='tight')

        plt.figure(3)
        plt.title('n_best')
        plt.plot(n_best_full, 'r-', label='full')
        plt.plot(n_best_pro, 'g-', label='pro')
        plt.plot(n_best_unprotect, 'b-', label='unprotect')
        plt.plot(n_best_ground, 'k-', label='ground')
        plt.plot(n_best_linucb, '-', c='xkcd:plum', label='linucb')
        plt.plot(n_best_linucbpro, '-', c='xkcd:cyan', label='linucbpro')
        plt.show()
        plt.legend()
        plt.grid(True)
        plt.savefig('./n_best_' + whichset + '.pdf', bbox_inches='tight')

        plt.figure(4)
        plt.title('discrimation')
        plt.plot(female_whole_full, 'r-', label='female_full')
        plt.plot(male_whole_full, 'r-.', label='male_full')
        plt.plot(female_whole_pro, 'g-', label='female_pro')
        plt.plot(male_whole_pro, 'g-.', label='male_pro')
        plt.plot(female_whole_unprotect, 'b-', label='female_unprotect')
        plt.plot(male_whole_unprotect, 'b-.', label='male_unprotect')
        plt.plot(female_whole_ground, 'k-', label='female_ground')
        plt.plot(male_whole_ground, 'k-.', label='male_ground')
        plt.plot(female_whole_linucb, '-', c='xkcd:plum', label='female_linucb')
        plt.plot(male_whole_linucb, '-.', c='xkcd:plum', label='male_linucb')
        plt.plot(female_whole_linucbpro, '-', c='xkcd:cyan', label='female_linucbpro')
        plt.plot(male_whole_linucbpro, '-.', c='xkcd:cyan', label='male_linucbpro')
        plt.show()
        plt.legend()
        plt.grid(True)
        plt.savefig('./discrimation_' + whichset + '.pdf', bbox_inches='tight')

        plt.figure(5)
        n_groups = 6
        female_n = (female_whole_pro[-1], female_whole_full[-1],
                    female_whole_unprotect[-1], female_whole_ground[-1],
                    female_whole_linucb[-1], female_whole_linucbpro[-1])
        male_n = (male_whole_pro[-1], male_whole_full[-1],
                  male_whole_unprotect[-1], male_whole_ground[-1],
                  male_whole_linucb[-1], male_whole_linucbpro[-1])
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
        plt.xticks(index + bar_width, ('pro', 'full', 'unprotect', 'ground', 'linucb', 'linucbpro'))
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig('./full_sequence_' + whichset + '.pdf', bbox_inches='tight')

        plt.figure(6)
        n_groups = 6
        female_n = (female_whole_pro[-1] - female_whole_pro[n_trial // (2 * recording_time) + 1],
                    female_whole_full[-1] - female_whole_full[n_trial // (2 * recording_time) + 1],
                    female_whole_unprotect[-1] - female_whole_unprotect[n_trial // (2 * recording_time) + 1],
                    female_whole_ground[-1] - female_whole_ground[n_trial // (2 * recording_time) + 1],
                    female_whole_linucb[-1] - female_whole_linucb[n_trial // (2 * recording_time) + 1],
                    female_whole_linucbpro[-1] - female_whole_linucbpro[n_trial // (2 * recording_time) + 1])
        male_n = (male_whole_pro[-1] - male_whole_pro[n_trial // (2 * recording_time) + 1],
                  male_whole_full[-1] - male_whole_full[n_trial // (2 * recording_time) + 1],
                  male_whole_unprotect[-1] - male_whole_unprotect[n_trial // (2 * recording_time) + 1],
                  male_whole_ground[-1] - male_whole_ground[n_trial // (2 * recording_time) + 1],
                  male_whole_linucb[-1] - male_whole_linucb[n_trial // (2 * recording_time) + 1],
                  male_whole_linucbpro[-1] - male_whole_linucbpro[n_trial // (2 * recording_time) + 1])
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
        plt.xticks(index + bar_width, ('pro', 'full', 'unprotect', 'ground', 'linucb', 'linucbpro'))
        plt.legend()

        plt.tight_layout()
        plt.show()
        plt.savefig('./lasthalf_' + whichset + '.pdf', bbox_inches='tight')

        print(1)


if __name__ == '__main__':
    main()









