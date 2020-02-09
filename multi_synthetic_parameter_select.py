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
n_feat_dim = 1
args = 0
# Arms = np.zeros(1)
Lambda = np.zeros(1)
# Projection = np.zeros(1)
# D_k = np.zeros(1)
# n_Dk = 1
# rewards_true = np.zeros(1)
# rewards_fake = np.zeros(1)
# n_feat_dim = 0

def runone(run_id):
    # global n_feat_dim
    global args
    # global Arms
    global Lambda
    # global Projection
    # global D_k
    # global n_Dk
    # global Theta
    # global Arms
    global n_feat_dim
    ########################Please think how to construct arms in the simulations#############
    n_protect_dim = args.n_protect_dim
    n_unprotect_dim = args.n_unprotect_dim
    n_feat_dim = n_protect_dim + n_unprotect_dim

    # Arms= 2 * rand(n_feat_dim, args.n_Arms-n_feat_dim) - 1
    Arms= 2 * rand(n_feat_dim, args.n_Arms) - 1
    # Arms = Arms/np.power(np.sum(Arms*Arms, axis=0), 1/2)
    # pca1 = decomposition.PCA(n_components=n_feat_dim)
    # pca1.fit(Arms.T)
    # Arms = pca1.transform(Arms.T).T
    if args.rand_pro:
        tempA = 2*np.random.rand(n_feat_dim, n_unprotect_dim)-1
        Projection = tempA@(inv(tempA.T@tempA))@ tempA.T  # get the projection operator
    else:
        Projection = np.diag(np.hstack([np.ones(n_unprotect_dim), np.zeros(n_protect_dim)]))
        # Projection = np.flip(Projection, axis=0)

    # np.save('Projection.npy', Projection)
    Theta = (2 * rand(n_feat_dim, 1) - 1)
    # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # n_feat_dim = n_feat_dim + 2


    # please define the projection operator here.
    # D_k = eye(n_feat_dim)
    # fake_D_k = np.random.randint(args.n_Arms, size=n_feat_dim)
    D_k = 2 * rand(n_feat_dim, n_feat_dim) - 1
    # D_k, r = np.linalg.qr(D_k)
    n_Dk = n_feat_dim
    # args.n_Arms += n_Dk
    Arms = np.hstack((D_k, Arms))
    ########################Please think how to construct arms in the simulations#############
    rewards_fake = Arms.T@Theta
    rewards_true = Arms.T@(Projection@Theta)
    # reward = Arms.T @ (Projection @ Theta)
    best_arm = argmax(rewards_true)
    best_reward = rewards_true.ravel()[best_arm]

    print(run_id)
    ooo = 0
    sigma_noise = args.sigma_noise
    # ProArms = Projection @ Arms
    decrease_fun_1 = 'f'
    decrease_fun_2 = 'f'
    decrease_fun_3 = 'o'

    epsilon_f_pro = lambda x: min(1, args.alpha_pro*n_Dk/x)
    epsilon_i_pro = lambda x: min(1, args.alpha_pro*n_Dk/x**(1.0/3))
    epsilon_o_pro = lambda x: min(1, args.alpha_pro*n_Dk/x**(1.0/2))
    functiondic = {'i': epsilon_i_pro, 'f': epsilon_f_pro, 'o': epsilon_o_pro}


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

####################################################################################
    #projection2 initial
    X_t_full = np.zeros([n_feat_dim, 1])
    sumrX_pro2 = np.zeros([n_feat_dim, 1])
    sumr_t_full = np.zeros([1])
    V_t_full = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_pro2 = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_pro2 = np.zeros([n_feat_dim, 1])
    sumr_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_full_seq = np.zeros([args.n_trial//args.recording_time+1, 1])


    #ground22 initial
    X_t_unprotect = np.zeros([n_feat_dim, 1])
    sumrX_unprotect = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect = np.zeros([1])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_ground2 = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
#############################################################################################

    #projection3 initial
    X_t_pro3= np.zeros([n_feat_dim, 1])
    sumrX_pro3= np.zeros([n_feat_dim, 1])
    sumr_t_pro3= np.zeros([1])
    V_t_pro3= np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_pro3= np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_pro3 = np.zeros([n_feat_dim, 1])
    sumr_t_pro3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_pro3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_pro3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])


    #ground3 initial
    X_t_ground3 = np.zeros([n_feat_dim, 1])
    sumrX_ground3 = np.zeros([n_feat_dim, 1])
    sumr_t_ground3 = np.zeros([1])
    V_t_ground3 = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_ground3 = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_ground3 = np.zeros([n_feat_dim, 1])
    sumr_t_ground3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_ground3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_ground3_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

################################################################################################
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
    V_t_pro = X_t_pro @ X_t_pro.T + Lambda * eye(n_feat_dim)
    V_t_inv_pro = inv(V_t_pro)
    r_t_pro = rewards_fake[SelectArm_pro, 0] + randn(1)*sigma_noise
    sumrX_pro = r_t_pro * X_t_pro
    sumr_t_pro = r_t_pro


    #############################   initial ground feature arms #########################
    SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_ground]]
    X_t_ground = Projection@X_t
    V_t_ground = X_t_ground @ X_t_ground.T + Lambda * eye(n_feat_dim)
    V_t_inv_ground = inv(V_t_ground)
    r_t_ground = rewards_true[SelectArm_ground, 0] + randn(1, 1)*sigma_noise
    sumrX_ground = r_t_ground * X_t_ground
    sumr_t_ground = r_t_ground



    ##################   initial full ########################
    SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_full]]
    X_t_full = X_t
    V_t_full = X_t_full @ X_t_full.T + Lambda * eye(n_feat_dim)
    V_t_inv_full = inv(V_t_full)
    r_t_full = rewards_fake[SelectArm_full, 0] + randn(1)*sigma_noise
    sumrX_full = r_t_full * X_t_full
    sumr_t_full = r_t_full


    #############################   initial unprotect #########################
    SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_unprotect]]
    X_t_unprotect = Projection@X_t
    V_t_unprotect = X_t_unprotect @ X_t_unprotect.T + Lambda * eye(n_feat_dim)
    V_t_inv_unprotect = inv(V_t_unprotect)
    r_t_unprotect = rewards_true[SelectArm_unprotect, 0] + randn(1, 1)*sigma_noise
    sumrX_unprotect = r_t_unprotect * X_t_unprotect
    sumr_t_unprotect = r_t_unprotect





    #############################   initial linucb #########################
    SelectArm_linucb = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_linucb]]
    X_t_linucb = X_t
    V_t_linucb = X_t_linucb @ X_t_linucb.T + args.lambda_pro * eye(n_feat_dim)
    V_t_inv_linucb = inv(V_t_linucb)
    r_t_linucb = rewards_true[SelectArm_linucb, 0] + randn(1, 1)*sigma_noise
    sumrX_linucb = r_t_linucb * X_t_linucb
    sumr_t_linucb = r_t_linucb



    n_best_pro = np.ones([1, 1])
    n_best_ground = np.ones([1, 1])
    n_best_full = np.ones([1, 1])
    n_best_unprotect = np.ones([1, 1])
    n_best_linucb = np.ones([1, 1])

    regret_pro = np.ones([1, 1])
    regret_ground = np.ones([1, 1])
    regret_full = np.ones([1, 1])
    regret_unprotect = np.ones([1, 1])
    regret_linucb = np.ones([1, 1])


    for rtime in range(args.n_trial):

        # if at each time, arms are update.
        # ProArms = Projection @ Arms
        if args.infinite:
            ########################Please think how to construct arms in the simulations#############
            # n_protect_dim = args.n_protect_dim
            # n_unprotect_dim = args.n_unprotect_dim
            # n_feat_dim = n_protect_dim + n_unprotect_dim

            # Arms = 2 * rand(n_feat_dim, args.n_Arms - n_feat_dim) - 1
            Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1

            # Arms = np.hstack((D_k, Arms))
            ########################Please think how to construct arms in the simulations#############

            rewards_fake = Arms.T @ Theta
            rewards_true = Arms.T @ (Projection @ Theta)
            best_arm = argmax(rewards_true)
            best_reward = rewards_true.ravel()[best_arm]

        ###############################  Projection ##############################
        EstTheta_pro = V_t_inv_pro @ sumrX_pro
        if rand(1) > functiondic[decrease_fun_1](rtime + 1):
            SelectArm_pro = argmax(Arms.T @ (Projection@EstTheta_pro))
            # SelectArm_pro = argmax(Arms.T @ np.vstack((EstTheta_pro[0:-1, :], np.zeros([1, 1]))))
        else:
            # SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_pro = np.random.randint(n_Dk, size=1)[0]
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

        # gender = int(Arms[-1, SelectArm_pro])
        #
        # gender_pro_n_rate[:, gender] += 1
        # gender_pro_true_reward[:, gender] += r_t_true
        # gender_pro_sumr_t[:, gender] += r_t_pro


        ###################################  ground truth_ #############################################
        EstTheta_ground = V_t_inv_ground @ sumrX_ground
        if rand(1) > functiondic[decrease_fun_1](rtime + 1):
            SelectArm_ground = argmax(Arms.T @ (Projection@EstTheta_ground))
        else:
            # SelectArm_ground = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_ground = np.random.randint(n_Dk, size=1)[0]
            # SelectArm_unprotect = np.random.choice(D_k, 1)

        X_t = Arms[:, [SelectArm_ground]]
        X_t_ground = Projection@X_t
        # X_t_ground = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        r_t_true = rewards_true[SelectArm_ground, 0]
        noise = randn(1) * sigma_noise
        r_t_ground = rewards_fake[SelectArm_ground, 0] + noise
        sumr_t_ground = sumr_t_ground + r_t_ground
        n_best_ground += (SelectArm_ground == best_arm)
        regret_ground += best_reward - r_t_true

        V_t_ground = V_t_ground + X_t_ground @ X_t_ground.T
        V_t_inv_ = V_t_inv_ground
        V_t_inv_ground = V_t_inv_ - V_t_inv_ @ X_t_ground @ X_t_ground.T @ V_t_inv_ / (
                    1 + X_t_ground.T @ V_t_inv_ @ X_t_ground)
        sumrX_ground += (r_t_true + noise) * X_t_ground

        # gender = int(Arms[-1, SelectArm_ground])
        # gender_ground_n_rate[:, gender] += 1
        # gender_ground_sumr_t[:, gender] += r_t_ground
        # gender_ground_true_reward[:, gender] += r_t_true

        #####################################  Full   ###########################################
        EstTheta_full = V_t_inv_full @ sumrX_full
        if rand(1) > functiondic[decrease_fun_1](rtime + 1):
            SelectArm_full = argmax(Arms.T @ EstTheta_full)
        else:
            # SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_full = np.random.randint(n_Dk, size=1)[0]
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

        # gender = int(Arms[-1, SelectArm_full])
        # gender_full_n_rate[:, gender] += 1
        # gender_full_sumr_t[:, gender] += r_t_full
        # gender_full_true_reward[:, gender] += r_t_true

        ################################### Only unprotect #############################################
        EstTheta_unprotect = V_t_inv_unprotect @ sumrX_unprotect
        if rand(1) > functiondic[decrease_fun_1](rtime + 1):
            SelectArm_unprotect = argmax(Arms.T @ (Projection@EstTheta_unprotect))
            # SelectArm_unprotect = argmax(Arms.T @ np.vstack((EstTheta_unprotect[0:-1, :], np.zeros([1, 1]))))
        else:
            # SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
            SelectArm_unprotect = np.random.randint(n_Dk, size=1)[0]
            # SelectArm_unprotect = np.random.choice(D_k, 1)

        X_t = Arms[:, [SelectArm_unprotect]]
        # X_t_unprotect = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        X_t_unprotect = Projection@X_t
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

        # gender = int(Arms[-1, SelectArm_unprotect])
        # gender_unprotect_n_rate[:, gender] += 1
        # gender_unprotect_sumr_t[:, gender] += r_t_unprotect
        # gender_unprotect_true_reward[:, gender] += r_t_true








        ##############################    linucb        ################################
        EstTheta_linucb = V_t_inv_linucb @ sumrX_linucb
        # upperbound = args.alpha_linucb * np.sqrt(np.sum(Arms * (V_t_inv_linucb @ Arms), axis=0)).reshape(-1, 1)
        upperbound = args.alpha_pro*(math.log1p(rtime)) * np.sqrt(np.sum(Arms * (V_t_inv_linucb @ Arms), axis=0)).reshape(-1, 1)
        SelectArm_linucb = argmax(Arms.T @ (EstTheta_linucb) + upperbound)

        X_t_linucb = Arms[:, [SelectArm_linucb]]
        r_t_linucb = rewards_fake[SelectArm_linucb, 0] + randn(1) * sigma_noise
        r_t_true = rewards_true[SelectArm_linucb, 0]
        sumr_t_linucb = sumr_t_linucb + r_t_linucb
        n_best_linucb += (SelectArm_linucb == best_arm)
        # regret_linucb += best_reward - Arms[:, [SelectArm_linucb]].T @ (Projection @ Theta)
        regret_linucb += best_reward - r_t_true
        V_t_linucb = V_t_linucb + X_t_linucb @ X_t_linucb.T
        V_t_inv_ = V_t_inv_linucb
        V_t_inv_linucb = V_t_inv_ - V_t_inv_ @ X_t_linucb @ X_t_linucb.T @ V_t_inv_ / (
                1 + X_t_linucb.T @ V_t_inv_ @ X_t_linucb)
        sumrX_linucb += r_t_linucb * X_t_linucb
        #











        ############################   store #######################################
        if (rtime + 1) % args.recording_time == 0:
            sumr_t_pro_seq[ooo + 1, :] = sumr_t_pro.flatten()
            regret_t_pro_seq[ooo + 1, :] = regret_pro.flatten()
            n_best_t_pro_seq[ooo + 1, :] = n_best_pro.flatten()


            sumr_t_ground_seq[ooo + 1, :] = sumr_t_ground.flatten()
            regret_t_ground_seq[ooo + 1, :] = regret_ground.flatten()
            n_best_t_ground_seq[ooo + 1, :] = n_best_ground.flatten()


            sumr_t_full_seq[ooo + 1, :] = sumr_t_full.flatten()
            regret_t_full_seq[ooo + 1, :] = regret_full.flatten()
            n_best_t_full_seq[ooo + 1, :] = n_best_full.flatten()

            sumr_t_unprotect_seq[ooo + 1, :] = sumr_t_unprotect.flatten()
            regret_t_unprotect_seq[ooo + 1, :] = regret_unprotect.flatten()
            n_best_t_unprotect_seq[ooo + 1, :] = n_best_unprotect.flatten()


            sumr_t_linucb_seq[ooo + 1, :] = sumr_t_linucb.flatten()
            regret_t_linucb_seq[ooo + 1, :] = regret_linucb.flatten()
            n_best_t_linucb_seq[ooo + 1, :] = n_best_linucb.flatten()


            ooo += 1

    # return np.sum(sumr_t_pro, axis=1)
    return np.hstack((sumr_t_pro_seq,
                      sumr_t_ground_seq,
                      sumr_t_full_seq,
                      sumr_t_unprotect_seq,
                      sumr_t_linucb_seq,

                      regret_t_pro_seq,
                      regret_t_ground_seq,
                      regret_t_full_seq,
                      regret_t_unprotect_seq,
                      regret_t_linucb_seq,

                      n_best_t_pro_seq,
                      n_best_t_ground_seq,
                      n_best_t_full_seq,
                      n_best_t_unprotect_seq,
                      n_best_t_linucb_seq))


def main():
    # Training settings
    global n_feat_dim
    global args
    # global Arms
    global Lambda
    # global Projection
    # global D_k
    # global n_Dk
    # global Theta
    # global rewards_true
    # global rewards_fake
    mkdirp('./result/syn')
    whichset = 'syn'
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                         help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                         help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=24, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_Arms', type=int, default=45, metavar='N',
                        help='set number of arms (default: 100)')

    parser.add_argument('--n_protect_dim', type=int, default=5, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_unprotect_dim', type=int, default=10, metavar='N',
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
    parser.add_argument('--sigma_noise', type=float, default=0.3, metavar='M',
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
    parser.add_argument('--rand_pro', action='store_false', default=True,
                            help='if specify, we will plot')
    parser.add_argument('--not_mul', action='store_true', default=False,
                        help='if specify, not use multiprocessor')
    # # parser.add_argument('--which_dataset', nargs='+',
    # #                     help='Please input which dataset: MNIST, FashionMNIST, CIFAR10', required=True)
    args = parser.parse_args()
    Lambda = args.lambda_pro

    #We currently choosely randomly a movie that has a rating for the specific user when each time when choosing it.
    #But we also construct D_k below which may be used.

    n_protect_dim = args.n_protect_dim
    n_unprotect_dim = args.n_unprotect_dim
    n_feat_dim = n_protect_dim + n_unprotect_dim

    # Arms= 2 * rand(n_feat_dim, args.n_Arms-n_feat_dim) - 1
    # # Arms = Arms/np.power(np.sum(Arms*Arms, axis=0), 1/2)
    # # pca1 = decomposition.PCA(n_components=n_feat_dim)
    # # pca1.fit(Arms.T)
    # # Arms = pca1.transform(Arms.T).T
    # if args.rand_pro:
    #     tempA = 2*np.random.rand(n_feat_dim, n_unprotect_dim)-1
    #     Projection = tempA@(inv(tempA.T@tempA))@ tempA.T  # get the projection operator
    # else:
    #     Projection = np.diag(np.hstack([np.ones(n_unprotect_dim), np.zeros(n_protect_dim)]))
    #     # Projection = np.flip(Projection, axis=0)
    #
    # # np.save('Projection.npy', Projection)
    # Theta = 2 * rand(n_feat_dim, 1) - 1
    # # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # # n_feat_dim = n_feat_dim + 2
    #
    #
    # # please define the projection operator here.
    # # D_k = eye(n_feat_dim)
    # # fake_D_k = np.random.randint(args.n_Arms, size=n_feat_dim)
    # D_k = 2 * rand(n_feat_dim, n_feat_dim) - 1
    # n_Dk = n_feat_dim
    # # args.n_Arms += n_Dk
    # Arms = np.hstack((D_k, Arms))


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if t1here is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters


    alpha_list = [0.001, 0.01, 0.1, 1, 10, 30, 100]
    result_finaly = []

    starttime = time.time()
    #
    for alpha in alpha_list:
        args.alpha_pro = alpha
        if not args.not_mul:
            with multiprocessing.Pool() as pool:
                result = pool.map(runone, range(0, args.runtimes))
                pool.close()
                pool.join()
                result_finaly = result_finaly + result
                # result_finaly = np.vstack((result_finaly, result)) if result_finaly.size else result
        else:
            result = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, 24])
            for run_id in range(args.runtimes):
                result[run_id, :] = runone(run_id)

    result_finaly = np.asarray(result_finaly)

    print('That took {} seconds'.format(time.time() - starttime))




    np.save('./result/'+whichset+'/result.npy', result_finaly)
    scipy.io.savemat('./result/'+whichset+'/result.mat',
                     mdict={'result': result_finaly})



    print(1)


if __name__ == '__main__':
    main()









