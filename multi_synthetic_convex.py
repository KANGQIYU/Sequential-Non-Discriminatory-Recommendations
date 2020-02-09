from sklearn import decomposition
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
from cvxopt import solvers, matrix, spdiag, log, div, sqrt
import cvxopt as ct
import sympy
import cvxpy as cp
import numpy as np
import time
import dccp
from numpy import linalg as LA
import matplotlib.pyplot as plt



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

solvers.options['show_progress'] = False


def lin_entropy_constraint(c,n):
    c_m = matrix(-c)
    m = 1
    def F(x=None, z=None):
        if x is None: return m, matrix(1.0/n, (n,1))
        if min(x)<=0: return None
        Df = log(x)+1
        val = sum(x.T * log(x))-5
        if z is None: return val, Df.T
        H = spdiag(z[0] * div(1, x))
        return val, Df.T, H
    return solvers.cpl(c_m, F, G, h, dims)['x']


# def lin_entropy_constraint(c,n):
#     c_m = matrix(-c)
#     m = 1
#     def F(x=None, z=None):
#         tempa = matrix(np.asarray([1, 10, 1]).reshape(-1,1))
#         if x is None: return m, matrix(1.0/n, (n,1))
#         if min(x)<=0: return None
#         Df = ct.mul(tempa, (log(x)+1))
#         val = sum((ct.mul(tempa, x)).T * log(x))-5
#         if z is None: return val, Df.T
#         H = spdiag(z[0] * div(tempa, x))
#         return val, Df.T, H
#     return solvers.cpl(c_m, F, G, h, dims)['x']

# def lin_entropy_constraint(c,n):
#     c_m = matrix(-c)
#     I = matrix(0.0, (n, n))
#     I[::+ 1] = 1.0
#     m = 1
#     G = matrix([matrix(0.0, (1, n)), I])
#     h = matrix([1.0] + n * [0.0])
#     dims = {'l': 0, 'q': [n+1], 's': []}
#     def F(x=None, z=None):
#         if x is None: return m, matrix(0, (n, 1))
#         if max(abs(x)) >= 1.0: return None
#         u = 1 - x ** 2
#         # print(x)
#         val = -sum(log(u))-0.1
#         Df = div(2 * x, u).T
#         if z is None: return val, Df
#         H = spdiag(2 * z[0] * div(1 + u ** 2, u ** 2))
#         return val, Df, H
#     return solvers.cpl(c_m, F, G, h, dims)['x']

# np.sqrt(np.sum(Arms * (V_t_inv_linucb @ Arms), axis=0)).
#fuck it's not convex
def ucb_entropy_constraint(A, constant, c, n):
    c_m = matrix(-c)
    A = matrix(A)
    constant = -constant
    m = 1
    def F(x=None, z=None):
        if x is None: return m, matrix(1.0/n, (n,1))
        if min(x)<=0: return None
        temp = sqrt(x.T * A * x)
        Df = matrix([ c_m+ constant/temp*A*x, log(x)+1])
        val = matrix([c_m.T* x + constant*temp, sum(x.T * log(x))-5])
        if z is None: return val.T, Df.T
        H = z[0]* (temp*A-A*x*x.T*A/temp)/(temp**2)+spdiag(z[1] * div(1, x))
        return val.T, Df.T, H.T
    return solvers.cp(F, G, h, dims)['x']


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
    global G, h, dims
    ########################Please think how to construct arms in the simulations#############
    n_protect_dim = args.n_protect_dim
    n_unprotect_dim = args.n_unprotect_dim
    n_feat_dim = n_protect_dim + n_unprotect_dim
    sigma_noise = args.sigma_noise
    # ProArms = Projection @ Arms
    decrease_fun = args.decrease_fun
    convex_const = args.convex_const

    lin_convex_constraint = {'entropy': lin_entropy_constraint}

    D_k = np.zeros((n_feat_dim, n_feat_dim))
    for i in range(n_feat_dim):
        temmpxx = sympy.Symbol('x')
        a = np.random.rand(n_feat_dim - 1, 1)
        b = float(sympy.solve(temmpxx * sympy.log(temmpxx) - (5 - a.T @ np.log(a)), temmpxx)[0])
        mask = np.ones(n_feat_dim,bool)
        mask[i] = False
        D_k[mask,i] = a[:, 0]
        D_k[i, i] = b

    if convex_const == 'entropy':
        c = cp.Parameter((n_feat_dim, 1))
        constant = cp.Parameter(nonneg=True)
        sqrtw = cp.Parameter((n_feat_dim, 1))
        v = cp.Parameter((n_feat_dim, n_feat_dim))
        # Construct the problem.
        x = cp.Variable((n_feat_dim, 1))
        obj = cp.Maximize(c.T @ x + cp.norm((cp.multiply(sqrtw, v * x))))
        x.value = 1 / np.ones((n_feat_dim, 1))
        constraints = [cp.sum(-cp.entr(x)) <= 5, x >= 0]
        # constraints = [cp.log_sum_exp(x)<=10, x>=0]
        prob = cp.Problem(obj, constraints)


    if args.rand_pro:
        tempA = 2*np.random.rand(n_feat_dim, n_unprotect_dim)-1
        Projection = tempA@(inv(tempA.T@tempA))@ tempA.T  # get the projection operator
    else:
        Projection = np.diag(np.hstack([np.ones(n_unprotect_dim), np.zeros(n_protect_dim)]))
        # Projection = np.flip(Projection, axis=0)

    # np.save('Projection.npy', Projection)
    Theta = 2*rand(n_feat_dim, 1)-1
    # Theta = np.asarray([5.2, 0.42, -2]).reshape(-1, 1)
    # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # n_feat_dim = n_feat_dim + 2


    n_Dk = n_feat_dim
    best_arm = np.array(lin_convex_constraint[convex_const](Projection@Theta, n_feat_dim))
    best_reward = (Projection@Theta).T@best_arm

    print(run_id)
    ooo = 0


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
    X_t_pro = D_k[:, np.random.randint(n_Dk, size=1)[0]].reshape(-1,1)
    V_t_pro = X_t_pro @ X_t_pro.T + Lambda * eye(n_feat_dim)
    V_t_inv_pro = inv(V_t_pro)
    r_t_pro = X_t_pro.T @ Theta + randn(1) * sigma_noise
    sumrX_pro = r_t_pro * X_t_pro
    ####################    initial full arms, they are the same as pro#######################
    X_t_full = X_t_pro.copy()
    V_t_full = V_t_pro.copy()
    r_t_full = r_t_pro.copy()
    V_t_inv_full = V_t_inv_pro.copy()
    sumrX_full = sumrX_pro.copy()


    #############################   initial unprotected feature arms #########################
    X_t_unprotect = Projection@X_t_pro
    V_t_unprotect = X_t_unprotect @ X_t_unprotect.T + Lambda * eye(n_feat_dim)
    V_t_inv_unprotect = inv(V_t_unprotect)
    r_t_unprotect = r_t_pro.copy()
    sumrX_unprotect = r_t_unprotect * X_t_unprotect



    #############################   initial linucb #########################
    X_t_linucb = X_t_pro.copy()
    V_t_linucb = V_t_pro.copy()
    V_t_inv_linucb = inv(V_t_linucb)
    r_t_linucb = r_t_pro.copy()
    sumrX_linucb = r_t_linucb * X_t_linucb

    #############################   initial linucbpro #########################
    X_t_linucbpro = X_t_linucb.copy()
    V_t_linucbpro = V_t_linucb.copy()
    V_t_inv_linucbpro = inv(V_t_linucbpro)
    r_t_linucb = r_t_pro.copy()
    sumrX_linucbpro = sumrX_linucb.copy



    regret_pro = np.ones([1, 1])
    regret_full = np.ones([1, 1])
    regret_unprotect = np.ones([1, 1])
    regret_ground = np.ones([1, 1])
    regret_linucb = np.ones([1, 1])
    regret_linucbpro = np.ones([1, 1])


    for rtime in range(args.n_trial):

        # if at each time, arms are update.
        # ProArms = Projection @ Arms
        # if args.infinite:
        #     ########################Please think how to construct arms in the simulations#############
        #     # n_protect_dim = args.n_protect_dim
        #     # n_unprotect_dim = args.n_unprotect_dim
        #     # n_feat_dim = n_protect_dim + n_unprotect_dim
        #
        #     # Arms = 2 * rand(n_feat_dim, args.n_Arms - n_feat_dim) - 1
        #     Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
        #
        #     # Arms = np.hstack((D_k, Arms))
        #     ########################Please think how to construct arms in the simulations#############
        #
        #     rewards_fake = Arms.T @ Theta
        #     rewards_true = Arms.T @ (Projection @ Theta)
        #     best_arm = argmax(rewards_true)
        #     best_reward = rewards_true.ravel()[best_arm]

        ###############################  Projection  ##############################
        EstTheta_pro = V_t_inv_pro @ sumrX_pro
        if rand(1) > functiondic_pro[decrease_fun](rtime + 1):
            X_t_pro = np.array(lin_convex_constraint[convex_const](Projection@EstTheta_pro, n_feat_dim)).reshape(-1,1)
        else:
            X_t_pro = D_k[:, np.random.randint(n_Dk, size=1)[0]].reshape(-1, 1)
            # SelectArm_pro = np.random.randint(n_Dk, size=1)[0]
            # SelectArm_pro = np.random.choice(D_k, 1)


        # r_t_pro = X_t_pro.T @ Theta + randn(1, 1)*sigma_noise
        r_t_true = (Projection@X_t_pro).T @ Theta
        r_t_pro = X_t_pro.T @ Theta+ randn(1)*sigma_noise

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

        #####################################  Full   ###########################################
        EstTheta_full = V_t_inv_full @ sumrX_full
        if rand(1) > functiondic_full[decrease_fun](rtime + 1):
            X_t_full = np.array(lin_convex_constraint[convex_const](EstTheta_full, n_feat_dim)).reshape(-1,1)
        else:
            X_t_full = D_k[:, np.random.randint(n_Dk, size=1)[0]].reshape(-1,1)
            # SelectArm_full = np.random.randint(n_Dk, size=1)[0]
            # SelectArm_full = np.random.choice(D_k, 1)

        r_t_full = X_t_full.T @ Theta + randn(1)*sigma_noise
        r_t_true = (Projection@X_t_full).T @ Theta
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
        if rand(1) > functiondic_unprotect[decrease_fun](rtime + 1):
            X_t = np.array(lin_convex_constraint[convex_const](EstTheta_unprotect, n_feat_dim)).reshape(-1,1)
        else:
            X_t= D_k[:, np.random.randint(n_Dk, size=1)[0]].reshape(-1,1)
            # SelectArm_unprotect = np.random.randint(n_Dk, size=1)[0]
            # SelectArm_unprotect = np.random.choice(D_k, 1)

        X_t_unprotect = Projection @ X_t
        r_t_unprotect = X_t.T @ Theta + randn(1) * sigma_noise
        r_t_true = (Projection@X_t).T @ Theta
        # regret_unprotect += best_reward - Arms[:, [SelectArm_unprotect]].T @ (Projection @ Theta)
        regret_unprotect += best_reward - r_t_true

        V_t_unprotect = V_t_unprotect+ X_t_unprotect @ X_t_unprotect.T
        V_t_inv_ = V_t_inv_unprotect
        V_t_inv_unprotect = V_t_inv_ - V_t_inv_ @ X_t_unprotect @ X_t_unprotect.T @ V_t_inv_ / (
                    1 + X_t_unprotect.T @ V_t_inv_ @ X_t_unprotect)
        sumrX_unprotect += r_t_unprotect * X_t_unprotect


        ###############################    linucb        ################################
        # EstTheta_linucb = V_t_inv_linucb @ sumrX_linucb
        # # upperbound = args.alpha_linucb*np.log(rtime+1)* np.sqrt(np.sum(Arms * (V_t_inv_linucb @ Arms), axis=0)).reshape(-1, 1)
        # # SelectArm_linucb = argmax(Arms.T @ (EstTheta_linucb) + upperbound)
        # if convex_const == 'entropy':
        #     w, vv = LA.eigh(V_t_inv_linucb)
        #     sqrtw.value = np.sqrt(abs(w)).reshape(-1, 1)
        #     c.value = EstTheta_linucb
        #     v.value = vv
        #     constant.value = 1#args.alpha_linucb*np.log(rtime+1)*n_feat_dim
        #     prob.solve(method='dccp')
        #     X_t_linucb = x.value.reshape(-1,1)
        #
        # r_t_linucb = X_t_linucb.T@Theta + randn(1) * sigma_noise
        # r_t_true =  (Projection@X_t_linucb).T @ Theta
        # regret_linucb += best_reward - r_t_true
        # V_t_linucb = V_t_linucb + X_t_linucb @ X_t_linucb.T
        # V_t_inv_ = V_t_inv_linucb
        # V_t_inv_linucb = V_t_inv_ - V_t_inv_ @ X_t_linucb @ X_t_linucb.T @ V_t_inv_ / (
        #         1 + X_t_linucb.T @ V_t_inv_ @ X_t_linucb)
        # sumrX_linucb += r_t_linucb * X_t_linucb

        # ###############################    linucbpro        ################################
        # EstTheta_linucbpro = V_t_inv_linucbpro @ sumrX_linucbpro
        # upperbound = args.alpha_linucbpro*(1+math.sqrt(math.log(rtime+1, 2))) * np.sqrt(np.sum(Arms * (V_t_inv_linucbpro @ Arms), axis=0)).reshape(-1, 1)
        # SelectArm_linucbpro = argmax(Arms.T @ (Projection @ EstTheta_linucbpro) + upperbound)
        #
        # X_t_linucbpro = Arms[:, [SelectArm_linucbpro]]
        # # X_t_linucbpro = np.vstack((X_t[0:-1, :], np.zeros([1, 1])))
        # r_t_linucbpro = rewards_fake[SelectArm_linucbpro, 0] + randn(1) * sigma_noise
        # r_t_true = rewards_true[SelectArm_linucbpro, 0]
        # sumr_t_linucbpro = sumr_t_linucbpro + r_t_linucbpro
        # n_best_linucbpro += (SelectArm_linucbpro == best_arm)
        # # regret_linucbpro += best_reward - Arms[:, [SelectArm_linucbpro]].T @ (Projection @ Theta)
        # regret_linucbpro += best_reward - r_t_true
        # V_t_linucbpro = V_t_linucbpro + X_t_linucbpro @ X_t_linucbpro.T
        # V_t_inv_ = V_t_inv_linucbpro
        # V_t_inv_linucbpro = V_t_inv_ - V_t_inv_ @ X_t_linucbpro @ X_t_linucbpro.T @ V_t_inv_ / (
        #         1 + X_t_linucbpro.T @ V_t_inv_ @ X_t_linucbpro)
        # sumrX_linucbpro += r_t_linucbpro * X_t_linucbpro
        #
        # gender = int(Arms[-1, SelectArm_linucbpro])

        ############################   store #######################################
        if (rtime + 1) % args.recording_time == 0:
            regret_t_pro_seq[ooo + 1, :] = regret_pro.flatten()

            ##why flatten() #
            # for future discrimination compare like male female
            regret_t_full_seq[ooo + 1, :] = regret_full.flatten()

            regret_t_unprotect_seq[ooo + 1, :] = regret_unprotect.flatten()

            regret_t_ground_seq[ooo + 1, :] = regret_ground.flatten()

            regret_t_linucb_seq[ooo + 1, :] = regret_linucb.flatten()

            regret_t_linucbpro_seq[ooo + 1, :] = regret_linucbpro.flatten()

            ooo += 1

    # return np.sum(sumr_t_pro, axis=1)
    return np.hstack((regret_t_pro_seq,
                      regret_t_full_seq,
                      regret_t_unprotect_seq,
                      regret_t_ground_seq,
                      regret_t_linucb_seq,
                      regret_t_linucbpro_seq,

                      ))


def main():
    # Training settings
    global n_feat_dim
    global args
    # global Arms
    global Lambda
    global I, G, h, dims
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
    parser.add_argument('--runtimes', type=int, default=12, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_Arms', type=int, default=200, metavar='N',
                        help='set number of arms (default: 100)')

    parser.add_argument('--n_protect_dim', type=int, default=2, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--n_unprotect_dim', type=int, default=2, metavar='N',
                        help='set number of arms (default: 100)')

    parser.add_argument('--alpha_pro', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_unprotect', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_full', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_ground', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_linucb', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_linucbpro', type=float, default=0.1, metavar='M',
                        help='set parameter alpha')

    parser.add_argument('--lambda_pro', type=float, default=0.5, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--sigma_noise', type=float, default=0.5, metavar='M',
                        help='set noise variance')

    parser.add_argument('--decrease_fun', type=str, default='i', metavar='M',
                        help='set decreasing function')

    parser.add_argument('--convex_const', type=str, default='entropy', metavar='M',
                       help='set convex constraints')

    parser.add_argument('--infinite', action='store_true', default=False,
                         help='if specify, we will generate new arms every time')
    parser.add_argument('--rand_pro', action='store_true', default=False,
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

    Arms= 2 * rand(n_feat_dim, args.n_Arms-n_feat_dim) - 1
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
    Theta = 2 * rand(n_feat_dim, 1) - 1
    # Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    # n_feat_dim = n_feat_dim + 2

    # c = np.asarray([[-1., -1.]])
    Theta = 2 * rand(n_feat_dim, 1) - 1

    # c = -np.random.rand(n_feat_dim, 1)
    # c = matrix(c)

    I = matrix(0.0, (n_feat_dim, n_feat_dim))
    I[::n_feat_dim + 1] = 1.0
    # G = matrix([-I, matrix(0.0, (1,n)), I])
    # h = matrix(n*[0.0] + [5.0] + n*[0.0])
    # dims = {'l': n, 'q': [n+1], 's': []}

    G = -I
    h = matrix(n_feat_dim * [0.0])
    dims = {'l': n_feat_dim, 'q': [], 's': []}


    # please define the projection operator here.
    # D_k = eye(n_feat_dim)
    # fake_D_k = np.random.randint(args.n_Arms, size=n_feat_dim)
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
        result = np.zeros([args.runtimes, args.n_trial // args.recording_time + 1, 6])
        for run_id in range(args.runtimes):
            result[run_id, :] = runone(run_id)



    print('That took {} seconds'.format(time.time() - starttime))


    # result=np.load('./result/syn/result.npy')



    regret_t_pro_seq_runs = result[:, :, 0]
    regret_t_full_seq_runs = result[:, :, 1]
    regret_t_unprotect_seq_runs = result[:, :, 2]
    regret_t_ground_seq_runs = result[:, :, 3]
    regret_t_linucb_seq_runs = result[:, :, 4]
    regret_t_linucbpro_seq_runs = result[:, :, 5]



    scipy.io.savemat('./result/'+whichset+'/result_matlab_full_'+whichset+'.mat',
                     mdict={ 'regret_t_pro_seq_runs': regret_t_pro_seq_runs,
                            'regret_t_full_seq_runs': regret_t_full_seq_runs,
                            'regret_t_unprotect_seq_runs': regret_t_unprotect_seq_runs,
                            'regret_t_ground_seq_runs': regret_t_ground_seq_runs,
                            'regret_t_linucb_seq_runs': regret_t_linucb_seq_runs,
                            'regret_t_linucbpro_seq_runs': regret_t_linucbpro_seq_runs,
    })



    np.save('./result/'+whichset+'/regret_t_pro_seq_runs.npy', regret_t_pro_seq_runs)
    np.save('./result/'+whichset+'/regret_t_full_seq_runs.npy', regret_t_full_seq_runs)
    np.save('./result/'+whichset+'/regret_t_unprotect_seq_runs.npy', regret_t_unprotect_seq_runs)
    np.save('./result/'+whichset+'/regret_t_ground_seq_runs.npy', regret_t_ground_seq_runs)
    np.save('./result/'+whichset+'/regret_t_linucb_seq_runs.npy', regret_t_linucb_seq_runs)
    np.save('./result/'+whichset+'/regret_t_linucbpro_seq_runs.npy', regret_t_linucbpro_seq_runs)




    np.save('./result/'+whichset+'/result.npy', result)
    scipy.io.savemat('./result/'+whichset+'/result.mat',
                     mdict={'result': result})






    print(1)


if __name__ == '__main__':
    main()









