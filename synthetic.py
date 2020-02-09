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
import math

def runone(run_id, Theta, n_feat_dim,  args, Arms, Lambda, Projection, functiondic_pro, functiondic_unprotect, functiondic_full, D_k):
    print(run_id)
    ooo = 0
    sigma_noise = args.sigma_noise
    # ProArms = Projection @ Arms
    decrease_fun = args.decrease_fun

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

    #only moive initial
    X_t_unprotect = np.zeros([n_feat_dim, 1])
    sumrX_unprotect = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect = np.zeros([1])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim])
    EstTheta_unprotect = np.zeros([n_feat_dim, 1])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    regret_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])
    n_best_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 1])

    ##################   initial projection ########################
    SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
    X_t = Arms[:, [SelectArm_pro]]
    X_t_pro = X_t
    V_t_pro = X_t_pro @ X_t_pro.T + Lambda * eye(n_feat_dim)
    V_t_inv_pro = inv(V_t_pro)
    r_t_pro = X_t.T @ Theta + randn(1, 1) * sigma_noise
    sumrX_pro = r_t_pro * X_t_pro
    sumr_t_pro = r_t_pro
    ####################    initial full arms, they are the same as pro#######################
    X_t_full = X_t_pro.copy()
    V_t_full = V_t_pro.copy()
    V_t_inv_full = V_t_inv_pro.copy()
    sumrX_full = sumrX_pro.copy()
    sumr_t_full = sumr_t_pro.copy()

    gender_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_full_seq = np.zeros([args.n_trial // args.recording_time + 1, 6])
    gender_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 6])
    gender_ground_seq = np.zeros([args.n_trial//args.recording_time+1, 6])

    if args.run_unprotect:
        #############################   initial unprotected feature arms #########################
        SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
        X_t = Arms[:, [SelectArm_unprotect]]
        X_t_unprotect = Projection@X_t
        V_t_unprotect = X_t_unprotect @ X_t_unprotect.T + Lambda * eye(n_feat_dim)
        V_t_inv_unprotect = inv(V_t_unprotect)
        r_t_unprotect = X_t.T @ Theta + randn(1, 1) * sigma_noise
        sumrX_unprotect = r_t_unprotect * X_t_unprotect
        sumr_t_unprotect = r_t_unprotect

    reward = Arms.T @ (Projection @ Theta)
    best_arm = argmax(reward)
    best_reward = reward.ravel()[best_arm]
    n_best_pro = np.zeros([1, 1])
    n_best_full = np.zeros([1, 1])
    n_best_unprotect = np.zeros([1, 1])
    regret_pro = np.zeros([1, 1])
    regret_full = np.zeros([1, 1])
    regret_unprotect = np.zeros([1, 1])

    for rtime in range(args.n_trial):

        # if at each time, arms are update.
        # ProArms = Projection @ Arms
        if args.infinite:
            Arms = 2 * rand(n_feat_dim, args.n_Arms) - 1
            Arms = Arms / np.power(np.sum(Arms * Arms, axis=0), 1 / 2)
            reward = Arms.T@(Projection@Theta)
            best_arm = argmax(reward)
            best_reward = reward.ravel()[best_arm]
        ###############################  Projection  ##############################
        EstTheta_pro = V_t_inv_pro @ sumrX_pro
        if rand(1) > functiondic_pro[decrease_fun](rtime + 1):
            SelectArm_pro = argmax(Arms.T @ (Projection@EstTheta_pro))
        else:
            SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
            # SelectArm_pro = np.random.choice(D_k, 1)

        X_t_pro = Arms[:, [SelectArm_pro]]
        r_t_pro = X_t_pro.T @ Theta + randn(1, 1)*sigma_noise
        sumr_t_pro = sumr_t_pro + r_t_pro
        n_best_pro += (SelectArm_pro==best_arm)
        regret_pro += best_reward-Arms[:, [SelectArm_pro]].T@ (Projection@Theta)

        V_t_pro = V_t_pro + X_t_pro @ X_t_pro.T
        V_t_inv_ = V_t_inv_pro
        V_t_inv_pro = V_t_inv_ - V_t_inv_ @ X_t_pro @ X_t_pro.T @ V_t_inv_ / (1 + X_t_pro.T @ V_t_inv_ @ X_t_pro)
        sumrX_pro += r_t_pro * X_t_pro

        #####################################  Full   ###########################################
        EstTheta_full = V_t_inv_full @ sumrX_full
        if rand(1) > functiondic_full[decrease_fun](rtime + 1):
            SelectArm_full = argmax(Arms.T @ EstTheta_full)
        else:
            SelectArm_full = np.random.randint((Arms.shape[1]), size=1)[0]
            # SelectArm_full = np.random.choice(D_k, 1)

        X_t_full = Arms[:, [SelectArm_full]]
        r_t_full = X_t_full.T@Theta + randn(1, 1)*sigma_noise
        sumr_t_full = sumr_t_full + r_t_full
        n_best_full += (SelectArm_full == best_arm)
        regret_full += best_reward - Arms[:, [SelectArm_full]].T @ (Projection @ Theta)
        V_t_full = V_t_full + X_t_full @ X_t_full.T
        V_t_inv_ = V_t_inv_full
        V_t_inv_full = V_t_inv_ - V_t_inv_ @ X_t_full @ X_t_full.T @ V_t_inv_ / (
                    1 + X_t_full.T @ V_t_inv_ @ X_t_full)
        sumrX_full += r_t_full * X_t_full

        ################################### Only unprotect #############################################
        if args.run_unprotect:
            EstTheta_unprotect = V_t_inv_unprotect @ sumrX_unprotect
            if rand(1) > functiondic_unprotect[decrease_fun](rtime + 1):
                SelectArm_unprotect = argmax(Arms.T @ (Projection@EstTheta_unprotect))
            else:
                SelectArm_unprotect = np.random.randint((Arms.shape[1]), size=1)[0]
                # SelectArm_unprotect = np.random.choice(D_k, 1)

            X_t = Arms[:, [SelectArm_unprotect]]
            X_t_unprotect = Projection@X_t
            r_t_unprotect = X_t.T@Theta + randn(1, 1)*sigma_noise
            sumr_t_unprotect = sumr_t_unprotect + r_t_unprotect
            n_best_unprotect += (SelectArm_unprotect == best_arm)
            regret_unprotect += best_reward - Arms[:, [SelectArm_unprotect]].T @ (Projection @ Theta)

            V_t_unprotect = V_t_unprotect+ X_t_unprotect @ X_t_unprotect.T
            V_t_inv_ = V_t_inv_unprotect
            V_t_inv_unprotect = V_t_inv_ - V_t_inv_ @ X_t_unprotect @ X_t_unprotect.T @ V_t_inv_ / (
                        1 + X_t_unprotect.T @ V_t_inv_ @ X_t_unprotect)
            sumrX_unprotect += r_t_unprotect * X_t_unprotect

    ##################################   store #######################################
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
            ooo += 1

    # return np.sum(sumr_t_pro, axis=1)
    return np.hstack((sumr_t_pro_seq,
                      sumr_t_full_seq,
                      sumr_t_unprotect_seq,
                      regret_t_pro_seq,
                      regret_t_full_seq,
                      regret_t_unprotect_seq,
                      n_best_t_pro_seq,
                      n_best_t_full_seq,
                      n_best_t_unprotect_seq,
                      gender_pro_seq,
                      gender_full_seq,
                      gender_unprotect_seq,
                      gender_ground_seq
                      ))

def main():
# Training settings
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=5, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=100000, metavar='N',
                         help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                         help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=300, metavar='N',
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
    # # parser.add_argument('--which_dataset', nargs='+',
    # #                     help='Please input which dataset: MNIST, FashionMNIST, CIFAR10', required=True)
    args = parser.parse_args()
    Lambda = args.lambda_pro

    #We currently choosely randomly a movie that has a rating for the specific user when each time when choosing it.
    #But we also construct D_k below which may be used.

    n_protect_dim = args.n_protect_dim
    n_unprotect_dim = args.n_unprotect_dim
    n_feat_dim = n_protect_dim + n_unprotect_dim

    Arms= 2 * rand(n_feat_dim, args.n_Arms) - 1
    Arms = Arms/np.power(np.sum(Arms*Arms, axis=0), 1/2)
    pca1 = decomposition.PCA(n_components=n_feat_dim)
    pca1.fit(Arms.T)
    Arms = pca1.transform(Arms.T).T
    tempid = np.random.choice(Arms.shape[1], math.floor(Arms.shape[1]/2), replace=False)
    templastrow = np.zeros([1, Arms.shape[1]])
    templastrow[0, tempid] = 1
    Arms = np.vstack((Arms, np.ones([1, Arms.shape[1]]), templastrow))

    if args.rand_pro:
        tempA = np.random.rand(n_feat_dim, n_feat_dim)
        Projection = tempA@(inv(tempA.T@tempA))@ tempA.T  # get the projection operator
    else:
        Projection = np.diag(np.hstack([np.ones(n_unprotect_dim), np.zeros(n_protect_dim)]))
        Projection = np.flip(Projection, axis=0)

    np.save('Projection.npy', Projection)
    # Theta = 2 * rand(n_feat_dim, 1) - 1
    Theta = np.vstack((2 * rand(n_feat_dim, 1) - 1, 2*np.ones([1, 1]), np.zeros([1, 1])))
    n_feat_dim = n_feat_dim + 2
    Projection = eye(n_feat_dim)

    # please define the projection operator here.
    # D_k = eye(n_feat_dim)
    fake_D_k = np.random.randint(args.n_Arms, size=n_feat_dim)
    D_k = Arms[:, fake_D_k]
    n_Dk = D_k.shape[1]


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




    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if there is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters

    sumr_t_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    sumr_t_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    sumr_t_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    regret_t_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    regret_t_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    regret_t_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    n_best_t_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    n_best_t_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])
    n_best_t_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1])

    result = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 9])

    starttime = time.time()
    for run_id in range(args.runtimes):
        result[run_id, :] =  runone(run_id, Theta, n_feat_dim, args, Arms, Lambda, Projection, functiondic_pro,
                           functiondic_unprotect, functiondic_full, fake_D_k)
    print('That took {} seconds'.format(time.time() - starttime))

    sumr_t_pro_seq_runs = result[:, :, 0]
    sumr_t_full_seq_runs = result[:, :, 1]
    sumr_t_unprotect_seq_runs = result[:, :, 2]
    regret_t_pro_seq_runs = result[:, :, 3]
    regret_t_full_seq_runs = result[:, :, 4]
    regret_t_unprotect_seq_runs = result[:, :, 5]
    n_best_t_pro_seq_runs = result[:, :, 6]
    n_best_t_full_seq_runs = result[:, :, 7]
    n_best_t_unprotect_seq_runs =result[:, :, 8]

    n_runtime = sumr_t_full_seq_runs.shape[0]
    sumr_full = np.sum(sumr_t_full_seq_runs, axis=0)/n_runtime
    sumr_pro = np.sum(sumr_t_pro_seq_runs, axis=0)/n_runtime
    sumr_unprotect = np.sum(sumr_t_unprotect_seq_runs, axis=0)/n_runtime
    regret_pro = np.sum(regret_t_pro_seq_runs, axis=0)/n_runtime
    regret_full = np.sum(regret_t_full_seq_runs, axis=0)/n_runtime
    regret_unprotect = np.sum(regret_t_unprotect_seq_runs, axis=0)/n_runtime
    n_best_pro = np.sum(n_best_t_pro_seq_runs, axis=0)/n_runtime
    n_best_full = np.sum(n_best_t_full_seq_runs, axis=0)/n_runtime
    n_best_unprotect = np.sum(n_best_t_unprotect_seq_runs, axis=0)/n_runtime

    scipy.io.savemat('./result/syn/result_matlab_full_syn.mat',\
                     mdict={'sumr_t_pro_seq_runs': sumr_t_pro_seq_runs, \
                            'sumr_t_full_seq_runs': sumr_t_full_seq_runs, \
                            'sumr_t_unprotect_seq_runs': sumr_t_unprotect_seq_runs,\
                            'regret_t_pro_seq_runs': regret_t_pro_seq_runs,\
                            'regret_t_full_seq_runs': regret_t_full_seq_runs,
                            'regret_t_unprotect_seq_runs': regret_t_unprotect_seq_runs,\
                            'n_best_t_pro_seq_runs': n_best_t_pro_seq_runs,\
                            'n_best_t_full_seq_runs': n_best_t_full_seq_runs,
                            'n_best_t_unprotect_seq_runs': n_best_t_unprotect_seq_runs,\
                            })
    np.save('./result/syn/sumr_t_pro_seq_runs.npy', sumr_t_pro_seq_runs)


    np.save('./result/syn/sumr_t_pro_seq_runs', sumr_t_pro_seq_runs)
    np.save('./result/syn/sumr_t_full_seq_runs', sumr_t_full_seq_runs)
    np.save('./result/syn/sumr_t_unprotect_seq_runs', sumr_t_unprotect_seq_runs)
    np.save('./result/syn/regret_t_pro_seq_runs', regret_t_pro_seq_runs)
    np.save('./result/syn/regret_t_full_seq_runs', regret_t_full_seq_runs)
    np.save('./result/syn/regret_t_unprotect_seq_runs', regret_t_unprotect_seq_runs)
    np.save('./result/syn/n_best_t_pro_seq_runs', n_best_t_pro_seq_runs)
    np.save('./result/syn/n_best_t_full_seq_runs', n_best_t_full_seq_runs)
    np.save('./result/syn/n_best_t_unprotect_seq_runs', n_best_t_unprotect_seq_runs)
    p.save('./result/syn/result.npy', result)
    if args.plot:
        plt.figure(1)
        plt.title('reward')
        plt.plot(sumr_full, 'r-', sumr_pro, 'g--', sumr_unprotect, 'b-.')
        plt.savefig('./reward.pdf', bbox_inches='tight')
        plt.figure(2)
        plt.title('regret')
        plt.plot(regret_full, 'r-', regret_pro, 'g--', regret_unprotect, 'b-.')
        plt.savefig('./regret.pdf', bbox_inches='tight')
        plt.figure(3)
        plt.title('n_best')
        plt.plot(n_best_full, 'r-', n_best_pro, 'g--', n_best_unprotect, 'b-.')
        plt.show()
        plt.savefig('./n_best.pdf', bbox_inches='tight')

    print(1)


if __name__ == '__main__':
    main()









