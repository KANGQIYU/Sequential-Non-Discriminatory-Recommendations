import numpy as np
from numpy.linalg import inv
from numpy import zeros
from numpy.random import rand
from numpy.random import randn
from numpy import eye
from numpy import argmax
import argparse
from ReadData import getDataSet
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

args = 0
movie = 0
true_rating = 0
n_feat_dim = 3
# quality = pd.DataFrame([])
true_rating = pd.DataFrame([])
Projection = 0
group_cluster = 0

def runone(run_id, args, group_cluster,  n_feat_dim, true_rating, movies, users, Lambda, Projection):
    armslastrow = np.zeros([1,100000])
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

    ooo = 0
    print(run_id)
    # decrease_fun = args.decrease_fun
    # initsample_user = group_cluster.apply(lambda x: x.sample(1)).reset_index(level=0, drop=True)
    #
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



    #only moive initial
    X_t_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumrX_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect = np.zeros([1, args.n_clusters])
    V_t_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv_unprotect = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    EstTheta_unprotect = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])



    gender_pro_sumr_t = np.zeros([1, 2])
    gender_pro_n_rate = np.zeros([1, 2])
    gender_full_sumr_t = np.zeros([1, 2])
    gender_full_n_rate = np.zeros([1, 2])
    gender_unprotect_sumr_t = np.zeros([1, 2])
    gender_unprotect_n_rate = np.zeros([1, 2])

    ##  first two column gender_sum_r_t, middle column number
    gender_pro_seq = np.zeros([args.n_trial // args.recording_time + 1, 4])
    gender_full_seq = np.zeros([args.n_trial // args.recording_time + 1, 4])
    gender_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, 4])


    # reward = Arms.T @ (Projection @ Theta)
    # best_arm = argmax(reward)
    # best_reward = reward.ravel()[best_arm]
    n_best_pro = np.zeros([1, args.n_clusters])
    n_best_full = np.zeros([1, args.n_clusters])
    n_best_unprotect = np.zeros([1, args.n_clusters])
    regret_true_pro = np.zeros([1, args.n_clusters])
    regret_true_full = np.zeros([1, args.n_clusters])
    regret_true_unprotect = np.zeros([1, args.n_clusters])
    reward_true_pro = np.zeros([1, args.n_clusters])
    reward_true_full = np.zeros([1, args.n_clusters])
    reward_true_unprotect = np.zeros([1, args.n_clusters])

    n_best_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_pro_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_full_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_unprotect_seq = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])

    num_arms = len(users)
    cluster = np.ones((num_arms, args.n_clusters), dtype=bool)
    tempuser = []
    temprating = []
    tempratingasarray = []
    moviewasarray = np.asarray(movies)
    usersasarray = np.asarray(users)
    for i in range(args.n_clusters):
        cluster_id = i
        cluster[:, cluster_id] = users['cluster_id'] == cluster_id
        tempuser.append(users[cluster[:, cluster_id]])
        temprating.append(true_rating.loc[(tempuser[cluster_id].index.tolist()), :].index.tolist())
        tempratingasarray.append(np.asarray(temprating[cluster_id]))

        leng = len(temprating[cluster_id])
        if leng >= args.n_Arms:
            sampleid = np.random.choice(leng, args.n_Arms, replace=False)
            rewards_and_tlastrow = true_rating.loc[temprating[sampleid]]
        else:
            rewards_true = true_rating.loc[temprating[sampleid]]
            rewards_and_tlastrow = true_rating[cluster_id]
        # wine_t = wine[wine['cluster_id'] == cluster_id].sample(args.n_Arms)
        # rewards_true = true_quality.loc[wine_t.index]
        rewards_true = np.asarray(rewards_and_tlastrow.rating)
        tlastrow = np.asarray(rewards_and_tlastrow.lastrow)
        # Arms = np.asarray(wine_t.iloc[:, 0:-1]).T
        infor_movie = moviewasarray[tempratingasarray[:, 1]]
        infor_user = usersasarray[tempratingasarray[:, 1], 0:-1]
        Arms = np.hstack((infor_movie, np.matlib.repmat(infor_user, infor_movie.shape[0], 1), tlastrow)).T
        gender = int(tlastrow[:, SelectArm_pro])
        ### random select one arm, same for all methods
        SelectArm_pro = np.random.randint((Arms.shape[1]), size=1)[0]
        # rewards_fake = np.asarray(quality.loc[wine_t.index])


        #### 0 is white, 1 is red
        # gender = 0 if wine_t.iloc[SelectArm_pro]['type'] < 0.1 else 1
        ##################   initial projection arms AND  full arms, they are the same######
        X_t_0 = Arms[:, [SelectArm_pro]]
        # r_t = quality.iloc[SelectArm_id_pro, 0]
        if gender == 1:
            r_t = rewards_true[SelectArm_pro, 0] + np.random.choice([0, 1], 1, p=[1-args.pr, args.pr])
        else:
            r_t = rewards_true[SelectArm_pro, 0]
        # r_t_pro = rewards_true[SelectArm_pro, 0]
        X_t_pro[:, [i]] = X_t_0
        V_t_pro[:, :, i] = X_t_0 @ X_t_0.T + args.lamba_pro * eye(n_feat_dim)
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



def main():
# Training settings
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=1, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                         help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                         help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=2, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--alpha_pro', type=float, default=0.01, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_unprotect', type=float, default=0.01, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--alpha_full', type=float, default=0.01, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--decrease_fun', type=str, default='o', metavar='M',
                        help='set decreasing function')
    # parser.add_argument('--levels', nargs='+', type=int, help='<Required> Set flag', required=True)
    # parser.add_argument('--new_sample', action='store_true', default=False,
    #                     help='if this is new-start, set with new wrong labels will generated')
    # # parser.add_argument('--which_dataset', nargs='+',
    # #                     help='Please input which dataset: MNIST, FashionMNIST, CIFAR10', required=True)
    parser.add_argument('--run_unprotect', action='store_true', default=False,
                         help='if specify, we will run unprotected version in the simulation')
    parser.add_argument('--infinite', action='store_true', default=False,
                         help='if specify, we will generate new arms every time')
    parser.add_argument('--plot', action='store_true', default=False,
                         help='if specify, we will plot')
    args = parser.parse_args()
    Lambda = 1

    true_rating, users, movies = getDataSet(args)
    # origin_rating = rating.copy()
    # movies['constant'] = 1
    # maleindex = users[users.loc[:, 'gender'] > 0.1].index
    # femaleindex = users[users.loc[:, 'gender'] < 0.1].index
    # sample = np.random.choice(np.random.np.asarray(list(maleindex)), math.floor(len(maleindex)*0.9), replace=False).tolist()
    # rating.loc[(sample, slice(None)), 'rating'] = rating.loc[(sample, slice(None)), 'rating'] - 8

    # rating.loc[(list(maleindex), slice(None)), 'rating'] = rating.loc[(list(maleindex), slice(None)), 'rating'] + 0.3
    # rating.loc[(list(femaleindex), slice(None)), 'rating'] = rating.loc[(list(femaleindex), slice(None)), 'rating'] - 0.3
    # rating.iloc[:, 0] = rating.iloc[:, 0].div(5)
    # rating.iloc[:, 0] = rating.iloc[:, 0]


    #We currently choosely randomly a movie that has a rating for the specific user each time when choosing it.
    #But we also construct D_k below which may be used.

    n_users = len(users)
    n_movies = len(movies)
    # n_protect_dim = len(list(users))-2
    n_protect_dim = 1
    n_movie_dim = len(list(movies))
    n_unproctect_dim = n_movie_dim + 2
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


    # for i in users.index:
    #     D_k[i-1, 0] = i
    #     D_k[i-1, 1] = rating.loc[i].sample(1).index[0]

    #to speed up the simulation, we can run different cluster at the same time
    #First get the size for each cluster, sorry, may be implement this later.
    # group_cluster = users.groupby(by=['cluster_ind'])
    # clusterasize = np.asarray(group_cluster.count().iloc[:, 0])


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if there is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    #4. considering when just use occupation or gender to group clusters
    sumr_t_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    sumr_t_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    sumr_t_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    n_best_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    regret_true_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    reward_true_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])


    gender_pro_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 4])
    gender_full_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 4])
    gender_unprotect_seq_runs = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, 4])
    starttime = time.time()
    for run_id in range(args.runtimes):
        runone(run_id, args, group_cluster,  n_feat_dim, true_rating, movies, users, Lambda, Projection)
    print('That took {} seconds'.format(time.time() - starttime))

    r_full = np.sum(np.sum(sumr_t_full_seq_runs, axis=2), axis=0)/args.runtimes
    r_pro= np.sum(np.sum(sumr_t_pro_seq_runs, axis=2), axis=0)/args.runtimes
    r_unprotect = np.sum(np.sum(sumr_t_unprotect_seq_runs, axis=2), axis=0)/args.runtimes

    reward_true_full = np.sum(np.sum(reward_true_full_seq_runs, axis=2), axis=0)/args.runtimes
    reward_true_pro = np.sum(np.sum(reward_true_pro_seq_runs, axis=2), axis=0)/args.runtimes
    reward_true_unprotect = np.sum(np.sum(reward_true_unprotect_seq_runs, axis=2), axis=0)/args.runtimes


    regret_true_full = np.sum(np.sum(regret_true_full_seq_runs, axis=2), axis=0) / args.runtimes
    regret_true_pro = np.sum(np.sum(regret_true_pro_seq_runs, axis=2), axis=0) / args.runtimes
    regret_true_unprotect = np.sum(np.sum(regret_true_unprotect_seq_runs, axis=2), axis=0) / args.runtimes

    n_best_true_full = np.sum(np.sum(n_best_full_seq_runs, axis=2), axis=0) / args.runtimes
    n_best_true_pro = np.sum(np.sum(n_best_pro_seq_runs, axis=2), axis=0) / args.runtimes
    n_best_true_unprotect = np.sum(np.sum(n_best_unprotect_seq_runs, axis=2), axis=0) / args.runtimes

    temp_full = gender_full_seq_runs[:, 1:, [0,1]]/gender_full_seq_runs[:, 1:, [2,3]]
    female_full = np.sum(temp_full[:,:,0], axis=0)/args.runtimes
    male_full = np.sum(temp_full[:,:,1], axis=0)/args.runtimes

    temp_pro = gender_pro_seq_runs[:, 1:, [0, 1]] / gender_pro_seq_runs[:, 1:, [2, 3]]
    female_pro = np.sum(temp_pro[:, :, 0], axis=0) / args.runtimes
    male_pro = np.sum(temp_pro[:, :, 1], axis=0) / args.runtimes

    temp_unprotect = gender_unprotect_seq_runs[:, 1:, [0, 1]] / gender_unprotect_seq_runs[:, 1:, [2, 3]]
    female_unprotect = np.sum(temp_unprotect[:, :, 0], axis=0) / args.runtimes
    male_unprotect = np.sum(temp_unprotect[:, :, 1], axis=0) / args.runtimes

    scipy.io.savemat('./result_matlab_real.mat',\
                     mdict={'r_full': r_full, 'r_pro': r_pro, 'r_unprotect': r_unprotect,\
                'female_full': female_full, 'male_full': male_full, 'female_pro': female_pro,\
                'male_pro': male_pro, 'female_unprotect': female_unprotect, 'male_unprotect': male_unprotect,\
                'reward_true_full':reward_true_full,\
                'reward_true_pro':reward_true_pro,\
                'regret_true_unprotect':reward_true_unprotect, \
                'regret_true_full':regret_true_full, \
                'regret_true_pro':regret_true_pro, \
                'regret_true_unprotect':regret_true_unprotect, \
                'n_best_true_full':n_best_true_full, \
                'n_best_true_pro':n_best_true_pro, \
                'n_best_true_unprotect':n_best_true_unprotect \
                })
    scipy.io.savemat('./result_matlab_full_real.mat',\
                     mdict={'sumr_t_pro_seq_runs': sumr_t_pro_seq_runs, \
                            'sumr_t_full_seq_runs': sumr_t_full_seq_runs, \
                            'sumr_t_unprotect_seq_runs': sumr_t_unprotect_seq_runs,\
                            'gender_pro_seq_runs': gender_pro_seq_runs,\
                            'gender_full_seq_runs': gender_full_seq_runs,
                            'gender_unprotect_seq_runs': gender_unprotect_seq_runs, \
                            'reward_true_full_seq_runs': reward_true_full_seq_runs, \
                            'reward_true_pro_seq_runs': reward_true_pro_seq_runs, \
                            'regret_true_unprotect_seq_runs': reward_true_unprotect_seq_runs, \
                            'regret_true_full_seq_runs': regret_true_full_seq_runs, \
                            'regret_true_pro_seq_runs': regret_true_pro_seq_runs, \
                            'regret_true_unprotect_seq_runs': regret_true_unprotect_seq_runs, \
                            'n_best_full_seq_runs': n_best_full_seq_runs, \
                            'n_best_pro_seq_runs': n_best_pro_seq_runs, \
                            'n_best_unprotect_seq_runs': n_best_unprotect_seq_runs \
                            })
    np.save('sumr_t_pro_seq_runs.npy', sumr_t_pro_seq_runs)
    np.save('sumr_t_full_seq_runs.npy', sumr_t_full_seq_runs)
    np.save('sumr_t_unprotect_seq_runs.npy', sumr_t_unprotect_seq_runs)
    np.save('gender_pro_seq_runs.npy', gender_pro_seq_runs)
    np.save('gender_full_seq_runs.npy', gender_full_seq_runs)
    np.save('gender_unprotect_seq_runs.npy', gender_unprotect_seq_runs)

    np.save('reward_true_full_seq_runs', reward_true_full_seq_runs)
    np.save('reward_true_pro_seq_runs',reward_true_pro_seq_runs)
    np.save('regret_true_unprotect_seq_runs', reward_true_unprotect_seq_runs)
    np.save('regret_true_full_seq_runs', regret_true_full_seq_runs)
    np.save('regret_true_pro_seq_runs', regret_true_pro_seq_runs)
    np.save('regret_true_unprotect_seq_runs', regret_true_unprotect_seq_runs)
    np.save('n_best_full_seq_runs', n_best_full_seq_runs)
    np.save('n_best_pro_seq_runs', n_best_pro_seq_runs)
    np.save('n_best_unprotect_seq_runs', n_best_unprotect_seq_runs)

    if args.plot:
        plt.figure(1)
        plt.title('rating')
        plt.plot(r_full, 'r-', r_pro, 'g--', r_unprotect, 'b-.')
        plt.figure(2)
        plt.title('gender_difference')
        plt.plot(female_full, 'r-', male_full, 'r-.', female_pro, 'g-', male_pro, 'g-.',\
                 female_unprotect, 'b-', male_unprotect, 'b-.')
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



