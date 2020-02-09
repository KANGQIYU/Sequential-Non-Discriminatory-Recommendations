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
import multiprocessing
import time

def runone(runtime, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection):
    ooo = 0
    epsilon_f = lambda x: min(1, args.alpha*934/x)
    epsilon_i = lambda x: min(1, args.alpha*934/x**(1.0/3))
    epsilon_o = lambda x: min(1, args.alpha*934/x**(1.0/2))
    functiondic = {'i': epsilon_i, 'f': epsilon_f, 'o': epsilon_o}
    sumr_t_pro = np.zeros([args.n_trial//args.recording_time+1, args.n_clusters])
    initsample_user = group_cluster.apply(lambda x: x.sample(1)).reset_index(level=0, drop=True)
    X_t = np.zeros([n_feat_dim, args.n_clusters])
    sumrX = np.zeros([n_feat_dim, args.n_clusters])
    sumr_t = np.zeros([1, args.n_clusters])
    V_t = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    V_t_inv = np.zeros([n_feat_dim, n_feat_dim, args.n_clusters])
    #initial
    for i in range(args.n_clusters):
        movie_0 = rating.loc[initsample_user.iloc[[i], :].index[0]].sample(1)
        infor_movie = np.asarray(movies.loc[movie_0.index])
        infor_user = np.asarray(initsample_user.iloc[i, 0:-1])
        X_t_0 = np.hstack([infor_movie[0], infor_user]).reshape(-1, 1)
        # X_t_0 = np.hstack([infor_movie, np.matlib.repmat(infor_user, infor_movie.shape[0], 1)]).T
        X_t[:, [i]] = X_t_0
        V_t[:, :, i] = X_t_0 @ X_t_0.T + Lambda * eye(n_feat_dim)
        V_t_inv[:, :, i] = inv(V_t[:, :, i])
        r_t = rating.loc[initsample_user.iloc[[i],:].index[0], movie_0.index[0]].iloc[0]
        sumrX[:, [i]] = r_t*X_t_0
        sumr_t[:, i] = r_t
    EstTheta = np.zeros([n_feat_dim, args.n_clusters])
    for time in range(args.n_trial):
        # user_t = np.random.choice(n_users, 1)[0]
        user_t = users.sample(1)
        cluster_id = user_t.iloc[:, -1].iloc[0]
        # movie_t = rating.loc[user_t].sample(10)
        movie_t = rating.loc[user_t.index[0]]
        infor_movie = np.asarray(movies.loc[movie_t.index])
        infor_user = np.asarray(user_t.iloc[:, 0:-1])
        Arms = np.hstack([infor_movie, np.matlib.repmat(infor_user, infor_movie.shape[0], 1)]).T
        ProArms = Projection@Arms
        EstTheta[:, [cluster_id]] = V_t_inv[:, :, cluster_id] @ sumrX[:, [cluster_id]]

        if rand(1) > functiondic['f'](time+1):
            SelectArm = argmax(ProArms.T @ EstTheta[:, [cluster_id]])
        else:
            SelectArm = np.random.randint((Arms.shape[1]), size=1)[0]
        try:
            SelectArm_id = movie_t.index[SelectArm]
            X_t = Arms[:, [SelectArm]]
            r_t = rating.loc[(user_t.index[0], SelectArm_id)].iloc[0]
            sumr_t[:, cluster_id] = sumr_t[:, cluster_id] + r_t
        except:
            print(X_t)
        V_t[:, :, cluster_id] = V_t[:, :, cluster_id] + X_t @ X_t.T
        V_t_inv_ = V_t_inv[:, :, cluster_id]
        V_t_inv[:, :, cluster_id] = V_t_inv_ - V_t_inv_ @ X_t @ X_t.T @ V_t_inv_ / (1 + X_t.T @ V_t_inv_ @ X_t)
        sumrX[:, [cluster_id]] += r_t * X_t
        if (time+1)%args.recording_time == 0:
            sumr_t_pro[ooo+1, :] = np.ravel(sumr_t)
            ooo += 1
    # return np.sum(sumr_t_pro, axis=1)
    np.save('a_'+str(runtime)+'.npy', np.sum(sumr_t_pro, axis=1))
    # print()
def main():
# Training settings
    parser = argparse.ArgumentParser(description='Projection Simulation')
    parser.add_argument('--n_clusters', type=int, default=5, metavar='n',
                        help='set the number of clusters (default: 5)')
    parser.add_argument('--n_trial', type=int, default=10000, metavar='N',
                         help='set number of trials(default: 10000)')
    parser.add_argument('--recording_time', type=int, default=100, metavar='N',
                         help='record the reward every recording_time times')
    parser.add_argument('--runtimes', type=int, default=4, metavar='N',
                        help='set number of runtimes(default: 10)')
    parser.add_argument('--alpha', type=float, default=0.5, metavar='M',
                        help='set parameter alpha')
    parser.add_argument('--decrease_fun', type=str, default='i', metavar='M',
                        help='set decreasing function')
    # parser.add_argument('--levels', nargs='+', type=int, help='<Required> Set flag', required=True)
    # parser.add_argument('--new_sample', action='store_true', default=False,
    #                     help='if this is new-start, set with new wrong labels will generated')
    # # parser.add_argument('--which_dataset', nargs='+',
    # #                     help='Please input which dataset: MNIST, FashionMNIST, CIFAR10', required=True)
    args = parser.parse_args()
    Lambda = 0.1
    rating, users, movies = getDataSet(args)
    rating.iloc[:, 0] = rating.iloc[:, 0].div(5)

    #We currently choosely randomly a movie that has a rating for the specific user when each time when choosing it.
    #But we also construct D_k below which may be used.

    n_users = len(users)
    n_movies = len(movies)
    n_dim = len(list(users))+len(list(movies))
    n_feat_dim = n_dim - 1

    # please define the projection operator here.
    # here we first define it as all user information
    Projection = np.diag(np.hstack([np.ones(len(list(movies))), np.zeros(len(list(users))-1)]))


    D_k = np.zeros([n_users, 2])
    n_Dk = D_k.shape[0]
    epsilon_f = lambda x: min(1, args.alpha*n_Dk/x)
    epsilon_i = lambda x: min(1, args.alpha*n_Dk/x**(1.0/3))
    epsilon_o = lambda x: min(1, args.alpha*n_Dk/x**(1.0/2))
    functiondic = {'i': epsilon_i, 'f': epsilon_f, 'o': epsilon_o}
    for i in users.index:
        D_k[i-1, 0] = i
        D_k[i-1, 1] = rating.loc[i].sample(1).index[0]

    #to speed up the simulation, we can run different cluster at the same time
    #First get the size for each cluster, sorry, may be implement this later.
    group_cluster = users.groupby(by=['cluster_ind'])
    clusterasize = np.asarray(group_cluster.count().iloc[:, 0])


    # Please note here it seems we could drop the user information, but it's not
    #correct. I'm still think if there is other way to do:
    #1. drop all the users information for protection purpose (discrimination)
    #2. reduce dimension first
    #3. reduce some features dimension but reserve gender information
    manager = multiprocessing.Manager()
    # sumr_t_pro = np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
#   sumr_t_pro = manager.np.zeros([args.runtimes, args.n_trial//args.recording_time+1, args.n_clusters])
    starttime = time.time()
    # processes = []
    # for run_id in range(args.runtimes):
    #     p = multiprocessing.Process(target=runone, args=(run_id, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection))
    #     processes.append(p)
    #     p.start()
    # for process in processes:
    #     process.join()
    #
    pool = multiprocessing.Pool(4)
    iterable=[(0, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection),
          (1, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection ),
          (2, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection),
          (3, args, group_cluster, n_feat_dim, rating, movies, users, Lambda, Projection)]
    pool.starmap(runone, iterable)
    pool.close()
    print('That took {} seconds'.format(time.time() - starttime))

        # X_0 = Arms[:, [0]]
            # X_t = X_0
            # r_t = X_t.T @ Theta + randn(1, 1) * sigma_noise
            # sumrX = r_t * X_t
            # sumr_t = ProArms[:, [0]].T @ Theta
            #
            # V_t = X_t @ X_t.T + Lambda * eye(Dim)
            # V_t_inv = inv(V_t)
            # Totalr_T = 0
            # for t in range(2, T):
            #     EstTheta = V_t_inv @ sumrX
            #
            #     if rand(1) > min(1.0, Alpha * NumArms / t):
            #         SelectArm = argmax(ProArms.T @ EstTheta)
            #     else:
            #         SelectArm = np.random.randint(0, NumArms - 1)
            #
            #     try:
            #         X_t = Arms[:, [SelectArm]]
            #         r_t = X_t.T @ Theta + randn(1, 1) * sigma_noise
            #         sumr_t = sumr_t + ProArms[:, [SelectArm]].T @ Theta
            #     except:
            #         print(X_t)
            #
            #     PullTime[SelectArm, IndTheta] += 1
            #
            #     V_t = V_t + X_t @ X_t.T
            #     V_t_inv = V_t_inv - V_t_inv @ X_t @ X_t.T @ V_t_inv / (1 + X_t.T @ V_t_inv @ X_t)
            #     sumrX += r_t * X_t
            #
            # Totalr_T += sumr_t
            #
            # cluster_t = 0[-1]
            # EstTheta
    print(1)

if __name__ == '__main__':
    main()



