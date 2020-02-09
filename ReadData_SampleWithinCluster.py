import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
from sklearn import decomposition

def getDataSet(args):
    # Training settings

    age_cat = [0, 20, 40, 100]
    zip_code_cat = 10
    datasetpath100k = './dataset/ml-100k/'

    # first read movie information from u.item
    # normalize the genres
    filemoviegenres = pd.read_csv(datasetpath100k+'u.genre', delimiter='|', names = ['genre', 'genreindex'])
    moviegenres = filemoviegenres.genre.tolist()
    otherattriname = ['movie_id', 'movie_title', 'release_date', 'video_release_date', 'imdb_url']
    movieattributes = otherattriname + moviegenres
    filemovieitem = pd.read_csv(datasetpath100k +'u.item', delimiter='|', names = movieattributes, encoding="iso-8859-1")
    filemovieitem.iloc[:, 5:] = filemovieitem.iloc[:, 5:].apply(pd.to_numeric)
    filemovieitem.iloc[:, 5:] = filemovieitem.iloc[:, 5:].div(filemovieitem.iloc[:, 5:].sum(axis=1), axis=0)
    movies = filemovieitem.set_index('movie_id').iloc[:, 4:]

    #here we use pca to reduce movie dimension.
    pca1 = decomposition.PCA(n_components=3)
    moviedata = np.asarray(movies)
    pca1.fit(moviedata)
    newmoviedata = pca1.transform(moviedata)
    movies = pd.DataFrame(newmoviedata, index=movies.index)

    # then read user information from u.user
    # categorize the age(8 classes), occupation and zip_code(10 classes)
    # also normalize the vector
    userattributes = ['user_id', 'age', 'gender', 'occupation', 'zip_code']
    fileusers = pd.read_csv(datasetpath100k+'u.user', delimiter='|', names = userattributes)
    fileusers['gender'].replace('f', 1, inplace=True)
    fileusers['gender'].replace('m', 0, inplace=True)
    fileusers['occupation'] = fileusers['occupation'].astype("category")
    fileusers['age'] = pd.to_numeric(fileusers['age'], errors='coerce')
    fileusers['age'] = pd.cut(fileusers['age'].fillna(-1),\
                                                        age_cat, right=False)
    fileusers['zip_code'] = pd.to_numeric(fileusers['zip_code'], errors='coerce').fillna(-1)
    fileusers['zip_range'] = pd.cut(fileusers['zip_code'], \
                            [-10,0,10000,20000, 30000,40000,50000,60000,70000,80000,90000,100000],right=False)
    # fileusers = pd.get_dummies(fileusers)
    # fileusers.drop("zip_code", axis=1, inplace=True)
    # cols = fileusers.columns
    # fileusers[cols] = fileusers[cols].apply(pd.to_numeric, errors='coerce')
    # # a = pd.concat([fileusers[['user_id']], fileusers.apply(lambda x: x.iloc[1:].squeeze()/sum(x.iloc[1:]), axis=1)], sort=false, axis = 1)
    # fileusers.iloc[:, 1:] = fileusers.iloc[:, 1:].div(fileusers.iloc[:, 1:].sum(axis=1), axis=0)
    # users = fileusers.set_index('user_id')
    # users = users.drop(users.columns[-11:], axis=1) # use this to drop zip_range

    fileusers.drop("zip_code", axis=1, inplace=True)
    fileusers.drop("zip_range", axis=1, inplace=True)
    occup = pd.get_dummies(fileusers.loc[:, ['occupation']])
    fileusers.drop("occupation", axis=1, inplace=True)
    fileusers = pd.get_dummies(fileusers)
    occupdata = np.asarray(occup)
    pca2 = decomposition.PCA(n_components=3)
    pca2.fit(occupdata)
    newoccupdata = pca2.transform(occupdata)
    fileusers.iloc[:, 1:] = fileusers.iloc[:, 1:].div(fileusers.iloc[:, 1:].sum(axis=1), axis=0)
    users = pd.concat([fileusers, pd.DataFrame(newoccupdata)], axis=1)
    users = users.set_index('user_id')


    # read u.data (the rating pairs)
    filerating = pd.read_csv(datasetpath100k+'u.data', delimiter='\t', names = ['user_id', 'movie_id', 'rating', 'timestamp'])
    # filerating.set_index(['user_id', 'movie_id'], inplace=True)
    filerating = filerating.apply(pd.to_numeric)
    filerating.set_index(['user_id', 'movie_id'], inplace=True)


    # now run k mean to cluster users to n cluster, and add one column to users.
    kmeans = KMeans(n_clusters=args.n_clusters).fit(users)
    labels = kmeans.labels_.astype(int)
    users['cluster_id'] = labels
    # users.set_index('cluster_ind', append=True, inplace=True)
    ###!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! BE  CAREFUL, now user only contains gender  !!!!!!!!!!!!!
    users = users.loc[:, ['gender_M', 'gender_F', 'cluster_id']]

    return filerating, users, movies
if __name__ == '__main__':
    parsertemp = argparse.ArgumentParser(description='test ReadData')
    parsertemp.add_argument('--n_clusters', type=int, default=5, metavar='n',
                        help='set the number of clusters (default: 5)')
    argstemp = parsertemp.parse_args()
    rating, users, movies = getDataSet(argstemp)
    rating.to_csv('rating.csv')
    users.to_csv('users.csv')
    movies.to_csv('movies.csv')
