import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import argparse
from sklearn import decomposition
import math
from sklearn import preprocessing


def getWine(args):
    # Training settings

    age_cat = [0, 20, 40, 100]
    zip_code_cat = 10
    datasetpath100k = './dataset/wine/'

    # first read movie information from u.item
    # normalize the genres
    filered = pd.read_csv(datasetpath100k+'winequality-red.csv', delimiter=';')
    filewhite = pd.read_csv(datasetpath100k+'winequality-white.csv', delimiter=';')

    cols = filered.columns
    filered[cols] = filered[cols].apply(pd.to_numeric, errors='coerce')

    cols = filewhite.columns
    filewhite[cols] = filewhite[cols].apply(pd.to_numeric, errors='coerce')

    # white_index = filewhite.index
    # sample = np.random.choice(np.random.np.asarray(list(white_index)), math.floor(len(white_index)*0.7), replace=False).tolist()

    red_quality = filered.iloc[:, [-1]].copy()
    white_quality = filewhite.iloc[:, [-1]].copy()
    # white_quality.iloc[sample] = white_quality.iloc[sample]+1

    wine = pd.concat([filewhite, filered]).reset_index(drop=True)
    quality = pd.concat([white_quality, red_quality]).reset_index(drop=True)

    x = wine.values  # returns a numpy array
    # min_max_scaler = preprocessing.MinMaxScaler()
    # x_scaled = min_max_scaler.fit_transform(x)
    # wine = pd.DataFrame(x_scaled, index=wine.index, columns=wine.columns)
    x = np.hstack((x[:, 0:-1], np.vstack((np.zeros([len(filewhite),1 ]), np.ones([len(filered), 1])))))
    x_scaled = preprocessing.scale(x)
    wine = pd.DataFrame(x_scaled, index=wine.index, columns=wine.columns)


    # wine = wine.loc[:, ['volatile acidity', 'chlorides']]

    kmeans = KMeans(n_clusters=args.n_clusters).fit(wine)
    labels = kmeans.labels_.astype(int)

    #here we use pca to reduce dimension.
    pca1 = decomposition.PCA(n_components=3)
    winedata = np.asarray(wine)
    pca1.fit(winedata)
    newwinedata = pca1.transform(winedata)
    wine = pd.DataFrame(newwinedata, index=wine.index)


    # kmeans = KMeans(n_clusters=args.n_clusters).fit(wine)
    # labels = kmeans.labels_.astype(int)

    #
    wine['constant'] = 1
    typeindicator = np.hstack((np.zeros([len(filewhite)]), np.ones([len(filered)])))
    # wine['type'] = typeindicator


    wine['cluster_id'] = labels


    return wine, quality, typeindicator
if __name__ == '__main__':
    parsertemp = argparse.ArgumentParser(description='test ReadData')
    parsertemp.add_argument('--n_clusters', type=int, default=5, metavar='n',
                        help='set the number of clusters (default: 5)')
    argstemp = parsertemp.parse_args()
    wine, quality, typeindicator = getWine(argstemp)
    wine.to_csv('./wine.csv')
    quality.to_csv('./quality.csv')
