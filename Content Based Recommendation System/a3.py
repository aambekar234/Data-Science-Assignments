# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 10:37:00 2018

@author: Aambekar
"""
from collections import Counter, defaultdict
import math
import numpy as np
import os
import pandas as pd
import re
from scipy.sparse import csr_matrix
import urllib.request
import zipfile

def download_data():
    """ DONE. Download and unzip data.
    """
    url = 'https://www.dropbox.com/s/h9ubx22ftdkyvd5/ml-latest-small.zip?dl=1'
    urllib.request.urlretrieve(url, 'ml-latest-small.zip')
    zfile = zipfile.ZipFile('ml-latest-small.zip')
    zfile.extractall()
    zfile.close()
    
    
def tokenize_string(my_string):
    """ DONE. You should use this in your tokenize function.
    """
    return re.findall('[\w\-]+', my_string.lower())

def tokenize(movies):
    mtokens = []
    for i in movies['genres'].tolist():
        mtokens.append(tokenize_string(i))
    movies['tokens'] = mtokens
    return movies

def featurize(movies):
    

    num_movies = len(movies)
    csr_matrix_list = []
    f_freq = defaultdict(lambda: 0);
    f_freq_list = []
    
    for token_l in movies['tokens']:
        tmp = defaultdict(lambda: 0)
        for tk in token_l:      
            tmp[tk] += 1
        f_freq_list.append(tmp)
        
        for token in token_l:     
            f_freq[token] += 1
    
    vocab = defaultdict(lambda: 0)
    c = 0
    f_freq_sorted = sorted(f_freq)
    
    for key in f_freq_sorted:
        vocab[key] = f_freq_sorted.index(key)
    
        
    
    for i in range(num_movies):
        
        column = []
        row = []
        
        ffl_item = f_freq_list[i]
        mk = ffl_item[max(ffl_item, key = ffl_item.get)]
        data = []
        
        for key in ffl_item:
            if key  in vocab:
                column.append(vocab[key])
                tfidf = ffl_item[key] / mk * math.log(num_movies / f_freq[key],10)
                data.append(tfidf)
                row.append(0)
                
        m = csr_matrix((data, (row, column)), shape = (1, len(vocab)))
        csr_matrix_list.append(m)
        
    movies['features'] = csr_matrix_list
    return movies, vocab
        
        
    
def train_test_split(ratings):
    """DONE.
    Returns a random split of the ratings matrix into a training and testing set.
    """
    test = set(range(len(ratings))[::1000])
    train = sorted(set(range(len(ratings))) - test)
    test = sorted(test)
    return ratings.iloc[train], ratings.iloc[test]

def cosine_sim(a, b):
    
    num = a.dot(b.T).toarray()[0][0]
    den = math.sqrt(sum([i*i for i in  a.toarray()[0]])) * math.sqrt(sum([j*j for j in  b.toarray()[0]]))
    
    return num/den
    
def mean_absolute_error(predictions, ratings_test):
    """DONE.
    Return the mean absolute error of the predictions.
    """
    return np.abs(predictions - np.array(ratings_test.rating)).mean()


def make_predictions(movies, ratings_train, ratings_test):
    
    ratings = []
    
    for index, row in ratings_test.iterrows():
        
        user_id = row['userId']
        movie_id = row['movieId']
        
        a = movies.loc[movies.movieId == movie_id].squeeze()['features']
        other = ratings_train.loc[ratings_train.userId == user_id]
        
        total = 0
        weighted_r = 0.0
        status = 0
        
        for u_i, u_r in other.iterrows():
            b = movies.loc[movies.movieId == u_r['movieId']].squeeze()['features']
            ans = cosine_sim(a, b)
            
            if ans > 0:
                weighted_r += ans * u_r['rating']
                total += ans
                status = ans
                
        if status > 0:
            ratings.append(weighted_r / total)
        else:
            ratings.append(other['rating'].mean())


    return np.array(ratings)




def main():
    download_data()
    path = 'ml-latest-small'
    ratings = pd.read_csv(path + os.path.sep + 'ratings.csv')
    movies = pd.read_csv(path + os.path.sep + 'movies.csv')
    movies = tokenize(movies)
    movies, vocab = featurize(movies)
    print('vocab:')
    print(sorted(vocab.items())[:10])
    ratings_train, ratings_test = train_test_split(ratings)
    print('%d training ratings; %d testing ratings' % (len(ratings_train), len(ratings_test)))
    predictions = make_predictions(movies, ratings_train, ratings_test)
    print('error=%f' % mean_absolute_error(predictions, ratings_test))
    print(predictions[:10])


if __name__ == '__main__':
    main()
