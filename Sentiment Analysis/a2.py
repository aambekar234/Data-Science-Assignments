from collections import Counter, defaultdict
from itertools import chain, combinations
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from scipy.sparse import csr_matrix
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
import string
import tarfile
import urllib.request

def download_data():
    """ Download and unzip data.
    DONE ALREADY.
    """
    url = 'https://www.dropbox.com/s/xk4glpk61q3qrg2/imdb.tgz?dl=1'
    urllib.request.urlretrieve(url, 'imdb.tgz')
    tar = tarfile.open("imdb.tgz")
    tar.extractall()
    tar.close()
    
def read_data(path):
    """
    Walks all subdirectories of this path and reads all
    the text files and labels.
    DONE ALREADY.
    Params:
      path....path to files
    Returns:
      docs.....list of strings, one per document
      labels...list of ints, 1=positive, 0=negative label.
               Inferred from file path (i.e., if it contains
               'pos', it is 1, else 0)
    """
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'pos', '*.txt'))])
    data = [(1, open(f).readlines()[0]) for f in sorted(fnames)]
    fnames = sorted([f for f in glob.glob(os.path.join(path, 'neg', '*.txt'))])
    data += [(0, open(f).readlines()[0]) for f in sorted(fnames)]
    data = sorted(data, key=lambda x: x[1])
    return np.array([d[1] for d in data]), np.array([d[0] for d in data])


def filter_stopwords(token_list, fname='stopwords.txt'):
    content = []
    with open(fname) as f:
        content = f.readlines()
        content = [x.strip() for x in content]
    
    stop_words_set = set(content)
    token_set = set(token_list)
    new_token_list = token_set - (token_set & stop_words_set)
    return list(new_token_list)

def tokenize(doc, keep_internal_punct=False):
    tokens = []
    if keep_internal_punct == False:
        tokens = np.array(re.findall('[\w_]+', doc.lower()))
    else:
        tokens = np.array(re.findall('[\w_][^\s]*[\w_]|[\w_]', doc.lower()))
        
    return tokens

def token_features(tokens, feats):
    c = Counter(tokens)
    for key in c:
        feats['token=%s' %key] = c[key]

def token_pair_features(tokens, feats, k=3):
    for i in range(0, len(tokens) - k + 1):
        a = tokens[i : i + k]
        for co in combinations(a,2):
            key = 'token_pair=%s__%s' % (co[0],co[1])
            if key in feats:
                feats[key] = feats[key] + 1
            else:
                feats[key] = 1
            


neg_words = set(['bad', 'hate', 'horrible', 'worst', 'boring'])
pos_words = set(['awesome', 'amazing', 'best', 'good', 'great', 'love', 'wonderful'])

def lexicon_features(tokens, feats):
    n = 0;
    p = 0;
    for token in tokens:
        if token.lower() in neg_words:
            n += 1
        if token.lower() in pos_words:
            p += 1
    feats['neg_words'] = n
    feats['pos_words'] = p
    


def featurize(tokens, feature_fns):
    feats = defaultdict(lambda: 0)
    for fn in feature_fns:
        fn(tokens, feats)
    return (list(sorted(feats.items())))
            

def vectorize(tokens_list, feature_fns, min_freq, vocab=None):

    features_list = []
    feature_freq = defaultdict(lambda: 0)
    for tokens in tokens_list:
        features_list.append(featurize(tokens, feature_fns))
                
    if vocab == None:    
        for features in features_list:
            for k,v in features:
                if v != 0:
                    feature_freq[k] += 1
    
        vocab = defaultdict(lambda: 0)
        c = 0
    
        for feature in sorted(feature_freq):
            if feature_freq[feature] >= min_freq:
                vocab[feature] = c
                c += 1
    
    data = []
    rows= []
    col = []
    
        
    #print(vocab)
    for i in range(0, len(features_list)):
        for k,v in features_list[i]:
            if k in vocab:
                rows.append(i)
                col.append(vocab[k])
                data.append(v)
                
    return csr_matrix((data, (rows, col)), shape=(len(features_list), len(vocab))), vocab


def accuracy_score(truth, predicted):
    
    return len(np.where(truth==predicted)[0]) / len(truth)


def cross_validation_accuracy(clf, X, labels, k):
    cv = KFold(n_splits=k, random_state=len(labels))
    accuracies = []
    for train_ind, test_ind in cv.split(X):
        clf.fit(X[train_ind], labels[train_ind])
        predictions = clf.predict(X[test_ind])
        accuracies.append(accuracy_score(labels[test_ind], predictions))
    
    return np.mean(accuracies)

def eval_all_combinations(docs, labels, punct_vals,feature_fns, min_freqs):

    data = []
    feat_combs = []
    for i in range(1, len(feature_fns)+1):
        for combo in combinations(feature_fns, i):
            feat_combs.append(list(combo))

    for punct in punct_vals:
        tokens_list = [tokenize(doc, punct) for doc in docs]
        for combo in feat_combs:
            for freq in min_freqs:
                X, vocab = vectorize(tokens_list, combo, freq)
                accuracy = cross_validation_accuracy(LogisticRegression(), X, labels, 5)
                data.append({'features': combo, 'punct': punct, 'accuracy': accuracy, 'min_freq': freq})

    return sorted(data, key = lambda x: (x['accuracy'], x['min_freq']), reverse = True)



def plot_sorted_accuracies(results):
    X = range(0, len(results))
    results = sorted(results, key = lambda x : x['accuracy'])
    Y = [r['accuracy'] for r in results]
    plt.plot(X, Y)
    plt.xlabel('setting')
    plt.ylabel('accuracy')
    plt.xlim(xmin=0)
    #plt.ylim(ymin=0.64)
    plt.savefig('accuracies.png')
    
def mean_accuracy_per_setting(results):
    
    settings = defaultdict(lambda: 0)
    data = []
    for result in results:
        k1 = "punct=%s" % str(result['punct'])
        k2 = "min_freq=%s" % str(result['min_freq'])
        
        combo = ""
        for fn in result['features']:
            combo = combo + " " + fn.__name__
            
        k3 = "features="+combo    
        
        if k1 in settings:
            settings[k1].append(result)
        else:
            settings[k1] = []
            settings[k1].append(result)

        if k2 in settings:
            settings[k2].append(result)
        else:
            settings[k2] = []
            settings[k2].append(result)   
        
        if k3 in settings:
            settings[k3].append(result)
        else:
            settings[k3] = []
            settings[k3].append(result) 
       
    for key, v in settings.items():
        l1 = [row['accuracy'] for row in v]
        m1 = np.mean(l1)
        data.append(tuple((m1,key)))
    
    return sorted(data, key = lambda x: x[0], reverse = True)


def fit_best_classifier(docs, labels, best_result):
    
    tokens_list = [tokenize(doc,best_result['punct']) for doc in docs]
    
    X,vocab=vectorize(tokens_list,best_result['features'],best_result['min_freq'])
    clf = LogisticRegression()
    clf.fit(X,labels)
    return clf,vocab
    

def top_coefs(clf, label, n, vocab):
    
    ids = []
    coef = clf.coef_[0]
    if label == 0:
        ids = np.argsort(coef)[:n]
    if label == 1:
        ids = np.argsort(coef)[::-1][:n]
    
    top_coef_terms = np.array([k for k,v in sorted(vocab.items(), key=lambda x: -x[1], reverse = True)])[ids]
    top_coef = coef[ids]
    
    data = []
    if label == 0:
        for f in zip(top_coef_terms, top_coef*-1):
            data.append(f)
    
    if label == 1:
        for f in zip(top_coef_terms, top_coef):
           data.append(f) 
    
    return data
    



def parse_test_data(best_result, vocab):
    test_docs, test_labels = read_data(os.path.join('data', 'test'))

    tokens_list = [tokenize(d,best_result['punct']) for d in test_docs]
    X_test,vocab=vectorize(tokens_list,best_result['features'],best_result['min_freq'],vocab)

    return test_docs, test_labels, X_test

def print_top_misclassified(test_docs, test_labels, X_test, clf, n):
    
    
    predictions = clf.predict(X_test)
    predict_prob_vals = clf.predict_proba(X_test)
    
    data=[]
    
    for i in range(len(predictions)):
        
        row = {}
        if predictions[i] != test_labels[i]:
            
            if predictions[i] == 0:
                
                row['prob'] = predict_prob_vals[i][0]
                row['truth'] = test_labels[i]
                row['predicted']=predictions[i]
                row['test'] =test_docs[i] 
            else:
            
                row['prob'] = predict_prob_vals[i][1]
                row['truth'] = test_labels[i]
                row['predicted']=predictions[i]
                row['test'] =test_docs[i] 
            data.append(row)
    
    data=sorted(data, key=lambda x: (x['prob']), reverse = True)[:n]
    for r in data:
        print('truth=%d predicted=%d proba=%.6f'%(r['truth'],r['predicted'],r['prob']))
        print(r['test']+"\n")  



def main():
    """
    Put it all together.
    ALREADY DONE.
    """
    feature_fns = [token_features, token_pair_features, lexicon_features]
    # Download and read data.
    download_data()
    docs, labels = read_data(os.path.join('data', 'train'))
    # Evaluate accuracy of many combinations
    # of tokenization/featurization.
    results = eval_all_combinations(docs, labels,
                                    [True, False],
                                    feature_fns,
                                    [2,5,10])
    # Print information about these results.
    best_result = results[0]
    worst_result = results[-1]
    print('best cross-validation result:\n%s' % str(best_result))
    print('worst cross-validation result:\n%s' % str(worst_result))
    plot_sorted_accuracies(results)
    print('\nMean Accuracies per Setting:')
    print('\n'.join(['%s: %.5f' % (s,v) for v,s in mean_accuracy_per_setting(results)]))

    # Fit best classifier.
    clf, vocab = fit_best_classifier(docs, labels, results[0])

    # Print top coefficients per class.
    print('\nTOP COEFFICIENTS PER CLASS:')
    print('negative words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 0, 5, vocab)]))
    print('\npositive words:')
    print('\n'.join(['%s: %.5f' % (t,v) for t,v in top_coefs(clf, 1, 5, vocab)]))

    # Parse test data
    test_docs, test_labels, X_test = parse_test_data(best_result, vocab)

    # Evaluate on test set.
    predictions = clf.predict(X_test)
    print('testing accuracy=%f' %
          accuracy_score(test_labels, predictions))

    print('\nTOP MISCLASSIFIED TEST DOCUMENTS:')
    print_top_misclassified(test_docs, test_labels, X_test, clf, 2)


if __name__ == '__main__':
    main()






    
    
    
