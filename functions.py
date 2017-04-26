import matplotlib.pyplot as plt
from IPython import display

import numpy as np
import pandas as pd
import os
import pickle

from scipy.sparse.linalg import svds
from scipy.spatial.distance import cosine
from scipy.stats import spearmanr

from word2vec_as_EMF import Word2vecMF

def load_sentences(mode='debug'):
    """
    Load training corpus sentences/
    """
    
    if (mode == 'imdb'):
        sentences = pickle.load(open('data/sentences_all.txt', 'rb'))
    elif (mode == 'debug'):
        sentences = pickle.load(open('data/sentences1k.txt', 'rb'))
    elif (mode == 'enwik9'):
        sentences = pickle.load(open('data/enwik9_sentences.txt', 'rb'))
    return sentences

def plot_MF(MFs, x=None, xlabel='Iterations', ylabel='MF'):
    """
    Plot given MFs.
    """
    
    fig, ax = plt.subplots(figsize=(15, 5))
    if not x:
        ax.plot(MFs)
    else:
        ax.plot(x, MFs)
    ax.set_xlabel(xlabel, fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.grid(True)
    
    
################################## Projector splitting experiment ##################################
def opt_experiment(model,
                   mode='PS',
                   min_count=200,
                   d = 100,
                   eta = 5e-6,
                   batch_size = 1000,
                   MAX_ITER = 100,
                   from_iter = 0,
                   start_from = 'RAND',
                   init = (False, None, None)            
                  ):
    """
    Aggregator for projector splitting experiment.
    """ 
    
    # Start projector splitting from svd of SPPMI or random initialization
  
    def folder_path(num_iter):
        return 'enwik-'+str(min_count)+'/'+mode+str(num_iter)+'iter_from'+start_from+'_dim'+str(d)+'_step'+str(eta)+'_factors'

    
    from_folder = folder_path(from_iter+MAX_ITER)
    if (from_iter > 0):
        os.rename(folder_path(from_iter), from_folder)  
        C, W = model.load_CW(from_folder, from_iter)
        init_ = (True, C, W)
    else:
        init_ = init
    
    
    if (mode == 'PS'):
        model.projector_splitting(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter,
                                  init=init_, save=(True, from_folder))
        
    if (mode == 'SPS'):
        model.stochastic_ps(eta=eta, batch_size=batch_size, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter,
                            init=init_, save=(True, from_folder))
        
    if (mode == 'AM'):
        model.alt_min(eta=eta, d=d, MAX_ITER=MAX_ITER, from_iter=from_iter,
                      init=init_, save=(True, from_folder))
    
    return model


################################## Word similarity experiments ##################################

def datasets_corr(model, datasets_path, from_folder, MAX_ITER=100, plot_corrs=False, matrix='W', train_ratio=1.0):
    """
    Calculate correlations for all datasets in datasets_path
    """
    
    indices = np.load(open(datasets_path+'/indices.npz', 'rb'))
    sorted_names = ['mc30', 'rg65', 'verb143', 'wordsim_sim', 'wordsim_rel', 'wordsim353', 
                    'mturk287', 'mturk771', 'simlex999', 'rw2034', 'men3000']
    
    # Calculate correlations
    corrs_dict = {}
    #for filename in os.listdir(datasets_path):
        #if filename[-4:]=='.csv':
        
    for name in sorted_names:
        
        corrs = []
        
        pairs_num = indices['0'+name].size
        idx = np.arange(pairs_num)
        np.random.shuffle(idx)
        idx = idx[:int(train_ratio * pairs_num)]
        
        ind1 = indices['0'+name][idx]
        ind2 = indices['1'+name][idx]
        scores = indices['2'+name][idx]

        for it in xrange(MAX_ITER):
            W, C = model.load_CW(from_folder, it)
            if (matrix == 'W'):
                G = W
            else:
                G = (C.T).dot(W)
            G = G / np.linalg.norm(G, axis=0)
            cosines = (G[:,ind1]*G[:,ind2]).sum(axis=0)
            corrs.append(spearmanr(cosines, scores)[0])

        corrs = np.array(corrs)  
        corrs_dict[name] = corrs
            
    # Plot correlations
    if (plot_corrs):
        
        #w2v_ds_corrs = datasets_corr(model, datasets_path, 'enwik-200/w2v/factors', MAX_ITER=1, plot_corrs=False)
        #svd_ds_corrs = datasets_corr(model, datasets_path, 'enwik-200/PS800iter_fromSVD_factors', MAX_ITER=1, plot_corrs=False)
        
        fig, ax = plt.subplots(4, 3, figsize=(15, 20))
        for num, name in enumerate(sorted_names):
            x = ax[num/3,num%3]

            x.plot(corrs_dict[name], lw=2, label='opt method')
            x.set_title(name, fontsize=14)

            # plot original word2vec correlation
            #w2v_corr = w2v_ds_corrs[name][0]
            #x.plot((0, MAX_ITER), (w2v_corr, w2v_corr), 'k-', color='red', lw=2, label='SGNS')

            # plot SVD correlation
            #svd_corr = svd_ds_corrs[name][0]
            #x.plot((0, MAX_ITER), (svd_corr, svd_corr), 'k-', color='green', lw=2, label='SVD')

            x.legend(loc='best')
            x.grid()
    
    
    return corrs_dict
    
def corr_experiment(model, data, from_folder, MAX_ITER=100, plot_corrs=False):
    """
    Aggregator for word similarity correlation experiment.
    """
    
    # Load dataset and model dictionary

    dataset = data.values
    model_vocab = model.vocab

    # Choose only pairs of words which exist in model dictionary
    ind1 = []
    ind2 = []
    vec2 = []
    chosen_pairs = []
    for i in xrange(dataset.shape[0]):
        word1 = dataset[i, 0].lower()
        word2 = dataset[i, 1].lower()
        if (word1 in model_vocab and word2 in model_vocab):
            ind1.append(int(model_vocab[word1]))
            ind2.append(int(model_vocab[word2]))
            vec2.append(np.float64(dataset[i, 2]))
            chosen_pairs.append((word1, word2))
            
    # Calculate correlations
    corrs = []
    vecs = []
    for it in xrange(MAX_ITER):
        vec1 = []
        C, W = model.load_CW(from_folder, it)
        
        G = (C.T).dot(W)
        pc = (model.D).sum(axis=1) / (model.D).sum()
        vec1 = (pc.reshape(-1, 1)*G[:,ind1]*G[:,ind2]).sum(axis=0)
        vec1 = list(vec1)
        
        #W = W / np.linalg.norm(W, axis=0)
        #vec1 = (W[:,ind1]*W[:,ind2]).sum(axis=0)
        #vec1 = list(vec1)
        corrs.append(spearmanr(vec1, vec2)[0])
        vecs.append(vec1)
    corrs = np.array(corrs)  
    vecs = np.array(vecs)
    
    # Plot correlations
    if (plot_corrs):
        plots = [corrs, vecs.mean(axis=1), vecs.std(axis=1)]
        titles = ['Correlation', 'Mean', 'Standard deviation']

        fig, ax = plt.subplots(1, 3, figsize=(15, 5))
        for i in xrange(3):
            ax[i].plot(plots[i])
            ax[i].set_title(titles[i], fontsize=14)
            ax[i].set_xlabel('Iterations', fontsize=14)
            ax[i].grid()
    
    return corrs, vecs, vec2, chosen_pairs

def plot_dynamics(vecs, vec2, n=5, MAX_ITER=100):
    """
    Plot how the distances between pairs change with each n
    iterations of the optimization method.
    """
    
    for i in xrange(MAX_ITER):
        if (i%n==0):
            plt.clf()
            plt.xlim([-0.01, 1.2])
            plt.plot(vecs[i], vec2, 'ro', color='blue')
            plt.grid()
            display.clear_output(wait=True)
            display.display(plt.gcf())
    plt.clf()
    
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    for i in xrange(2):
        ax[i].set_xlim([-0.01, 1.2])
        ax[i].plot(vecs[-i], vec2, 'ro', color='blue')
        ax[i].set_title(str(i*MAX_ITER)+' iterations')
        ax[i].set_xlabel('Cosine distance', fontsize=14)
        ax[i].set_ylabel('Assesor grade', fontsize=14)
        ax[i].grid()
    
    
def dist_change(vecs, vec2, chosen_pairs, n=5, dropped=True, from_iter=0, to_iter=-1):
    """
    Get top pairs which change distance between words the most.
    """
    
    vecs_diff = vecs[to_iter, :] - vecs[from_iter, :]
    args_sorted = np.argsort(vecs_diff)

    for i in xrange(n):
        if (dropped):
            idx = args_sorted[i]
        else:
            idx = args_sorted[-1-i]
        print "Words:", chosen_pairs[idx]
        print "Assesor score:", vec2[idx]
        print "Distance change:", vecs[from_iter, idx], '-->', vecs[to_iter, idx]
        print '\n'
    
    
def nearest_words_from_iter(model, word, from_folder, top=20, display=False, it=1):
    """
    Get top nearest words from some iteration of optimization method.
    """
    C, W = model.load_CW(from_folder=from_folder, iteration=it)

    model.W = W.copy()
    model.C = C.copy()

    nearest_sum = model.nearest_words(word, top, display)
    
    return nearest_sum

################################## Analogical reasoning experiments ##################################

def argmax_fun(W, indices, argmax_type='levi'):
    """
    cosine: b* = argmax cosine(b*, b - a + a*) 
    levi: b* = argmax cos(b*,a*)cos(b*,b)/(cos(b*,a)+eps)
    """
    
    if (argmax_type == 'levi'):
        W = W / np.linalg.norm(W, axis=0)
        words3 = W[:, indices]
        cosines = ((words3.T).dot(W) + 1) / 2
        obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
        pred_idx = np.argmax(obj)
        
    elif (argmax_type == 'cosine'):
        words3_vec = W[:, indices].sum(axis=1) - 2*W[:, indices[0]]
        W = W / np.linalg.norm(W, axis=0)
        words3_vec = words3_vec / np.linalg.norm(words3_vec)
        cosines = (words3_vec.T).dot(W)
        pred_idx = np.argmax(cosines)
        
    return pred_idx

def analogical_reasoning(model, dataset, from_folder, it=0):
    """
    Calculate analogical reasoning accuracy for given dataset.
    """
    dic = model.dictionary
    
    _, W = model.load_CW(from_folder, iteration=it)
    W = W / np.linalg.norm(W, axis=0)

    good_sum = 0
    miss_sum = 0

    for words in dataset.values:

        a, b, a_, b_ = words

        if (a in dic and b in dic and a_ in dic and b_ in dic):

            indices = [dic[a], dic[b], dic[a_]]
            
            words3 = W[:, indices]
            cosines = ((words3.T).dot(W) + 1) / 2
            obj = (cosines[1] * cosines[2]) / (cosines[0] + 1e-3)
            pred_idx = np.argmax(obj)
            
            if (model.inv_dict[pred_idx] == b_):
                good_sum += 1
        else: 
            miss_sum += 1

    # calculate accuracy
    acc = (good_sum) / float(dataset.shape[0]-miss_sum)
    
    return acc, miss_sum

def AR_experiment(model, dataset, from_folder, MAX_ITER=100, step_size=5, plot_accs=False):
    """
    Aggregator for analogical reasoning accuracy experiment.
    """
    
    # Calculate accuracies 
    accs = []
    num_points = MAX_ITER/step_size + 1
    for i in xrange(num_points):
        acc, miss = analogical_reasoning(model, dataset, from_folder, i*step_size)
        accs.append(acc)
    accs = np.array(accs)
    
    # Plot accuracies
    if (plot_accs):
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(np.arange(num_points)*step_size, accs)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.set_xlabel('Iterations', fontsize=14)
        ax.grid()    
        
    return accs, miss