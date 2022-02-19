## taken from GRGNN paper, code in GitHub: preprocessing/preprocessing_DREAM5.py
import numpy as np
from scipy.stats import pearsonr, spearmanr
import scipy.sparse
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc, precision_score
from sklearn.metrics import recall_score
from preprocessing import mask_test_edges, construct_adj


def pearsonMatrix_thres(data, threshold=0.8):
    row=[]
    col=[]
    edata=[]
    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            corr, _ = pearsonr(data[:,i],data[:,j])
            if abs(corr) >= threshold:
                row.append(i)
                col.append(j)
                edata.append(1.0)

    row=np.asarray(row)
    col=np.asarray(col)
    edata=np.asarray(edata)
    #check and get full matrix
    mtx = scipy.sparse.csc_matrix((edata, (row,col)), shape=(data.shape[1], data.shape[1]))
    return mtx

def pearsonMatrix(data):
    row, col, edata = ([] for i in range(3))
    for i in np.arange(data.shape[0]):
        for j in np.arange(data.shape[0]):
            corr, _ = pearsonr(data[i,:],data[j,:])
            row.append(j)
            col.append(i)
            edata.append(corr)

    row = np.asarray(row)
    col = np.asarray(col)
    edata = np.asarray(edata)
    edata = edata.ravel()
    mtx = scipy.sparse.csc_matrix((edata, (row,col)), shape=(data.shape[0], data.shape[0]))
    mtx = mtx.toarray()
    mtx = mtx - scipy.sparse.identity(data.shape[0]) #delete diagonal since we allow no self-connected nodes
    return mtx

def pearson_get_scores(adj_rec, adj_orig, edges_pos, edges_neg):
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(abs(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(abs(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    rp_auc = auc(recall, precision)
    f_score = 2 * (np.mean(precision) * np.mean(recall)) / (np.mean(precision) + np.mean(recall))

    return roc_score, ap_score, rp_auc, f_score


def randomMatrix(cols, rows):
    return np.random.rand(cols,rows)


    
