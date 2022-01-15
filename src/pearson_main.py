## taken from GRGNN paper, code in GitHub: preprocessing/preprocessing_DREAM5.py
import numpy as np
from scipy.stats import pearsonr
import scipy.sparse
from input_data import load_data
import time
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc


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
    for i in np.arange(data.shape[1]):
        for j in np.arange(data.shape[1]):
            corr, _ = pearsonr(data[:,i],data[:,j])
            row.append(i)
            col.append(j)
            edata.append(corr)

    row = np.asarray(row)
    col = np.asarray(col)
    edata = np.asarray(edata)
    edata = edata.ravel()
    mtx = scipy.sparse.csc_matrix((edata, (row,col)), shape=(data.shape[1], data.shape[1]))
    return mtx

#copied from main.py
model_timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + 'gasch_GSE102475' + '_' + 'yeast_chipunion_KDUnion_intersect'
norm_expression_path = 'data/normalized_expression/gasch_GSE102475.csv'
gold_standard_path = 'data/gold_standards/' + 'yeast_chipunion_KDUnion_intersect' + '.txt'
adj, features, gene_names = load_data(norm_expression_path, gold_standard_path, model_timestamp, 0)

#from scipy sparse matrix
features = features.todense()
pearson_test = pearsonMatrix_thres(np.transpose(features))
pearson_test = pearson_test.todense()
pearson_tmp = pearson_test.flatten()
pearson_test_array = np.squeeze(np.asarray(pearson_tmp))


#path spaeter noch anpassen, dass immer das neueste genommen wird
ground_truth_path = 'logs/outputs/20220115_193357_gasch_GSE102475_yeast_chipunion_KDUnion_intersect_preprocessed_adj.csv'
my_data = np.genfromtxt(ground_truth_path, delimiter=';', dtype=None)
my_data = my_data.flatten()
my_data = np.transpose(my_data)

print('ROC AUC score: ' + str(roc_auc_score(my_data, pearson_test_array)))

















############################### NOT USED CODE
edge_filename = 'data/gold_standards/yeast_chipunion_KDUnion_intersect.txt'

norm_expression = pd.read_csv(norm_expression_path, sep=',', header=0, index_col=0)

rownum = 3847
colnum = 163
data = np.zeros((rownum,colnum))

count = -1
with open(norm_expression_path) as f:
    lines = f.readlines()
    for line in lines:
        if count >= 0:
            line = line.strip()
            words = line.split(',')
            ncount = -1
            for word in words:
                if ncount >= 0:
                    word = float(word)
                    data[count, ncount] = word
                ncount = ncount +1
        count = count + 1
    f.close()
############################################