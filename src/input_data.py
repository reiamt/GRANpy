import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn import preprocessing


def parse_index_file(filename):
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def load_data(norm_expression_path, gold_standard_path, model_timestamp, random_prior):

    #read input data
    norm_expression = pd.read_csv(norm_expression_path, sep=',', header=0, index_col=0)
    features = np.array(norm_expression.values)
    gene_names = norm_expression.index.values

    if random_prior:
        adj = np.random.choice([0, 1], size=(len(gene_names),len(gene_names)), p=[0.9999, 0.0001])
        
    else: 
        gold_standard = pd.read_csv(gold_standard_path, sep='\t', header=None, index_col=False)
        adj = pd.DataFrame(0, index=norm_expression.index.values, columns=norm_expression.index.values)
        for index, row in gold_standard.iterrows():
            if row.iat[0] in norm_expression.index.values and row.iat[1] in norm_expression.index.values:
                adj.at[row.iat[0],row.iat[1]] = 1
        adj = np.array(adj.values)

    #preprocess matrices
    adj = preprocess_input_adj(adj, sym=True, diag=0)
    adj, features, gene_names = crop_isolated_nodes(adj, features, gene_names)
    adj, features, gene_names = shuffle_nodes(adj, features, gene_names)
    features = preprocessing.StandardScaler().fit_transform(features)
    #features = preprocessing.MinMaxScaler().fit_transform(features)
   
    adj = sp.csr_matrix(adj)
    features = sp.csr_matrix(features)
    print("shape of adj matrix: " + str(adj.shape))
    print("shape of features matrix: " + str(features.shape))

    #save preprocessed adjacency matrix
    np.savetxt('logs/outputs/' + model_timestamp + '_preprocessed_adj.csv', adj.toarray(), delimiter=";")

    return adj, features, gene_names

def crop_isolated_nodes(adj_orig, feat_orig, genes_orig):
    adj_cropped = adj_orig
    feat_cropped = feat_orig
    genes_cropped = genes_orig
    
    node_degrees = np.sum(adj_orig, axis=0) + np.sum(adj_orig, axis=1)
    isolated_nodes = [i for i,d in enumerate(node_degrees) if d==0]
    remaining_nodes = [i for i,d in enumerate(node_degrees) if d!=0]
    adj_cropped = np.delete(adj_cropped, isolated_nodes, axis=0)
    adj_cropped = np.delete(adj_cropped, isolated_nodes, axis=1)
    feat_cropped = np.delete(feat_cropped, isolated_nodes, axis=0)
    genes_cropped = [genes_cropped[i] for i in remaining_nodes]
    
    return (adj_cropped, feat_cropped, genes_cropped)

def shuffle_nodes(adj_orig, feat_orig, gene_names_orig):
    order = np.arange(adj_orig.shape[0])
    np.random.shuffle(order)
    adj_orig = adj_orig[order, :]
    adj_orig = adj_orig[:, order]
    feat_orig = feat_orig[order, :]
    gene_names = [gene_names_orig[i] for i in order]
        
    return (adj_orig, feat_orig, gene_names)

def preprocess_input_adj(adj_orig, sym, diag):
    if sym:
        adj_sym = adj_orig + adj_orig.T
        adj_sym[adj_sym > 1] = 1
    else:
        adj_sym = adj_orig
    if diag is not None:
        np.fill_diagonal(adj_sym, diag)
    
    return adj_sym
