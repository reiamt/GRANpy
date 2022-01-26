from __future__ import division
from __future__ import print_function

import time
import os
import warnings

# Train on CPU (hide GPU) due to memory constraints
os.environ['CUDA_VISIBLE_DEVICES'] = ""

import tensorflow as tf
from tensorboard.plugins.hparams import api as hp
import numpy as np
import scipy.sparse as sp
import pandas as pd

from optimizer import OptimizerAE, OptimizerVAE
from input_data import load_data
from model import GCNModelAE, GCNModelVAE
from preprocessing import preprocess_graph, sparse_to_tuple, gen_train_val_test_sets
from train import train_test_model
from outputs import save_adj

from pearson_main import pearsonMatrix, pearson_get_scores, randomMatrix

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('verbose', 1, 'Verbosity of output from low (0) to high (1)')

flags.DEFINE_float('learning_rate', 0.00001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 1000, 'Number of max epochs to train.')
flags.DEFINE_integer('hidden1', 64, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 48, 'Number of units in hidden layer 2.')
flags.DEFINE_float('weight_decay', 0., 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_float('dropout', 0., 'Dropout rate (1 - keep probability).')
flags.DEFINE_integer('early_stopping', 5, 'Tolerance for early stopping (# of epochs).')

flags.DEFINE_string('model', 'gcn_ae', 'Model string.')
flags.DEFINE_float('ratio_val', 0.2, 'Ratio of edges used for validation metrics.')
flags.DEFINE_float('ratio_test', 0.1, 'Ratio of edges used for test metrics.')
flags.DEFINE_integer('balanced_metrics', 1, 'Whether to use balanced metrics (1) or not (0).')

flags.DEFINE_string('dataset', 'gasch_GSE102475', 'Dataset file name.')
flags.DEFINE_string('ground_truth', 'yeast_chipunion_KDUnion_intersect', 'Gold standard edges file name.')
flags.DEFINE_string('inFilePath', None, 'Input Files path')
flags.DEFINE_string('outFilePath', None, 'Output Files path')

flags.DEFINE_integer('features', 1, 'Whether to use features (1) or not (0).')
flags.DEFINE_integer('random_prior', 0, 'When prior adjacency matrix should be set to random matrix (1) or not (0).')

flags.DEFINE_integer('crossvalidation', 0, 'Whether to use crossvalidation (1) or not (0).')
flags.DEFINE_integer('hp_optimization', 0, 'Whether to start the hyperparameter optimization run (1) or not (0).')

model_str = FLAGS.model
model_timestamp = time.strftime("%Y%m%d_%H%M%S") + '_' + FLAGS.dataset + '_' + FLAGS.ground_truth

if FLAGS.verbose == 0:
    warnings.filterwarnings("ignore")

if not os.path.exists('logs'):
    os.mkdir('logs')
if not os.path.exists('logs/outputs'):
    os.mkdir('logs/outputs')
if not os.path.exists('logs/training_plots'):
    os.mkdir('logs/training_plots')

# Load data
if FLAGS.inFilePath is None:
    norm_expression_path = 'data/normalized_expression/' + FLAGS.dataset + '.csv'
    gold_standard_path = 'data/gold_standards/' + FLAGS.ground_truth + '.txt'
else:
    norm_expression_path = FLAGS.inFilePath + 'ExpressionData' + '.csv'
    gold_standard_path = FLAGS.inFilePath + 'PriorNetwork' + '.txt'

adj, features, gene_names = load_data(norm_expression_path, gold_standard_path, model_timestamp, FLAGS.random_prior)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj
adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
adj_orig.eliminate_zeros()

adj_train, crossval_edges, test_edges, test_edges_false = gen_train_val_test_sets(adj_orig, FLAGS.crossvalidation, FLAGS.balanced_metrics, FLAGS.ratio_val, FLAGS.ratio_test)
adj = adj_train

##### pearson matrix extension #################
val_features = features.toarray()
pearson = pearsonMatrix(np.transpose(val_features))

random_matrix = randomMatrix(len(gene_names), len(gene_names))
print('length of gene names: ' + str(len(gene_names)))

pearson_val_edges = np.array(crossval_edges[2])
pearson_val_edges = np.reshape(pearson_val_edges, (355,2))

pearson_val_edges_false = np.array(crossval_edges[3])
pearson_val_edges_false = np.reshape(pearson_val_edges_false, (355,2))

pear_roc_score, pear_ap_score, pear_rp_auc, pear_f1_score = pearson_get_scores(pearson, adj_orig, pearson_val_edges, pearson_val_edges_false)
random_roc_score, random_ap_score, random_rp_auc, random_f1_score = pearson_get_scores(random_matrix, adj_orig, pearson_val_edges, pearson_val_edges_false)

print('pearson roc score is: ' + str(pear_roc_score))
print('pearson ap score is: ' + str(pear_ap_score))
print('pearson rp score is: ' + str(pear_rp_auc))
#print('pearson f1 score is: ' + str(pear_f1_score))

print('random roc score is: ' + str(random_roc_score))
print('random ap score is: ' + str(random_ap_score))
print('random rp score is: ' + str(random_rp_auc))
#print('random f1 score is: ' + str(random_f1_score))

###### LEAP extension##########################################
leap_gasch_path = 'logs/leap/MAC_symmetric_gasch_example.csv'
leap_gasch = pd.read_csv(leap_gasch_path, sep=',', header=0, index_col=False)
leap_gasch = leap_gasch.fillna(1)
leap_gasch_adj = leap_gasch.values

leap_roc_score, leap_ap_score, leap_rp_auc, leap_f1_score = pearson_get_scores(leap_gasch_adj, adj_orig, pearson_val_edges, pearson_val_edges_false)
print('leap roc score is: ' + str(leap_roc_score))
print('leap ap score is: ' + str(leap_ap_score))
print('leap rp score is: ' + str(leap_rp_auc))
#################################################

if FLAGS.features == 0:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = [preprocess_graph(m) for m in adj]

adj_label = [(m + sp.eye(m.shape[0])) for m in adj_train]
#np.savetxt('logs/outputs/' + model_timestamp + '_adj_train.csv', adj_label[-1].toarray(), delimiter=";")
adj_label = [sparse_to_tuple(m) for m in adj_label]

features = sparse_to_tuple(features.tocoo())

def build_tf_graph(model_str, features, adj):
    # Define placeholders
    placeholders = {
        'features': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj': tf.compat.v1.sparse_placeholder(tf.float32),
        'adj_orig': tf.compat.v1.sparse_placeholder(tf.float32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=())
    }

    num_features = features[2][1]
    features_nonzero = features[1].shape[0]
    num_nodes = adj[0].shape[0]

    # Create model
    model = None
    if model_str == 'gcn_ae':
        model = GCNModelAE(placeholders, num_features, features_nonzero)
    elif model_str == 'gcn_vae':
        model = GCNModelVAE(placeholders, num_features, num_nodes, features_nonzero)

    pos_weight = float(adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) / adj[0].sum()
    norm = adj[0].shape[0] * adj[0].shape[0] / float((adj[0].shape[0] * adj[0].shape[0] - adj[0].sum()) * 2)

    # Optimizer
    with tf.name_scope('optimizer'):
        if model_str == 'gcn_ae':
            opt = OptimizerAE(preds=model.reconstructions,
                              labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                          validate_indices=False), [-1]),
                              pos_weight=pos_weight,
                              norm=norm)
        elif model_str == 'gcn_vae':
            opt = OptimizerVAE(preds=model.reconstructions,
                               labels=tf.reshape(tf.sparse.to_dense(placeholders['adj_orig'],
                                                                           validate_indices=False), [-1]),
                               model=model, num_nodes=num_nodes,
                               pos_weight=pos_weight,
                               norm=norm)

    return placeholders, model, opt


#Build, train and test model
adj_pred = None
if FLAGS.hp_optimization:
    if not os.path.exists('logs/hparam_tuning'):
        os.mkdir('logs/hparam_tuning')
        
    #Hyperparameter Optimization
    HP_NUM_UNITS1 = hp.HParam('num_units1', hp.Discrete([2, 5, 8, 12, 16, 32, 64, 128]))
    HP_RATIO_UNITS2 = hp.HParam('ratio_units2', hp.Discrete([0.1, 0.25, 0.4, 0.65, 0.8]))
    HP_LR = hp.HParam('lr', hp.Discrete([0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]))

    session_num = 0

    for num_units1 in HP_NUM_UNITS1.domain.values:
      for ratio_units2 in HP_RATIO_UNITS2.domain.values:
        for lr in HP_LR.domain.values:
            hparams = {
                HP_NUM_UNITS1: num_units1,
                HP_RATIO_UNITS2: ratio_units2,
                HP_LR: lr,
            }
            FLAGS.learning_rate = hparams[HP_LR]
            FLAGS.hidden1 = hparams[HP_NUM_UNITS1]
            FLAGS.hidden2 = int(np.ceil(hparams[HP_RATIO_UNITS2]*hparams[HP_NUM_UNITS1]))
            run_name = "run" + str(session_num) + "_" + model_str + "_hid1-" + str(FLAGS.hidden1) + "_hid2-" + str(FLAGS.hidden2) + "_lr-" + str(FLAGS.learning_rate)
            print('--- Starting trial %d' % session_num)
            print({h.name: hparams[h] for h in hparams})
            
            tf.compat.v1.reset_default_graph()
            placeholders, model, opt = build_tf_graph(model_str, features, adj)
            acc, ap, roc, adj_pred = train_test_model(adj_norm, adj_label, features, adj_orig, FLAGS, crossval_edges,
                                            placeholders, opt, model, model_str, (model_timestamp + '_' + run_name),
                                            adj, test_edges, test_edges_false)
            session_num += 1

            #Save output adj matrix and gene interaction list    
            save_adj(adj_pred, FLAGS.outFilePath, model_timestamp, gene_names)
      
else:
    #Run model with given hyperparameters
    placeholders, model, opt = build_tf_graph(model_str, features, adj)
    model_timestamp = model_timestamp + "_" + model_str + "_hid1-" + str(FLAGS.hidden1) + "_hid2-" + str(FLAGS.hidden2) + "_lr-" + str(FLAGS.learning_rate)
    _, _, _, adj_pred = train_test_model(adj_norm, adj_label, features, adj_orig, FLAGS, crossval_edges,
                               placeholders, opt, model, model_str, model_timestamp,
                               adj, test_edges, test_edges_false)
    
    #Save output adj matrix and gene interaction list    
    save_adj(adj_pred, FLAGS.outFilePath, model_timestamp, gene_names)
    
