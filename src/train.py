import tensorflow as tf
import numpy as np
import time
import scipy.sparse as sp

from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, auc

from preprocessing import construct_feed_dict
from outputs import viz_train_val_data, viz_roc_pr_curve, max_gmean_thresh

def train_test_model(adj_norm, adj_label, features, adj_orig, FLAGS, crossval_edges, placeholders, opt, model, model_str, model_timestamp, adj, test_edges, test_edges_false):
    acc_cv, ap_cv, roc_cv, acc_init_cv, ap_init_cv, roc_init_cv = ([] for i in range(6))

    feed_dict = None
    adj_pred = None
    iterations = 1 + FLAGS.crossvalidation * (len(adj) - 1)

    for cv_set in range(iterations):
        print("\nCV run " + str(cv_set+1) + " of " + str(iterations) + "...")
        # Construct feed dictionary
        feed_dict = construct_feed_dict(adj_norm[cv_set], adj_label[cv_set], features, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})
        # Train model
        acc_last, ap_last, roc_last, acc_init, ap_init, roc_init, opt_thresh, adj_pred = train_model(adj_orig, FLAGS, [x[cv_set] for x in crossval_edges],
                                            placeholders, opt, model, feed_dict, model_str, model_timestamp)
        for x,l in zip([acc_last, ap_last, roc_last, acc_init, ap_init, roc_init], [acc_cv, ap_cv, roc_cv, acc_init_cv, ap_init_cv, roc_init_cv]):
            l.append(x)

    if FLAGS.crossvalidation:
        cv_str = str(iterations) + " fold CV "
    else:
        cv_str = "Validation "

    print("\n" + cv_str + "ROC AUC score: " + str(np.round(np.mean(roc_cv), 2)) + " with SD: " + str(np.round(np.std(roc_cv),2)))
    print(cv_str + "Average Precision: " + str(np.round(np.mean(ap_cv), 2)) + " with SD: " + str(np.round(np.std(ap_cv),2)))
    print(cv_str + "Accuracy: " + str(np.round(np.mean(acc_cv), 2)) + " with SD: " + str(np.round(np.std(acc_cv),2)))

    #print('\nTest ROC score: ' + str(np.round(test_roc,2)))
    #print('Test AP score: ' + str(np.round(test_ap,2)))
    #print('Test accuracy (threshold= ' + str(opt_thresh) + "): "  + str(np.round(test_acc,2)))

    #print('\nRandom Control ROC score: ' + str(np.round(random_roc,2)))
    #print('Random Control AP score: ' + str(np.round(random_ap,2)))
    #print('Random Control accuracy: ' + str(np.round(random_acc,2)))

    #print('\nAverage Init ROC score: ' + str(np.round(np.mean(roc_init_cv),2)))
    #print('Average Init AP score: ' + str(np.round(np.mean(ap_init_cv),2)))
    #print('Average Init accuracy: ' + str(np.round(np.mean(acc_init_cv),2)) + '\n')

    return np.mean(acc_cv), np.mean(ap_cv), np.mean(roc_cv), adj_pred

def train_model(adj_orig, FLAGS, edges, placeholders, opt, model, feed_dict, model_str, model_timestamp):
    # Initialize session
    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())
    
    loss_train, kl_train, acc_train, ap_train, roc_train, loss_val, acc_val, ap_val, roc_val = ([] for i in range(9))
    hist_scores = [loss_train, kl_train, acc_train, ap_train, roc_train, loss_val, acc_val, ap_val, roc_val]

    #initial metrics
    train_edges, train_edges_false, val_edges, val_edges_false = edges
    adj_pred = predict_adj(feed_dict, sess, model, model_timestamp, placeholders)
    _, _, train_loss, train_acc, train_ap, train_roc, _, opt_thresh = get_scores(adj_pred, adj_orig, train_edges, train_edges_false, model_timestamp)
    _, _, val_loss, val_acc, val_ap, val_roc, _, _ = get_scores(adj_pred, adj_orig, val_edges, val_edges_false, model_timestamp, thresh=opt_thresh)
    train_kl = None
    scores = [train_loss, train_kl, train_acc, train_ap, train_roc, val_loss, val_acc, val_ap, val_roc]    
    for x, l in zip(scores, hist_scores):
        l.append(x)
    
    for epoch in range(FLAGS.epochs):

        t = time.time()

        # Run single weight update
        if model_str == 'gcn_vae':
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy, opt.kl], feed_dict=feed_dict)
        else:
            outs = sess.run([opt.opt_op, opt.cost, opt.accuracy], feed_dict=feed_dict)

        # Compute metrics
        adj_pred = predict_adj(feed_dict, sess, model, model_timestamp, placeholders)

        ctrl_cost = outs[1]
        ctrl_accuracy = outs[2]

        if model_str == 'gcn_vae':
            train_kl = outs[3]
        else:
            train_kl = 0

        _, total_train_acc, train_loss, train_acc, train_ap, train_roc, train_rp, opt_thresh = get_scores(adj_pred, adj_orig, train_edges, train_edges_false, model_timestamp)
        _, _, val_loss, val_acc, val_ap, val_roc, val_rp, _ = get_scores(adj_pred, adj_orig, val_edges, val_edges_false, model_timestamp, thresh=opt_thresh)
        
        scores = [train_loss, train_kl, train_acc, train_ap, train_roc, val_loss, val_acc, val_ap, val_roc]
        for x, l in zip(scores, hist_scores):
            l.append(x)
        
        print("Epoch:", '%04d' % (epoch + 1),
              #"time=", "{:.5f}".format(time.time() - t),
              "train_loss=", "{:.5f}".format(train_loss),
              #"train_loss_control=", "{:.5f}".format(ctrl_cost),
              #"recon loss=", "{:.5f}".format(train_loss-train_kl), "kl_loss=", "{:.5f}".format(train_kl),
              "val_loss=", "{:.5f}".format(val_loss), 
              #"train_acc_control=", "{:.5f}".format(ctrl_accuracy),
              #"total_train_acc=", "{:.5f}".format(total_train_acc),
              "train_acc=", "{:.5f}".format(train_acc),
              #"train_ap=", "{:.5f}".format(train_ap), "train_roc=", "{:.5f}".format(train_roc),
              "val_acc=", "{:.5f}".format(val_acc),
              "val_ap=", "{:.5f}".format(val_ap),
              "val_rp=", "{:.5f}".format(val_rp),
              "val_roc=", "{:.5f}".format(val_roc))
        
        if epoch > FLAGS.early_stopping and loss_val[-1] > np.mean(loss_val[-(FLAGS.early_stopping+1):-1]):
            print("\nEarly stopping...")
            break

    # Plot training & validation metrics
    viz_train_val_data(hist_scores, model_str, model_timestamp)

    #Get final predicted adj matrix
    adj_pred = sigmoid(predict_adj(feed_dict, sess, model, model_timestamp, placeholders))

    #Resulting ROC curve
    #viz_roc = True
    #get_scores(adj_pred, adj_orig, test_edges, test_edges_false, model_timestamp, viz_roc=viz_roc, thresh=opt_thresh)

    #best_epoch = roc_val.index(max(roc_val))
    #print("Best Epoch (ROC): " + str(best_epoch))

    print("Optimization Finished!")
    sess.close()

    return acc_val[-1], ap_val[-1], roc_val[-1], acc_val[0], ap_val[0], roc_val[0], opt_thresh, adj_pred

def predict_adj(feed_dict, sess, model, model_timestamp, placeholders, emb=None):
    if emb is None:
        feed_dict.update({placeholders['dropout']: 0})
        emb = sess.run(model.z_mean, feed_dict=feed_dict)

    adj_rec = np.dot(emb, emb.T)

    return adj_rec

def get_scores(adj_rec, adj_orig, edges_pos, edges_neg, model_timestamp, viz_roc=False, random=False, thresh=None):

    if random:
        adj_rec = random_adj(adj_rec, (adj_orig.sum()-adj_rec.sum())/2, mode="random_normal")
      
    preds = []
    pos = []
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj_orig[e[0], e[1]])
    
    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)
    precision, recall, _ = precision_recall_curve(labels_all, preds_all)
    rp_auc = auc(recall, precision)

    if viz_roc:
        viz_roc_pr_curve(preds_all, labels_all, model_timestamp)
    if thresh is None:
        fpr, tpr, thresholds = roc_curve(labels_all,  preds_all)
        thresh, _, _, _ = max_gmean_thresh(fpr, tpr, thresholds)

    #Total accuracy and loss
    adj_curr = adj_from_edges(edges_pos, adj_orig.shape)
    adj_curr = adj_curr.reshape(1, -1)
    adj_rec = adj_rec.reshape(1, -1)
    
    pos_weight = float(adj_orig.shape[0] * adj_orig.shape[0] - (edges_pos.shape[0] * 2)) / (edges_pos.shape[0] * 2)
    norm = adj_orig.shape[0] * adj_orig.shape[0] / float((adj_orig.shape[0] * adj_orig.shape[0] - (edges_pos.shape[0] * 2)) * 2)
    cost_total = norm * np.mean(weighted_cross_entropy_with_logits(adj_curr, adj_rec, pos_weight))

    correct_prediction = (sigmoid(adj_rec) > thresh) == adj_curr
    accuracy_total = np.mean(correct_prediction)

    #Subset accuracy and loss
    test_mask = adj_from_edges(np.vstack([edges_pos, edges_neg]), adj_orig.shape, diag=0)
    test_mask = test_mask.reshape(1, -1)
    accuracy = np.mean(correct_prediction[test_mask==1])
    cost = np.mean(weighted_cross_entropy_with_logits(adj_curr[test_mask==1], adj_rec[test_mask==1], 1))
    
    return cost_total, accuracy_total, cost, accuracy, ap_score, roc_score, rp_auc, thresh

def weighted_cross_entropy_with_logits(label, pred, pos_weight):
    return ((1 - label) * pred + (1 + (pos_weight - 1) * label) * (np.log(1 + np.exp(-abs(pred))) + np.maximum(-pred, 0)))

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def adj_from_edges(edges, shape, diag=1):
    data = np.ones(edges.shape[0])
    adj = sp.csr_matrix((data, (edges[:, 0], edges[:, 1])), shape=shape)
    adj = adj + adj.T
    if diag==1:
        adj = adj + sp.eye(adj.shape[0])

    return np.array(adj.todense())

def random_adj(adj, edges_to_add, mode="add_edges"):
    if mode=="add_edges":
        edges_added = 0
        adj[adj==0] = -1
        while edges_added < edges_to_add:
            idx_i = np.random.randint(0, adj.shape[0])
            idx_j = np.random.randint(0, adj.shape[0])
            if idx_i == idx_j:
                continue
            elif adj[idx_i, idx_j] == 1:
                continue
            else:
                adj[idx_i, idx_j] = 1
                adj[idx_j, idx_i] = 1
                edges_added += 1
    elif mode=="random_normal":
        adj = np.random.normal(size=adj.shape)
    else:
        print("\nRandom adj matrix: mode not known\n")
        
    return adj
                
    
