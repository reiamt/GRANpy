import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, precision_recall_curve, auc

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def viz_train_val_data(hist_scores, model_str, model_timestamp):
    
    # Plot training & validation metrics
    loss_train, kl_train, acc_train, ap_train, roc_train, f_train, loss_val, acc_val, ap_val, roc_val, f_val = hist_scores
    figure, axis = plt.subplots(2,2)

    axis[0, 0].plot(loss_train)
    axis[0, 0].plot(loss_val, color='tab:orange')
    axis[0, 0].set_title('Total loss')
    axis[0, 0].set_xlabel('Epoch')

    axis[0, 1].plot(ap_train)
    axis[0, 1].plot(ap_val, color='tab:orange')
    #axis[0, 1].set_ylim([0.5, 1.0])
    axis[0, 1].set_title('Average Precision')
    axis[0, 1].set_xlabel('Epoch')

    axis[1, 0].plot([np.subtract(x1, x2) for (x1, x2) in zip(loss_train[1:], kl_train[1:])])
    axis[1, 0].set_title('Recon loss')
    axis[1, 0].set_xlabel('Epoch')
    if model_str == 'gcn_vae':
        axis2 = axis[1, 0].twinx()
        axis2.plot(kl_train, color='tab:orange')
        axis[1, 0].set_title('Recon/KL loss')

    axis[1, 1].plot(roc_train)
    axis[1, 1].plot(roc_val, color='tab:orange')
    #axis[1, 1].set_ylim([0.5, 1.0])
    axis[1, 1].set_title('ROC AUC')
    axis[1, 1].set_xlabel('Epoch')

    figure.tight_layout()
    figure.savefig('logs/training_plots/' + model_timestamp + '_training_history.png', dpi=300)
    plt.close(figure)

def viz_roc_pr_curve(y_pred, y_true, model_timestamp):
    figure, axis = plt.subplots(1,3)
    figure.set_size_inches(12.8, 4.8)
    
    #plot ROC curve
    fpr, tpr, thresholds = roc_curve(1-y_true,  1-y_pred)
    axis[0].plot(fpr, tpr, label="negative class: auc="+str(np.round(auc(fpr, tpr),2)))
    
    fpr, tpr, thresholds = roc_curve(y_true,  y_pred)
    axis[0].plot(fpr, tpr, label="positive class: auc="+str(np.round(auc(fpr, tpr),2)))
    
    thresholdOpt, _, fprOpt, tprOpt = max_gmean_thresh(fpr, tpr, thresholds)
    axis[0].plot(fprOpt, tprOpt, 'ro', label=('max g-mean threshold: ' + str(thresholdOpt)))
    
    axis[0].set_title("ROC curve")
    axis[0].set_xlabel("False Positive Rate")
    axis[0].set_ylabel("True Positive Rate")
    axis[0].set_xlim([0, 1])
    axis[0].set_ylim([0, 1])
    axis[0].legend(loc=4)

    #plot PR curve
    precision, recall, thresholds = precision_recall_curve(1-y_true, 1-y_pred)
    axis[1].plot(recall, precision, label="negative class: auc="+str(np.round(auc(recall, precision),2)))
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    axis[1].plot(recall, precision, label="positive class: auc="+str(np.round(auc(recall, precision),2)))
                    
    axis[1].set_title("PR curve")
    axis[1].set_xlabel("Recall")
    axis[1].set_ylabel("Precision")
    axis[1].set_xlim([0, 1])
    axis[1].set_ylim([0, 1])
    axis[1].legend(loc=2)

    #plot histogram
    axis[2].hist(y_pred, bins=100)
    axis[2].set_title("Histogram of predictions (" + str(len(y_pred)) + ")")
    axis[2].set_xlabel("Prediction")
    axis[2].set_ylabel("Count")
    axis[2].set_xlim([0, 1])

    figure.tight_layout()
    figure.savefig('logs/training_plots/' + model_timestamp + '_ROC_PR_curve.png', dpi=300)
    plt.close(figure)

def max_gmean_thresh(fpr, tpr, thresholds):
    gmean = np.sqrt(tpr * (1 - fpr))
    index = np.argmax(gmean)
    thresholdOpt = np.round(thresholds[index], 2)
    gmeanOpt = np.round(gmean[index], 2)
    fprOpt = np.round(fpr[index], 2)
    tprOpt = np.round(tpr[index], 2)
    
    return thresholdOpt, gmeanOpt, fprOpt, tprOpt

def save_adj(adj, outPath, model_timestamp, gene_names):
    if outPath is None:
        outPath = "logs/outputs/" + model_timestamp + '_'
        
    #save adjacency matrix
    np.savetxt(outPath + 'adj_pred.csv', adj, delimiter=",")

    #save gene interaction list
    adj_df = pd.DataFrame(data=adj, index=gene_names, columns=gene_names)
    adj_df = adj_df.where(np.triu(np.ones(adj_df.shape)).astype(np.bool))
    gene_list = adj_df.stack().reset_index()
    gene_list.columns = ['Gene1','Gene2','EdgeWeight']
    gene_list = gene_list.sort_values(by='EdgeWeight', ascending=False)
    gene_list.to_csv(outPath + 'gene_edge_weights.txt', header=True, index=None, sep='\t', mode='w')
