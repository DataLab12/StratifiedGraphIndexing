import numpy as np
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
import time


def recall(true,predicted):
    match = np.intersect1d(true,predicted)
    # print('Matchhhhhhhh:',match)
    recall = len(match)/len(true)
    return recall

# def recall(pred, true_neighbors):
#     total = 0
#     for gt_row, row in zip(true_neighbors, pred):
#         total += np.intersect1d(gt_row, row).shape[0]
#     return total / true_neighbors.size

# def recall(pred,gt,dim):
#     print('Pred in eval: ',pred,pred.shape,'\n',gt,gt.shape)
#     t0 = time.time()
#     ks = [1,5,10]
#     for k in ks: 
#         print('K in recall:',k)
#         recall_at_1 = (pred[:, :k] == gt[:, :k]).sum() / float(dim) / k
#         print("\t %7.3f ms per query, R@%-2d %.4f" % ((time.time() - t0) * 1000.0 / dim, k, recall_at_1))

def ownpre(true,pred):
    count=0
    for (t,p) in zip(true,pred):
        if t==p:
            count=count+1

    return count/len(true)

def precision(true,pred):
    precision = precision_score(true,pred,average='micro')

    # print('Precision : ',precision)
    return precision