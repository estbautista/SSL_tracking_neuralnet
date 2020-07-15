import numpy as np
import math

def accuracy(updated_rankings, seq, snapshot, method):
    # indices of the active nodes 
    idx = np.where( seq[snapshot+1].clust_memb > 0 )[0]
    
    # ground truth to compare with
    if method == 'PageRank':
        ground_truth = np.array(seq[snapshot+1].pr.T[idx])
    elif method == 'GammaPageRank':
        ground_truth = np.array(seq[snapshot+1].gpr[idx])

    # rankings on the active nodes
    prediction = np.array(updated_rankings[idx])

    # error in magnitude
    err_mag = np.linalg.norm( ground_truth - prediction, ord=2 ) / np.linalg.norm( ground_truth, ord=2 )

    # error in angle
    err_deg = math.degrees( math.acos( np.dot(prediction.T, ground_truth) / (np.linalg.norm(ground_truth, ord=2)*np.linalg.norm(prediction, ord=2)) ) )

    return err_mag, err_deg
