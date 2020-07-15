import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from accuracy import accuracy

def prepare_data_test(input_rank, seq, snapshot, method, hops):

    # indices of the active nodes, to not consider the nodes that leave as they do not need update
    idx = np.where( seq[snapshot+1].clust_memb > 0 )[0]

    # feature 1: pagerank on the nodes that are active
    feat_1 = input_rank[idx] 

    # diffuse in both graphs
    val_g2 = [feat_1]
    val_g1 = [feat_1]
    tmp_feat = []
    for k in range(hops):
        if method == 'PageRank':
            val_g2.append( seq[snapshot+1].P[np.ix_(idx,idx)].T.dot(val_g2[k]))
            val_g1.append( seq[snapshot].P[np.ix_(idx,idx)].T.dot(val_g1[k]) )
            tmp_feat.append( val_g1[k+1] )
            tmp_feat.append( val_g2[k+1] )
        elif method == 'GammaPageRank':
            val_g2.append( seq[snapshot+1].P2[np.ix_(idx,idx)].dot(val_g2[k]) )
            val_g1.append( seq[snapshot].P2[np.ix_(idx,idx)].dot(val_g1[k]) ) 
            tmp_feat.append( val_g1[k+1] )
            tmp_feat.append( val_g2[k+1] )
    tmp_feat = np.concatenate( tmp_feat, axis=1 )

    # corresponding indices to update
    nzero_idx = np.where(idx)[0]
    idx_to_update = idx[nzero_idx]
    
    # features 
    feat_test_data = torch.tensor(np.concatenate([feat_1[nzero_idx], tmp_feat[nzero_idx]], axis=1)).double()

    return feat_test_data, idx_to_update


def model_testing(model_eval, seq, snaps, num_time_steps, hops, method ):

    err_magnitude = []
    err_degrees = []
    predictions = []
    if method == 'PageRank':
        input_rank = np.array( seq[snaps].pr.T )
    elif method == 'GammaPageRank':
        input_rank = np.array( seq[snaps].gpr )

    for snapshot in range(snaps, num_time_steps):
        #######################
        # compute feature vector
        feat_test_data, idx_to_update = prepare_data_test(input_rank, seq, snapshot, method, hops)

        #######################
        # predict new rank
        model_eval.eval()
        predict = model_eval(feat_test_data)

        #######################
        # update rankings
        updated_rankings = np.array(input_rank)
        updated_rankings[idx_to_update] = predict.detach().numpy()

        #######################
        # evaluate accuracy of prediction
        err_mag, err_deg = accuracy(updated_rankings, seq, snapshot, method)
        err_magnitude.append( err_mag ); 
        err_degrees.append( err_deg )

        #######################
        # set the prediction as the input of the new graph
        input_rank = np.array(updated_rankings)
        
        #######################
        # store the prediction
        predictions.append( updated_rankings )

    return err_magnitude, err_degrees, predictions

def analytic_update( seq, snaps, num_time_steps, hops, alpha, method) :

    if method == 'PageRank':
        input_rank = np.array( seq[snaps].pr.T )
    elif method == 'GammaPageRank':
        input_rank = np.array( seq[snaps].gpr )

    err_magnitude = []
    err_degrees = []
    predictions = []
    for snapshot in range(snaps, num_time_steps):
    
        #######################
        # update distribution local on the perturbation
        if method == 'PageRank':
            update_dist = (1-alpha)*( seq[snapshot+1].P.T - seq[snapshot].P.T ).dot(input_rank)
        elif method == 'GammaPageRank':
            mu = alpha/(1-alpha)
            psi = -10/(2*mu + 10);
            rho = (2*mu)/(2*mu + 10); 
            update_dist = psi*( seq[snapshot+1].Op_shift - seq[snapshot].Op_shift ).dot( input_rank )

        #######################
        # diffuse the initial update distribution
        tmp_dif = np.array(update_dist)
        man_up = np.array(update_dist)
        if method == 'PageRank':
            for k in range(hops-1):
                tmp_dif = (1-alpha)*(seq[snapshot+1].P.T).dot( tmp_dif )
                man_up += tmp_dif
        elif method == 'GammaPageRank':
            for k in range(hops-1):
                tmp_dif = psi*(seq[snapshot+1].Op_shift).dot( tmp_dif )
                man_up += tmp_dif
            
        #######################
        # update the rankings
        updated_rankings = np.array(input_rank + man_up)

        #######################
        # compute the error 
        err_mag, err_deg = accuracy(updated_rankings, seq, snapshot, method)
        err_magnitude.append( err_mag ); 
        err_degrees.append( err_deg )

        #######################
        # set the prediction as the input of the new graph
        input_rank = np.array(updated_rankings)
        
        #######################
        # store the prediction
        predictions.append( updated_rankings )

    return err_magnitude, err_degrees, predictions

def no_update(seq, snaps,num_time_steps, method):
    err_magnitude = []
    err_degrees = []
    predictions = []

    for snapshot in range(snaps, num_time_steps):
        if method == 'PageRank':
            updated_rankings = np.array(seq[snaps].pr.T)
        elif method == 'GammaPageRank':
            updated_rankings = np.array(seq[snaps].gpr)

        err_mag, err_deg = accuracy(seq[snaps].pr.T, seq, snapshot, method)
        err_magnitude.append( err_mag ); 
        err_degrees.append( err_deg )

    return err_magnitude, err_degrees

