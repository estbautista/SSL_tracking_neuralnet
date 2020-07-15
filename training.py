import torch
import math
import torch.optim as optim
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from pytorchtools import EarlyStopping


def training(model, train_data, ground_truth, eval_data, eval_ground_truth, lr, epochs, patience, weight_decay):
    num_batches = len(train_data)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=patience, verbose=False)
     
    # print train_data and ground truth
    fig_dif = plt.figure()
    plt.plot( np.concatenate( train_data )[:,0] )
    plt.plot( np.concatenate( ground_truth) )
    plt.savefig('Figs/score_difference.png')
    plt.close()

    for epoch in range(epochs):

        store_res = []  
        store_gt = []
        store_loss = []
        store_ev_loss = []
        for batch in range(num_batches):
            #loss = nn.L1Loss()
            loss = nn.MSELoss()
            model.train()
            optimizer.zero_grad()

            # compute the loss 
            output = model(train_data[batch])
            loss_train = loss(output, ground_truth[batch])
            store_loss.append( loss_train.detach().numpy() )
            store_res.append( output.detach().numpy() )
            store_gt.append( ground_truth[batch].detach().numpy() )

            # backpropagate
            loss_train.backward() 

            # new step
            optimizer.step() 

            # validate the model
            model.eval()
            ev_output = model( eval_data[batch] )
            ev_loss = loss(ev_output, eval_ground_truth[batch])
            store_ev_loss.append( ev_loss.detach().numpy() ) 
        
        early_stopping(np.mean(store_ev_loss), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break


        if epoch%50 == 0:
            print('epoch, loss, ev_loss',epoch, np.mean(store_loss), np.mean(store_ev_loss))
                   
    # Load model
    model.load_state_dict(torch.load('checkpoint.pt'))

    fig_train = plt.figure()
    plt.plot( np.concatenate( store_gt, axis=0), label='gt' )
    plt.plot( np.concatenate( store_res, axis=0 ), label='learned' )
    plt.legend()
    plt.savefig('Figs/train_res.png')
    plt.close()

    return model


def prepare_data_train(seq, snaps, method, hops):
    #####################
    # Feature vectors for all the points on all the snapshopts
    feat_train_data = []; train_data = []; feat_eval_data = []; eval_data = [];

    for snapshot in range(snaps):

        # indices of the active nodes (not the ones that have left or havent joined the graph)
        idx = np.where( seq[snapshot+1].clust_memb > 0 )[0]
          
        # feature 1: pagerank on the nodes that are active
        if method == 'PageRank':
            feat_1 = seq[snapshot].pr.T[idx] 
        elif method == 'GammaPageRank':
            feat_1 = seq[snapshot].gpr[idx]
 
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


        # subsample the amount of changes
        nzero_idx = np.where(idx)[0]
        tot_nzero = math.ceil( len(nzero_idx) )
        if len(idx)/2 < tot_nzero:
            tmp_idx = np.arange(math.ceil(tot_nzero/2))
            tmp_eval = np.arange(math.ceil(tot_nzero/2), tot_nzero)
            tmp_nzero_idx = np.random.permutation(nzero_idx) 
            nzero_idx = tmp_nzero_idx[tmp_idx]
            eval_idx = tmp_nzero_idx[tmp_eval]
        
        # features
        feat_train_data.append( torch.tensor(np.concatenate([feat_1[nzero_idx], tmp_feat[nzero_idx]], axis=1)).double() )

        feat_eval_data.append( torch.tensor(np.concatenate([feat_1[eval_idx], tmp_feat[eval_idx]], axis=1)).double() )

        # Ground truth
        if method == 'PageRank':
            ground_truth = seq[snapshot+1].pr.T[idx]
            train_data.append( torch.tensor( ground_truth[nzero_idx] ).double() )
            eval_data.append( torch.tensor( ground_truth[eval_idx] ).double() )
        elif method == 'GammaPageRank':
            ground_truth = seq[snapshot+1].gpr[idx]
            train_data.append( torch.tensor( ground_truth[nzero_idx] ).double() )
            eval_data.append( torch.tensor( ground_truth[eval_idx] ).double() )

    return feat_train_data, train_data, feat_eval_data, eval_data
