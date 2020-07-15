import numpy as np 
import random
import math
import torch
import scipy.io as sio
from accuracy import accuracy
from model import Net
from graph_matrices import graph_matrices, create_graph, transition_mat
from graph_data import graph_sequence
from testing import prepare_data_test, model_testing, analytic_update, no_update
from ranking import PageRank_vec
from training import training, prepare_data_train
from numpy import inf
import matplotlib.pyplot as plt

#####################
## Graph parameters 
p_intra = 0.5
p_inter = 0.05
p_njoin = 1
p_nleave = 1
p_ejoin_in = p_intra/20
p_ejoin_out = p_inter/20
p_eleave_in = p_intra/20
p_eleave_out = p_inter/20
p_switch = 0
num_time_steps = 170
num_clust = 2
nodes_clust = 100

## Network Paraterms
np.random.seed(50)
hops = 3
weight_decay = 1e-9
# previous lr was 8e6
lr = 1e-6 # normal pr
epochs = 60000 # normal pr
#lr = 5e-5 # gamma pr
#epochs = 20000 # gamma pr
snaps = 20
do_train = False
method = 'PageRank'

## Ranking parameters
alpha = 0.1
seed = [1,10]

#####################
# Create graph matrices
seq = graph_sequence(num_time_steps, num_clust, nodes_clust, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha)

if do_train == True: 
    ####################
    # prepare data for training
    feat_train_data, train_data, feat_eval_data, eval_data  = prepare_data_train(seq, snaps, method, hops)

    ##################### 
    # model
    model = Net(2*hops+1,1).double()

    #####################
    # train
    train_model = training(model, feat_train_data, train_data, feat_eval_data, eval_data, lr, epochs, 10, weight_decay)
    torch.save(train_model.state_dict(), 'model_params_save')

#######################
##load trained model 
model_eval = Net(2*hops+1,1).double()
model_eval.load_state_dict(torch.load('model_params_save'))

#######################
# assess the model
err_magnitude, err_degrees, predictions = model_testing(model_eval, seq, snaps, num_time_steps, hops, method)

########################
# Evaluate the analytic formula
err_magnitude_analytic, err_degrees_analytic, predictions_analytic = analytic_update(seq, snaps, num_time_steps, hops*2, alpha, method)

########################
# Assess no update
err_magnitude_noup, err_degrees_noup = no_update(seq, snaps, num_time_steps, method)

## set nodes to zero
ix = np.where(seq[-1].clust_memb == -1)[0];
predictions[-1][ix] = 0

fig_n = plt.figure()
plt.plot(np.array( err_magnitude), label='proposed')
plt.plot(np.array( err_magnitude_analytic ), label='analytic update')
#plt.plot(np.array( err_magnitude_noup ), label='noup')
plt.legend()
plt.xlabel('Graph realization (t)');
plt.ylabel('Error in L2-norm')
plt.title('Std. PageRank')
plt.savefig('Figs/error_mag.png',dpi=300)
plt.close()

plt.show()
plt.plot(np.array(err_degrees), label='proposed')
plt.plot(np.array(err_degrees_analytic), label='analytic update')
#plt.plot(np.array(err_degrees_noup), label='noup')
plt.legend()
plt.xlabel('Graph realization (t)');
plt.ylabel('Difference in angle')
plt.title('Std. PageRank')
plt.savefig('Figs/error_angle.png',dpi=300)
plt.close()

fig4 = plt.figure()
plt.plot( predictions[-1] )
plt.plot( predictions_analytic[-1] )
plt.plot( seq[-1].pr.T)
plt.savefig('Figs/prediction.png')
plt.close()


########################
# Store result
dictio = {'err_net': err_magnitude  ,'err_an': err_magnitude_analytic, 'err_noup': err_magnitude_noup}
sio.savemat('Res_L1Thesis.mat', dictio)
