import numpy as np
import math
import networkx as nx
import matplotlib.pyplot as plt
from graph_matrices import graph_matrices

## CLASS
class graph_init:
    def __init__(self, num_clust, nodes_clust, p_intra, p_inter, seed, alpha):
        self.A, self.clust_memb = init_graph(num_clust, nodes_clust, p_intra, p_inter)
        self.pr = PageRank_vec( seed, alpha, np.array(self.A) )
        self.gpr, self.P2, self.Op_shift, self.D2 = GammaPageRank_vec(seed, alpha, np.array(self.A), np.array(self.clust_memb) ) 
        self.time = 0
        self.D, self.Dinv, self.P = graph_matrices(np.array(self.A))
    
class graph_evolved:
    def __init__(self, graph_input, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha):

        # perturbation
        self.A, self.clust_memb = perturbation(np.array(graph_input.A), np.array(graph_input.clust_memb), p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed)

        # pagerank
        self.pr = PageRank_vec( seed, alpha, np.array(self.A) )
        
        # gamma pagerank
        self.gpr, self.P2, self.Op_shift, self.D2 = GammaPageRank_vec(seed, alpha, np.array(self.A), np.array(self.clust_memb) ) 
        
        # time step
        self.time = graph_input.time + 1

        # graph matrices
        self.D, self.Dinv, self.P = graph_matrices(np.array(self.A))

## METHODS
def init_graph(num_clust, nodes_clust, p_intra, p_inter):
        G = nx.planted_partition_graph( num_clust, nodes_clust, p_intra, p_inter, seed=42)
        #G = nx.grid_graph(dim=[int(math.sqrt(nodes_clust)), int(math.sqrt(nodes_clust))])
        A = np.array( nx.adjacency_matrix(G).todense() ) 

        ### Embed the initial graph in a matrix that is much larger
        N = A.shape[0]
        Amat = np.zeros([6*N,6*N]); Amat[0:N,0:N] = A
        clust_memb = np.zeros([6*N])
        for j in range(num_clust):
            clust_memb[j*nodes_clust:(j+1)*nodes_clust] = j+1

        return Amat, clust_memb

def perturbation(A_init, clust_memb, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed):
    
    A1, clust_memb1 = nodes_leaving(np.array(A_init), np.array(clust_memb), p_nleave, seed)
    A2, clust_memb2 = edges_leaving(np.array(A1), np.array(clust_memb1), p_eleave_in, p_eleave_out)
    A3, clust_memb3 = nodes_joining(np.array(A2), np.array(clust_memb2), p_njoin, p_intra, p_inter)
    A4, clust_memb4 = edges_joining(np.array(A3), np.array(A1), np.array(clust_memb3), np.array(clust_memb1), p_ejoin_in, p_ejoin_out)
    return A4, clust_memb4

def nodes_leaving(Amat, clust_memb, p_nleave, seed):
    # count number of clusters
    num_clust = int(max(clust_memb))
    num_nodes_leaving = np.random.poisson(p_nleave, num_clust)

    # from the indices of each class, select that amount at random
    for i in range(num_clust):
        idx = np.random.permutation( np.where(clust_memb == i+1)[0] )

        # dont remove the seed nodes
        new_idx = np.where(np.in1d(idx,seed))[0] 
        idx = np.delete( idx, new_idx ) 

        idx_to_del = idx[0:num_nodes_leaving[i]]
        Amat[ idx_to_del, : ] = 0; Amat[:, idx_to_del] = 0;
        clust_memb[ idx_to_del ]= -1;

    return Amat, clust_memb

def edges_leaving(Amat, clust_memb, p_eleave_in, p_eleave_out):
    # count number of clusters
    num_clust = int(max(clust_memb))

    # for each cluster list all the edges and some of them to zero
    for i in range(num_clust):
        # which nodes belong the the cluster
        idx = np.where(clust_memb == i+1)[0]

        # extract submatrix of the cluster
        tmp = Amat[idx,:]; tmp = tmp[:,idx];
        tmp_Amat = tmp - np.tril(tmp) 

        # where are the edges
        nzero_idx = np.where(tmp_Amat)
        edge_start =  idx[nzero_idx[0]] 
        edge_end =  idx[nzero_idx[1]] 

        # suprime some of them
        list_to_del = np.where(np.random.binomial(1, p_eleave_in, nzero_idx[1].shape[0]))[0]
        Amat[edge_start[list_to_del], edge_end[list_to_del]] -= 1
        Amat[edge_end[list_to_del], edge_start[list_to_del]] -= 1

        # which nodes belong to opposite clusters 
        idx_op = np.where( (clust_memb != i+1) & (clust_memb > 0))[0]
        tmp = Amat[idx,:]; tmp_Amat = tmp[:,idx_op]

        # where are the edges
        nzero_idx = np.where(tmp_Amat)        
        edge_start =  idx[nzero_idx[0]] 
        edge_end =  idx_op[nzero_idx[1]] 
       
        # suprime some of them
        list_to_del = np.where(np.random.binomial(1, p_eleave_out, nzero_idx[1].shape[0]))[0]
        Amat[edge_start[list_to_del], edge_end[list_to_del]] -= 1
        Amat[edge_end[list_to_del], edge_start[list_to_del]] -= 1

    return Amat, clust_memb

def nodes_joining(Amat, clust_memb, p_njoin, p_intra, p_inter):
    # count number of clusters
    num_clust = int(max(clust_memb))
    num_nodes_joining = np.random.poisson(p_njoin, num_clust)

    # to each class, add some amount at random
    for i in range(num_clust):

        # where are the nodes available to join
        idx_free = np.where(clust_memb == 0)[0] 
        idx_to_join = idx_free[0:num_nodes_joining[i]]
        idx_op = np.where( (clust_memb != i+1) & (clust_memb > 0))[0]

        # place a link between this node and the cluster nodes with some probability 
        for j in range( idx_to_join.shape[0] ):

            # extract the cluster nodes
            idx_class = np.where(clust_memb == i+1)[0]

            # where are the edges to add between this node and the cluster 
            list_to_add = np.where( np.random.binomial(1, p_intra, idx_class.shape[0]) )[0]
            edge_start = idx_to_join[j] 
            edge_end = idx_class[list_to_add]

            # place the edges
            Amat[edge_start, edge_end] = 1
            Amat[edge_end, edge_start] = 1

            # extract the external cluster nodes
            list_to_add = np.where( np.random.binomial(1, p_inter, idx_op.shape[0]) )[0]
            edge_start = idx_to_join[j] 
            edge_end = idx_op[list_to_add]

            # place the edges
            Amat[edge_start, edge_end] = 1
            Amat[edge_end, edge_start] = 1

            # set this node as belonging to the cluster
            clust_memb[ idx_to_join[j] ] = i+1
    
    return Amat, clust_memb

#
def edges_joining(Amat, Ainit, clust_memb, clust_memb_init, p_ejoin_in, p_ejoin_out):
    # count number of clusters
    num_clust = int(max(clust_memb_init))

    # for each cluster list all the edges and some of them to zero
    for i in range(num_clust):
        # which nodes belong the the cluster
        idx = np.where(clust_memb_init == i+1)[0]
        
        tmp_Ainit = np.zeros([len(idx),len(idx)])
        for j in range(len(idx)):
            randseq = np.random.binomial(1, p_ejoin_in, len(idx) )
            tmp_Ainit[j,:] = randseq
            
        # extract submatrix of the cluster
        tmp_Ainit = tmp_Ainit - np.tril(tmp_Ainit) 

        # where are we missing edges
        nzero_idx = np.where(tmp_Ainit == 1)
        new_edge_start = idx[nzero_idx[0]]
        new_edge_end = idx[nzero_idx[1]]
        
        Amat[new_edge_start, new_edge_end] += 1
        Amat[new_edge_end, new_edge_start] += 1
        
        # which nodes belong to opposite clusters 
        idx_op = np.where( (clust_memb_init != i+1) & (clust_memb_init > 0))[0]

        tmp_Ainit = np.zeros([len(idx),len(idx_op)])
        for j in range(len(idx)):
            randseq = np.random.binomial(1, p_ejoin_out, len(idx_op) )
            tmp_Ainit[j,:] = randseq

        # where are the missing edges
        nzero_idx = np.where(tmp_Ainit == 1)        
        new_edge_start = idx[nzero_idx[0]] 
        new_edge_end = idx_op[nzero_idx[1]] 
       
        # suprime some of them
        Amat[new_edge_start, new_edge_end] += 1
        Amat[new_edge_end, new_edge_start] += 1

    return Amat, clust_memb

def PageRank_vec(seed,alpha,A):
    N = A.shape[0]
    init_dist = np.zeros([1,N]); init_dist[0,seed] = 1;
    D, D_inv, P = graph_matrices( A )
    f = init_dist
    for k in range(100):
        f = alpha*init_dist + (1-alpha)*f.dot(P)
    return f

def GammaPageRank_vec(seed, alpha, A, memb_clust):
 
    # compute mu parameter from alpha
    mu = alpha/(1-alpha)
    N = A.shape[0]

    # initial dist
    init_dist = np.zeros([N,1]); init_dist[seed,0] = 1;

    # variable to store result
    g_pr = np.zeros([N,1]);
    g_Op = np.zeros([N,N]);
    g_D = np.zeros([N,N]);
    g_Op_shift = np.zeros([N,N])

    # graph matrices
    D, D_inv, P = graph_matrices( A )
    
    # nodes active in the graph
    idx = np.where(memb_clust > 0)[0]
    
    # Laplacian and related matrices of this subgraph region
    Lap = D[np.ix_(idx,idx)]-A[np.ix_(idx,idx)]
    L2 = Lap.dot(Lap)
    D2 = np.diag(np.diag(L2))
    A2 = D2 - L2
    D2inv = np.linalg.pinv(D2)
    
    # gamma pr method
    Op = L2.dot(D2inv)
    Pg = A2.dot(D2inv)
    Kernel = Op + mu*np.eye(len(idx))
    tmp_g_pr = mu*(np.linalg.solve(Kernel, init_dist[idx]))
    g_pr[idx] = tmp_g_pr
    g_Op[np.ix_(idx,idx)] = Pg
    g_D[np.ix_(idx,idx)] = D2

    # shifted operator
    tau = 10/2;
    Op_shift = (1/tau)*(Op - tau*np.eye(len(idx))) 
    g_Op_shift[np.ix_(idx,idx)] = Op_shift

    return g_pr, g_Op, g_Op_shift, g_D

def graph_sequence(num_graphs, num_clust, nodes_clust, p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha):
    g_seq = [graph_init(num_clust,nodes_clust,p_intra,p_inter,seed,alpha)]

    for i in range(num_graphs):
        g_seq.append( graph_evolved( g_seq[i], p_njoin, p_nleave, p_ejoin_in, p_ejoin_out, p_eleave_in, p_eleave_out, p_switch, p_intra, p_inter, seed, alpha) )

    return g_seq
