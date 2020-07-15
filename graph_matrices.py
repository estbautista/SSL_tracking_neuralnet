import numpy as np
import networkx as nx

def graph_matrices(Adj):
    D = np.diag( np.sum(Adj,axis=1) )
    D_inv = np.linalg.pinv(D)
    P = np.dot(D_inv, Adj)
    return D, D_inv, P

def create_graph(num_clusters,num_nodes,pin,pout,num_graphs,pnew):
    G1 = nx.planted_partition_graph(num_clusters, num_nodes, pin, pout, seed=42)
    A = [np.array( nx.adjacency_matrix(G1).todense()) ]

    for i in range(num_graphs):
        G = nx.planted_partition_graph(num_clusters, num_nodes, pnew, pnew)
        A.append( A[i] + np.array( nx.adjacency_matrix(G).todense()) )
    return A

def transition_mat(Adj):
    num_graphs = len(Adj)
    Pmat = []
    Dmat = []
    Dimat = []
    for i in range(num_graphs):
        D, D_inv, P = graph_matrices( Adj[i] )
        Pmat.append( P ) 
        Dmat.append( D )
        Dimat.append( D_inv )
    return Dmat, Dimat, Pmat
