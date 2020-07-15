from graph_matrices import graph_matrices

def PageRank_vec(seed,alpha,A):
    Pr = []
    num_graphs = len(A)
    for i in range(num_graphs):
        D, D_inv, P = graph_matrices( A[i] )
        f = seed
        for k in range(100):
            f = alpha*seed + (1-alpha)*f.dot(P)
        Pr.append( f )

    return Pr
