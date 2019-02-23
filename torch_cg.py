import torch

def spmv(A,x):
    return torch.mm(A,x[:,None]).squeeze()

def cg(A,b,M,x0,maxiter=None,rtol=1e-3,atol=1e-3):
    if maxiter is None:
        maxiter = 2*A.shape[0]
    rs, zs, ps, xs = [], [], [None], [x0]
    k = 0
    rs += [b-spmv(A,xs[k])]
    optimal = False
    while k < maxiter:
        zs += [spmv(M,rs[k])]
        k = k + 1
        if k == 1:
            ps += [zs[0]]
        else:
            beta = rs[k-1]@zs[k-1]/(rs[k-2]@zs[k-2])
            ps += [zs[k-1] + beta*ps[k-1]]
        alpha = (rs[k-1]@zs[k-1])/(ps[k]@spmv(A,ps[k]))
        xs += [xs[k-1] + alpha*ps[k]]
        rs += [rs[k-1] - alpha*spmv(A,ps[k])]
        if torch.norm(rs[-1]) <= max(rtol*torch.norm(b),atol):
            optimal = True
            break
        if k > 2:
            rs[k-3] = zs[k-3] = ps[k-3] = xs[k-3] = None
    return xs[-1], {"optimal":optimal,"|r|":torch.norm(rs[-1]).item()}

def sparse_numpy_to_torch(A):
    A = A.tocoo()
    values = A.data
    indices = np.vstack((A.row,A.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = A.shape
    return torch.sparse.FloatTensor(i,v,torch.Size(shape))

if __name__ == '__main__':
    import networkx as nx
    from scipy import sparse
    import numpy as np

    n = 10000
    A = nx.laplacian_matrix(nx.gnm_random_graph(n,15*n))+.01*sparse.eye(n)
    M = sparse.diags(1./A.diagonal())
    A = sparse_numpy_to_torch(A)
    M = sparse_numpy_to_torch(M)
    b = torch.randn(n)

    x, info = cg(A,b,M,torch.zeros(n))
    print (torch.mm(A,x[:,None]).squeeze()-b)
    print (info)