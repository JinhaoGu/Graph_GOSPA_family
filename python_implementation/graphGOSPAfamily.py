#Author: Jinhao Gu
#This code is a python implementation of the graph GOSPA metric family proposed in the paper
# "A family of graph GOSPA metrics for graphs with different sizes"
# by Jinhao Gu, Á. F. García-Fernández, Robert E. Firth, Lennart Svensson
import warnings

import numpy as np
import scipy.sparse as sps
from scipy.optimize import linprog


def sparse_from_index_pairs(rows,cols,value_of,shape):
    '''Build a sparse matrix that holds value_of(row,col) at the given pairs.

    The pairs are made unique first.  The dense code wrote a 1 into a mask at
    these pairs, so a repeated pair contributed a single 1, whereas a sparse
    constructor adds repeated entries together.

    rows, cols: index arrays of the same length.
    value_of:   callable applied to the unique (row,col) arrays.
    '''
    rows=np.asarray(rows,dtype=np.int64)
    cols=np.asarray(cols,dtype=np.int64)
    if rows.size==0:
        return sps.csr_matrix(shape,dtype=float)
    unique=np.unique(rows*shape[1]+cols)
    rows=unique//shape[1]
    cols=unique%shape[1]
    return sps.csr_matrix((value_of(rows,cols),(rows,cols)),shape=shape,dtype=float)


def computeLocCostPerTime(x,y,c,p):
    return np.linalg.norm(x-y)**p


def locCostComp(X_attr,Y_attr,c,p):
    n_x=X_attr.shape[0]
    n_y=Y_attr.shape[0]
    tmpCost=c**p/2
    locCostMat=np.zeros((n_x+1,n_y+1))

    for i in range(n_x+1):
        if i<=n_x-1:# x not dummy
            for j in range(n_y+1):
                if j<=n_y-1: # y not dummy
                    locCostMat[i,j]=computeLocCostPerTime(X_attr[i,:],Y_attr[j,:],c,p)
                else:
                    locCostMat[i,j]=tmpCost
        
        else:
            for j in range(n_y): # x is dummy
                locCostMat[i,j]=tmpCost

    return locCostMat


def graph_gospa_metric_family(X_attr,Y_attr,X_adj,Y_adj,c,p,epsilon,beta,eta,flag=0):
    '''
    This function calculates the graph GOSPA metric family between two graphs using linear programming.
    Input:
    X_attr: NxD array of node attributes for graph X
    Y_attr: MxD array of node attributes for graph Y
    X_adj: NxN non-negative, symmetric adjacency matrix with zero diagonal
    Y_adj: MxM non-negative, symmetric adjacency matrix with zero diagonal
    c: penalty for missing or false nodes
    p: p-norm
    epsilon: penalty for edge mismatch
    beta: hyperparameter controlling the penalty for unassigned edges
    eta: hyperparameter controlling the penalty for half-assigned edges,
        with 0<=beta<=eta<=1, and eta>=0.5 when flag=0
    flag: integrality of the linear program (0 continuous, 1 integer)

    Returns:
    graph GOSPA metric family cost, localisation cost, miss node cost, false node cost,
    assigned edge cost, unassigned edge cost, half-assigned edge cost

    dxy is the p-th root of the total cost: the exact distance for flag=1,
    or its LP lower bound for flag=0. The LP generally does not satisfy the
    triangle inequality. The
    six cost components are the costs themselves, not their p-th roots, so
    they add up to dxy**p up to numerical tolerance for every p. For p=1,
    the costs and the distance are in the same units and add up directly.
    For p>1 the components still describe costs before taking the root.

    The decomposition is exact for an integer assignment (flag=1).  The
    continuous relaxation (flag=0) can return a fractional assignment, for
    which assigned_edge_cost can be negative; the function warns when this
    happens.
    '''
    parameters=dict(c=c,p=p,epsilon=epsilon,beta=beta,eta=eta,flag=flag)
    for name,value in parameters.items():
        scalar=np.asarray(value)
        if scalar.ndim!=0 or scalar.dtype.kind not in 'biuf' or not np.isfinite(scalar):
            raise ValueError('%s must be a finite real scalar.'%name)
        parameters[name]=float(scalar)
    c,p,epsilon,beta,eta,flag=(parameters[name] for name in
                              ('c','p','epsilon','beta','eta','flag'))
    if c<=0 or epsilon<=0 or p<1:
        raise ValueError('The graph GOSPA metric family requires c>0, epsilon>0 and p>=1.')
    if flag not in (0,1):
        raise ValueError('flag must be 0 (continuous) or 1 (integer).')
    if not 0<=beta<=eta<=1:
        raise ValueError('The graph GOSPA metric family requires '
                         '0<=beta<=eta<=1. Got beta=%r, eta=%r.'%(beta,eta))
    if flag==0 and eta<0.5:
        raise ValueError('The LP relaxation (flag=0) requires eta>=0.5 '
                         'for non-negative costs. Use flag=1 for eta<0.5.')

    arrays=[]
    for name,value in (('X_attr',X_attr),('Y_attr',Y_attr),
                       ('X_adj',X_adj),('Y_adj',Y_adj)):
        array=np.asarray(value)
        if array.ndim!=2 or array.dtype.kind not in 'biuf':
            raise ValueError('%s must be a two-dimensional real numeric array.'%name)
        # Convert before subtraction to avoid overflow in integer attributes.
        array=array.astype(float,copy=False)
        if not np.isfinite(array).all():
            raise ValueError('%s must contain only finite values.'%name)
        arrays.append(array)
    X_attr,Y_attr,X_adj,Y_adj=arrays
    if X_attr.shape[1]!=Y_attr.shape[1]:
        raise ValueError('X_attr and Y_attr must have the same feature dimension.')
    for name,attr,adj in (('X',X_attr,X_adj),('Y',Y_attr,Y_adj)):
        n=attr.shape[0]
        if adj.shape!=(n,n):
            raise ValueError('%s_adj must be square and match the number of attribute rows.'%name)
        if (adj<0).any() or not np.array_equal(adj,adj.T) or np.any(np.diag(adj)!=0):
            raise ValueError('%s_adj must be non-negative and symmetric with zero diagonal '
                             '(an undirected graph without self-loops).'%name)

    n_x=len(X_adj)
    n_y=len(Y_adj)
    if n_x==0 and n_y==0:
        zero=0.0
        return zero,zero,zero,zero,zero,zero,zero
    DAB=locCostComp(X_attr,Y_attr,c,p)
    
    nxny=n_x*n_y
    nxny2=(n_x+1)*(n_y+1)
    WLen=nxny2
    h1Len=nxny
    h2Len=nxny
    eLen=2
    uLen=1
    nParam=WLen+eLen+h1Len+h2Len+WLen+uLen

    WPos=np.arange(WLen)
    e1Pos=np.arange(WLen,WLen+1)
    e2Pos=np.arange(WLen+1,WLen+2)
    h1Pos=np.arange(WLen+eLen,WLen+eLen+h1Len)
    h2Pos=np.arange(WLen+eLen+h1Len,WLen+eLen+h1Len+h2Len)
    wPos=np.arange(WLen+eLen+h1Len+h2Len,WLen+eLen+h1Len+h2Len+WLen)
    uPos=np.arange(WLen+eLen+h1Len+h2Len+WLen,WLen+eLen+h1Len+h2Len+WLen+uLen)

    
    index_y=np.tile([i*(n_x+1) + n_x for i in range(n_y)],(n_y,1))
    index_y=index_y.flatten(order='F')
    index_x=np.tile([i*(n_x+1) + n_x for i in range(n_y)],(1,n_y))
    index_x=index_x.flatten('F')
    A_y=Y_adj.flatten(order='F')
    Q_y=sps.csr_matrix((A_y,(index_x,index_y)),shape=(WLen,WLen))


    index1_x=np.tile([i+(n_x+1)*n_y  for i in range(n_x)],(1,n_x))
    index1_x=index1_x.flatten('F')
    index1_y=np.tile([i+(n_x+1)*n_y  for i in range(n_x)],(n_x,1))
    index1_y=index1_y.flatten(order='F')

    A_x=X_adj.flatten(order='F')
    Q_x=sps.csr_matrix((A_x,(index1_x,index1_y)),shape=(WLen,WLen))

    Q=Q_x+Q_y
    
    #Compute Q+ and Q-
    Q_plus=np.asarray(Q.maximum(0).sum(axis=1)).reshape(1,WLen)
    Q_minus=np.asarray(Q.minimum(0).sum(axis=1)).reshape(1,WLen)
    
    
    #Compute coefficients for half assigned edges 
    one_nx=np.ones((n_x,1))
    sum_r_Ax=np.dot(X_adj,one_nx)

    one_ny=np.ones((1,n_y))
    sum_c_Ay=np.dot(one_ny,Y_adj)

    idx_Ax=[i+(n_x+1)*n_y  for i in range(n_x)]
    idx_Ay=[i*(n_x+1) + n_x for i in range(n_y)]

    r1_Ax=sps.csr_matrix((np.squeeze(sum_r_Ax,axis=1),(np.zeros(n_x),idx_Ax)),shape=(1,nParam))
    r1_Ay=sps.csr_matrix((np.squeeze(sum_c_Ay,axis=0),(np.zeros(n_y),idx_Ay)),shape=(1,nParam))

    r1=r1_Ax+r1_Ay
    
    ############# Objective function ################
    f=np.zeros([nParam,1])
    f[WPos]=np.reshape(DAB,(WLen,1),order='F')
    f[e1Pos]=epsilon**p/4
    f[e2Pos]=epsilon**p/4
    f[wPos]=(beta/2-eta+1/2)*epsilon**p
    f[uPos]=(eta-1/2)*epsilon**p

    ###########Equality constraint############
    #Constraint 1
    index_x=np.tile(np.arange(n_y),(n_x+1,1))
    index_x=index_x.flatten(order='F')
    index_y=np.arange(n_y*(n_x+1))
    index_y=index_y.flatten(order='F')

    Aeq1=sps.csr_matrix((np.ones(len(index_x)),(index_x,index_y)),shape=(n_y,nParam))
    beq1=np.ones([n_y,1])

    #Constraint 2
    index_x=np.tile(np.arange(n_x),(n_y+1,1)).T
    index_x=index_x.flatten(order='F')

    index_y=np.tile(np.arange(n_y+1)*(n_x+1),(n_x,1))
    index_y2=np.tile(np.arange(n_x),(np.size(index_y,1),1)).T
    index_y=index_y+index_y2
    index_y=index_y.flatten(order='F')
    index_y=index_y.astype(int)
    Aeq2=sps.csr_matrix((np.ones(len(index_x)),(index_x,index_y)),shape=(n_x,nParam))
    beq2=np.ones([n_x,1])

    #Constraint 3
    ones_u=sps.csr_matrix((-np.ones(uLen),(np.zeros(uLen,dtype=int),uPos)),shape=(1,nParam))
    Aueq=r1+ones_u
    bueq=np.zeros((1,1))

    #All equality constraints
    Aeq=sps.vstack((Aeq1,Aeq2,Aueq)).tocsr()
    beq=np.vstack((beq1,beq2,bueq))

    
    
    ###########Inequality constraint############
    #inequality constraints for linear terms

    #Constraint 1
    # sum(h1)-e1<=0 and sum(h2)-e2<=0
    Ae1=sps.csr_matrix((np.hstack((np.ones(h1Len),[-1.0])),
                        (np.zeros(h1Len+1,dtype=int),np.hstack((h1Pos,e1Pos)))),
                       shape=(1,nParam))
    Ae2=sps.csr_matrix((np.hstack((np.ones(h2Len),[-1.0])),
                        (np.zeros(h2Len+1,dtype=int),np.hstack((h2Pos,e2Pos)))),
                       shape=(1,nParam))

    A1=sps.vstack((Ae1,Ae2))


    #Constraint 2
    index_1_x=np.tile(np.arange(nxny),(n_x,1))
    index_1_x=index_1_x.flatten(order='F')
    index_1_x=index_1_x.astype(int)
    index_1_y=np.tile(np.arange(n_y)*(n_x+1),(n_x,1))
    index_1_y2=np.tile(np.arange(n_x),(np.size(index_1_y,1),1)).T
    index_1_y=index_1_y+index_1_y2
    index_1_y=np.tile(index_1_y,(1,n_x))
    index_1_y=np.reshape(index_1_y,(nxny*n_x),order='F')
    index_1_y=index_1_y.astype(int)
    # Row r repeats row r//n_y of X_adj, padded with a zero column for the
    # dummy node and tiled once per node of Y.  The trailing n_x+1 columns of
    # the block are zero, which takes the width up to WLen.
    X_pad=np.hstack((X_adj,np.zeros([n_x,1])))
    A_adj_1=sparse_from_index_pairs(index_1_x,index_1_y,
                                    lambda r,c: X_pad[r//n_y,c%(n_x+1)],
                                    (nxny,WLen))

    index_2_x=np.tile(np.arange(nxny),(n_y,1))
    index_2_x=index_2_x.flatten(order='F')
    index_2_y=np.tile(np.arange(n_y)*(n_x+1),(n_x,1))
    index_2_y2=np.tile(np.arange(n_x),(np.size(index_2_y,1),1)).T
    index_2_y=index_2_y+index_2_y2
    index_2_y=np.tile(index_2_y,(1,n_y)).T
    index_2_y=index_2_y.flatten(order='F')
    index_2_y=index_2_y.astype(int)
    # Column c repeats column c//(n_x+1) of Y_adj transposed.
    A_adj_2=sparse_from_index_pairs(index_2_x,index_2_y,
                                    lambda r,c: Y_adj[c//(n_x+1),r%n_y],
                                    (nxny,WLen))

    zeros_e=sps.csr_matrix((nxny,eLen))
    zeros_tail=sps.csr_matrix((nxny,WLen+uLen))
    eye_h1=sps.hstack((sps.eye(h1Len,format='csr'),sps.csr_matrix((h1Len,h2Len))))
    eye_h2=sps.hstack((sps.csr_matrix((h2Len,h1Len)),sps.eye(h2Len,format='csr')))

    A2_adj=sps.hstack((A_adj_1-A_adj_2,zeros_e,-eye_h1,zeros_tail))
    A3_adj=sps.hstack((A_adj_2-A_adj_1,zeros_e,-eye_h1,zeros_tail))


    #Contraint 3

    index_1_x=np.tile(np.arange(nxny),(n_y,1))
    index_1_x=index_1_x.flatten(order='F')

    index_1_y=np.tile(np.arange(n_y)*(n_x+1),(n_x,1))
    index_1_y2=np.tile(np.arange(n_x),(np.size(index_1_y,1),1)).T
    index_1_y=index_1_y+index_1_y2
    index_1_y=np.tile(index_1_y.T,(1,n_y))
    index_1_y=index_1_y.flatten(order='F')
    index_1_y=index_1_y.astype(int)
    ind=np.tile(np.arange(n_y)*(n_x+1),(n_x,1))
    ind2=np.tile(np.arange(n_x),(np.size(ind,1),1)).T
    ind=ind+ind2
    ind=ind.flatten(order='F')
    ind=ind.astype(int)
    # Column ind[i] holds column i of Y_adj, expanded n_x times along both
    # axes.  inverse_ind maps a column of the block back to that index.
    inverse_ind=np.full(WLen,-1,dtype=np.int64)
    inverse_ind[ind]=np.arange(nxny)
    A_adj_3=sparse_from_index_pairs(
        index_1_x,index_1_y,
        lambda r,c: np.where(inverse_ind[c]>=0,
                             Y_adj[r//n_x,np.maximum(inverse_ind[c],0)//n_x],0.0),
        (nxny,WLen))



    index_2_x=np.tile(np.arange(nxny),(n_x,1))
    index_2_x=index_2_x.flatten(order='F')
    index_2_y=np.tile(np.arange((n_x+1)*n_y,step=n_x+1),(n_x,1))
    index_2_y2=np.tile(np.arange(n_x),(n_y,1)).T
    index_2_y=index_2_y+index_2_y2
    index_2_y=np.tile(index_2_y.T,(1,n_x)).T
    index_2_y=index_2_y.flatten(order='F')
    index_2_y=index_2_y.astype(int)


    # X_adj transposed, zero padded for the dummy node and tiled n_y times
    # along both axes.
    XT_pad=np.hstack((X_adj.T,np.zeros([n_x,1])))
    A_adj_4=sparse_from_index_pairs(index_2_x,index_2_y,
                                    lambda r,c: XT_pad[r%n_x,c%(n_x+1)],
                                    (nxny,WLen))

    A4_adj=sps.hstack((A_adj_3-A_adj_4,zeros_e,-eye_h2,zeros_tail))
    A5_adj=sps.hstack((A_adj_4-A_adj_3,zeros_e,-eye_h2,zeros_tail))

    A_x=sps.vstack((A1,A2_adj,A3_adj,A4_adj,A5_adj))


    diag_plus=sps.diags(Q_plus.ravel(),format='csr')
    diag_minus=sps.diags(Q_minus.ravel(),format='csr')
    zeros_mid=sps.csr_matrix((WLen,eLen+h1Len+h2Len))
    eye_W=sps.eye(WLen,format='csr')
    zeros_u=sps.csr_matrix((WLen,uLen))

    Aw1=sps.hstack((-diag_plus,zeros_mid,eye_W,zeros_u))# wi-Qi_plus*xi <=0, i=1,...n

    Aw2=sps.hstack((diag_minus,zeros_mid,-eye_W,zeros_u))# -wi+Qi_minus*xi <=0, i=1,...n

    Q_=sps.hstack((Q,sps.csr_matrix((WLen,eLen+h1Len+h2Len+WLen+uLen)))) # extend Q to nParam

    Aw3=-Q_+sps.hstack((-diag_minus,zeros_mid,eye_W,zeros_u))  # wi-(Q*X)[i]-Qi_minus*xi <=-Qi_minus
    Aw4=Q_+sps.hstack((diag_plus,zeros_mid,-eye_W,zeros_u)) # (Q*X)[i] + Qi_plus*xi- wi <=Qi_plus 

    A=sps.vstack((A_x,Aw1,Aw2,Aw3,Aw4)).tocsr()
    lenb,_=A.shape
    b=np.vstack((np.zeros([lenb-2*WLen,1]),-Q_minus.transpose(),Q_plus.transpose()))
    f=f.flatten()
    # Only the assignment variables W may be constrained to be integer.  The
    # auxiliary variables e, h1, h2, w and u are continuous by construction;
    # rounding them changes the problem and gives a wrong cost for weighted
    # adjacency matrices.
    integrality=np.zeros(nParam)
    integrality[WPos]=flag
    res=linprog(f,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,method='highs',integrality=integrality)# 0 continous, 1 integer.

    if not res.success:
        raise RuntimeError('The linear programme did not solve: %s'%res.message)

    dxy=res.fun
    W=res.x
    tolerance=1e-9*max(1.0,abs(dxy))
    if not np.isfinite(dxy) or dxy < -tolerance:
        raise RuntimeError('The linear programme returned an invalid total cost: %r.'%dxy)
    if dxy<0:
        dxy=0.0
    Wx=np.reshape(W[0:nxny2],(n_x+1,n_y+1),order='F')
    loc_cost=np.sum(np.multiply(DAB[0:n_x,0:n_y],Wx[0:n_x,0:n_y]))
    false_cost=np.sum(np.multiply(DAB[n_x,0:n_y],Wx[n_x,0:n_y]))
    miss_cost=np.sum(np.multiply(DAB[0:n_x,n_y],Wx[0:n_x,n_y]))
    
    # Read the edge costs from the variables that the linear programme itself
    # prices.  sum(w) counts the edges with both end points unassigned and u
    # counts the edge end points of unassigned nodes, so u-sum(w) counts the
    # half assigned edges.  Constraint Aw1 gives sum(w)<=u, so u-sum(w)>=0.
    # The quadratic expressions in Wx that these replace agree with w and u
    # only when Wx is an assignment matrix.  The continuous relaxation
    # (flag=0) can return a fractional Wx, for which they disagree and the
    # cost components no longer add up to dxy**p.
    sum_w=np.sum(W[wPos])
    u=W[uPos].item()
    half_assigned_edges=u-sum_w
    unassigned_edge_cost=beta/2*epsilon**p * sum_w
    half_assigned_edge_cost=eta*epsilon**p * half_assigned_edges
    assigned_edge_cost=epsilon**p/4 * (W[e1Pos].item()+W[e2Pos].item())\
        -0.5*epsilon**p * half_assigned_edges

    # A cost that is negative only by the solver's tolerance is floating point
    # noise, so snap it to zero.  A cost that the relaxation makes genuinely
    # negative is orders of magnitude larger and is left alone; measured over
    # random graphs, tolerance noise stays below 1e-12 of the total while a
    # genuine negative sits around 0.3 of it.
    if -1e-9*max(1.0,abs(dxy))<assigned_edge_cost<0:
        assigned_edge_cost=0.0

    if flag==0 and np.any(np.abs(Wx-np.round(Wx))>1e-6):
        warnings.warn('The continuous relaxation returned a fractional '
                      'assignment. dxy is still a valid lower bound of the '
                      'metric, but the cost components are exact only for an '
                      'integer assignment, and assigned_edge_cost can be '
                      'negative. Use flag=1 for an exact decomposition.',
                      RuntimeWarning,stacklevel=2)

    # The validated objective is non-negative in exact arithmetic; materially
    # negative totals are rejected above. Node costs are non-negative as well.
    # The same is true
    # of the unassigned and half assigned edge costs, because w>=0 and
    # u-sum(w)>=0.  assigned_edge_cost is a difference of two of the linear
    # programme's variables and the relaxation can make it genuinely negative,
    # so it is reported as it is; see the warning above.
    loc_cost=np.maximum(loc_cost,0)
    miss_cost=np.maximum(miss_cost,0)
    false_cost=np.maximum(false_cost,0)
    unassigned_edge_cost=np.maximum(unassigned_edge_cost,0)
    half_assigned_edge_cost=np.maximum(half_assigned_edge_cost,0)

    # Only dxy is rooted; it is the metric.  The components stay as costs so
    # that they add up to dxy**p for every p.
    return dxy**(1/p),loc_cost,miss_cost,false_cost,\
        assigned_edge_cost,unassigned_edge_cost,half_assigned_edge_cost
