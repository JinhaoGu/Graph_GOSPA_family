#Author: Jinhao Gu
#This code is a python implementation of the graph GOSPA metric family proposed in the paper
# "A family of graph GOSPA metrics for graphs with different sizes"
# by Jinhao Gu, Á. F. García-Fernández, Robert E. Firth, Lennart Svensson
import numpy as np
import scipy.sparse as sps
from scipy.optimize import linprog


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
    X_adj: NxN adjacency matrix for graph X
    Y_adj: MxM adjacency matrix for graph Y
    c: penalty for missing or false nodes
    p: p-norm
    epsilon: penalty for edge mismatch
    beta: hyperparameter controlling the penalty for unassigned edges
    eta: hyperparameter controlling the penalty for half-assigned edges
    flag: integrality of the linear program (0 continuous, 1 integer)

    Returns:
    graph GOSPA metric family cost, localisation cost, miss node cost, false node cost,
    assigned edge cost, unassigned edge cost, half-assigned edge cost
    '''
    n_x=len(X_adj)
    n_y=len(Y_adj)
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
    Q_p=Q.todense()
    Q_m=Q.todense()
    Q_p[Q_p<0]=0
    Q_m[Q_m>0]=0

    Q_plus=np.sum(Q_p,axis=1).transpose()
    Q_minus=np.sum(Q_m,axis=1).transpose()
    
    
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

    Aeq1=np.zeros([n_y,nParam])
    Aeq1[index_x,index_y]=1
    beq1=np.ones([n_y,1])

    #Constraint 2
    index_x=np.tile(np.arange(n_x),(n_y+1,1)).T
    index_x=index_x.flatten(order='F')

    index_y=np.tile(np.arange(1,(n_x+1)/n_x*len(index_x),step=n_x+1),(n_x,1))-1
    index_y2=np.tile(np.arange(n_x),(np.size(index_y,1),1)).T
    index_y=index_y+index_y2
    index_y=index_y.flatten(order='F')
    index_y=index_y.astype(int)
    Aeq2=np.zeros([n_x,nParam])
    Aeq2[index_x,index_y]=1
    beq2=np.ones([n_x,1])

    #Constraint 3
    ones_u=np.zeros((1,nParam))
    ones_u[0,uPos]=-1
    Aueq=r1+ones_u
    bueq=np.zeros((1,1))

    #All equality constraints
    Aeq=np.vstack((Aeq1,Aeq2,Aueq))
    beq=np.vstack((beq1,beq2,bueq))

    
    
    ###########Inequality constraint############
    #inequality constraints for linear terms

    #Constraint 1
    index_minus_x=0
    index_minus_y=WLen+index_minus_x
    value_minus=-1
    index_one_x=np.zeros(nxny)
    index_one_x=index_one_x.astype(int)
    index_one_y=WLen+eLen+np.arange(nxny)
    Ae1=np.zeros((1,nParam))
    Ae1[index_one_x,index_one_y]=1
    Ae1[index_minus_x,index_minus_y]=-1

    index_minus_x=0
    index_minus_y=WLen+index_minus_x+1
    value_minus=-1
    index_one_x=np.hstack((index_minus_x,np.zeros(nxny)))
    index_one_x=index_one_x.astype(int)

    index_one_y=WLen+eLen+nxny+np.arange(nxny)
    index_one_y=np.hstack((index_minus_y,index_one_y))
    Ae2=np.zeros((1,nParam))
    Ae2[index_one_x,index_one_y]=1
    Ae2[index_minus_x,index_minus_y]=-1

    A1=np.vstack((Ae1,Ae2))


    #Constraint 2
    index_1_x=np.tile(np.arange(nxny),(n_x,1))
    index_1_x=index_1_x.flatten(order='F')
    index_1_x=index_1_x.astype(int)
    index_1_y=np.tile(np.arange(1,(n_x+1)/n_x*nxny,step=n_x+1),(n_x,1))-1
    index_1_y2=np.tile(np.arange(n_x),(np.size(index_1_y,1),1)).T
    index_1_y=index_1_y+index_1_y2
    index_1_y=np.tile(index_1_y,(1,n_x))
    index_1_y=np.reshape(index_1_y,(nxny*n_x),order='F')
    index_1_y=index_1_y.astype(int)
    mask=np.zeros([nxny,(n_x+1)*n_y])
    mask[index_1_x,index_1_y]=1

    A_adj1=np.tile(np.hstack((X_adj,np.zeros([n_x,1]))),(1,n_y))
    ind=np.tile(np.arange(n_x),(n_y,1))
    ind=ind.flatten(order='F')

    new_adj=np.zeros([nxny,(n_x+1)*n_y])

    for i in range(nxny):
        new_adj[i,:]=A_adj1[ind[i],:]
        
    A_adj_1=np.hstack((np.multiply(new_adj,mask),np.zeros([nxny,n_x+1])))

    index_2_x=np.tile(np.arange(nxny),(n_y,1))
    index_2_x=index_2_x.flatten(order='F')
    index_2_y=np.tile(np.arange(1,(n_x+1)/n_x*nxny,step=n_x+1),(n_x,1))-1
    index_2_y2=np.tile(np.arange(n_x),(np.size(index_2_y,1),1)).T
    index_2_y=index_2_y+index_2_y2
    index_2_y=np.tile(index_2_y,(1,n_y)).T
    index_2_y=index_2_y.flatten(order='F')
    index_2_y=index_2_y.astype(int)
    mask=np.zeros([nxny,(n_x+1)*n_y])
    mask[index_2_x,index_2_y]=1

    A_adj2=np.tile(Y_adj.T,(n_x,1))
    ind=np.tile(np.arange(n_y),(n_x+1,1))
    ind=ind.flatten(order='F')
    new_adj=np.zeros([nxny,(n_x+1)*n_y])

    for i in range((n_x+1)*n_y):
        new_adj[:,i]=A_adj2[:,ind[i]]

    A_adj_2=np.hstack((np.multiply(new_adj,mask),np.zeros([nxny,n_x+1])))
    A2_adj=np.hstack(
                        (A_adj_1-A_adj_2,np.zeros([nxny,eLen]),
                        -1*(np.zeros([nxny,h1Len+h2Len])+np.hstack((np.eye(h1Len),np.zeros([h2Len,h2Len])))),
                        np.zeros([nxny,WLen+uLen])
                        )
                    )

    A3_adj=np.hstack(
                        (A_adj_2-A_adj_1,np.zeros([nxny,eLen]),
                        -1*(np.zeros([nxny,h1Len+h2Len])+np.hstack((np.eye(h1Len),np.zeros([h2Len,h2Len])))),
                        np.zeros([nxny,WLen+uLen])
                        )
                    )


    #Contraint 3

    index_1_x=np.tile(np.arange(nxny),(n_y,1))
    index_1_x=index_1_x.flatten(order='F')

    index_1_y=np.tile(np.arange(1,(n_x+1)/n_x*nxny,step=n_x+1),(n_x,1))-1
    index_1_y2=np.tile(np.arange(n_x),(np.size(index_1_y,1),1)).T
    index_1_y=index_1_y+index_1_y2
    index_1_y=np.tile(index_1_y.T,(1,n_y))
    index_1_y=index_1_y.flatten(order='F')
    index_1_y=index_1_y.astype(int)
    mask=np.zeros([nxny,(n_x+1)*n_y])
    mask[index_1_x,index_1_y]=1


    A_adj3=np.repeat(np.repeat(Y_adj,n_x,axis=1),n_x,axis=0)

    ind=np.tile(np.arange(1,(n_x+1)/n_x*nxny,step=n_x+1),(n_x,1))-1
    ind2=np.tile(np.arange(n_x),(np.size(ind,1),1)).T
    ind=ind+ind2
    ind=ind.flatten(order='F')
    ind=ind.astype(int)

    new_adj=np.zeros([nxny,(n_x+1)*n_y])

    for i in range(nxny):
        new_adj[:,ind[i]]=A_adj3[:,i]

    A_adj_3=np.hstack((np.multiply(new_adj,mask),np.zeros([nxny,n_x+1])))



    index_2_x=np.tile(np.arange(nxny),(n_x,1))
    index_2_x=index_2_x.flatten(order='F')
    index_2_y=np.tile(np.arange((n_x+1)*n_y,step=n_x+1),(n_x,1))
    index_2_y2=np.tile(np.arange(n_x),(n_y,1)).T
    index_2_y=index_2_y+index_2_y2
    index_2_y=np.tile(index_2_y.T,(1,n_x)).T
    index_2_y=index_2_y.flatten(order='F')
    index_2_y=index_2_y.astype(int)


    mask=np.zeros([nxny,(n_x+1)*n_y])
    mask[index_2_x,index_2_y]=1

    A_adj4=np.tile(np.hstack((X_adj.T,np.zeros([n_x,1]))),(n_y,n_y))
    new_adj=A_adj4

    A_adj_4=np.hstack((np.multiply(new_adj,mask),np.zeros([nxny,n_x+1])))


    A4_adj=np.hstack(
                        (A_adj_3-A_adj_4,np.zeros([nxny,eLen]),
                        -1*(np.zeros([nxny,h1Len+h2Len])+np.hstack((np.zeros([h1Len,h1Len]),np.eye(h2Len)))),
                        np.zeros([nxny,WLen+uLen])
                        )
                    )

    A5_adj=np.hstack(
                        (A_adj_4-A_adj_3,np.zeros([nxny,eLen]),
                        -1*(np.zeros([nxny,h1Len+h2Len])+np.hstack((np.zeros([h1Len,h1Len]),np.eye(h2Len)))),
                        np.zeros([nxny,WLen+uLen])
                        )
                    )

    A_x=np.vstack((A1,A2_adj,A3_adj,A4_adj,A5_adj))


    Aw1=np.hstack((-np.diag(np.array(Q_plus).squeeze()),np.zeros([WLen,eLen+h1Len+h2Len]),np.eye(WLen),np.zeros([WLen,uLen])))# wi-Qi_plus*xi <=0, i=1,...n

    Aw2=np.hstack((np.diag(np.array(Q_minus).squeeze()),np.zeros([WLen,eLen+h1Len+h2Len]),-np.eye(WLen),np.zeros([WLen,uLen])))# -wi+Qi_minus*xi <=0, i=1,...n

    Q_=np.hstack((np.array(Q.todense()),np.zeros([WLen,eLen+h1Len+h2Len+WLen+uLen]))) # extend Q to nParam

    Aw3=-Q_+np.hstack((-np.diag(np.array(Q_minus).squeeze()),np.zeros([WLen,eLen+h1Len+h2Len]),np.eye(WLen),np.zeros([WLen,uLen])))  # wi-(Q*X)[i]-Qi_minus*xi <=-Qi_minus
    Aw4=Q_+np.hstack((np.diag(np.array(Q_plus).squeeze()),np.zeros([WLen,eLen+h1Len+h2Len]),-np.eye(WLen),np.zeros([WLen,uLen]))) # (Q*X)[i] + Qi_plus*xi- wi <=Qi_plus 

    A=np.vstack((A_x,Aw1,Aw2,Aw3,Aw4))
    lenb,_=A.shape
    b=np.vstack((np.zeros([lenb-2*WLen,1]),-Q_minus.transpose(),Q_plus.transpose()))
    bounds=[(0,None) for i in range(nParam)]
    f=f.flatten()
    res=linprog(f,A_ub=A,b_ub=b,A_eq=Aeq,b_eq=beq,method='highs',integrality=flag)# 0 continous, 1 integer.
    
    
    dxy=res.fun
    W=res.x
    if  res.success==False:
        print('Optimization failed')
    Wx=np.reshape(W[0:nxny2],(n_x+1,n_y+1),order='F')
    loc_cost=np.sum(np.multiply(DAB[0:n_x,0:n_y],Wx[0:n_x,0:n_y]))
    false_cost=np.sum(np.multiply(DAB[n_x,0:n_y],Wx[n_x,0:n_y]))
    miss_cost=np.sum(np.multiply(DAB[0:n_x,n_y],Wx[0:n_x,n_y]))
    
    WXW=Wx[:n_x,n_y].transpose()@X_adj@Wx[:n_x,n_y]
    WYW=Wx[n_x,:n_y]@Y_adj@Wx[n_x,:n_y].transpose()
    unassigned_edge_cost=beta/2*epsilon**p * (WXW+WYW)
    WX1=Wx[:n_x,n_y].transpose()@X_adj@np.ones(n_x)
    _1YW=np.ones(n_y)@Y_adj@Wx[n_x,:n_y].transpose()
    assigned_edge_cost=epsilon**p/4 * (W[e1Pos].item()+W[e2Pos].item())-0.5*epsilon**p * (WX1-WXW+_1YW-WYW)
    half_assigned_edge_cost=eta*epsilon**p * (WX1-WXW+_1YW-WYW)

    # For numerical stability
    dxy=np.maximum(dxy,0)
    loc_cost=np.maximum(loc_cost,0)
    miss_cost=np.maximum(miss_cost,0)
    false_cost=np.maximum(false_cost,0)
    assigned_edge_cost=np.maximum(assigned_edge_cost,0)
    unassigned_edge_cost=np.maximum(unassigned_edge_cost,0)
    half_assigned_edge_cost=np.maximum(half_assigned_edge_cost,0)

    return dxy**(1/p),loc_cost**(1/p),miss_cost**(1/p),false_cost**(1/p),\
        assigned_edge_cost**(1/p),unassigned_edge_cost**(1/p),half_assigned_edge_cost**(1/p)
