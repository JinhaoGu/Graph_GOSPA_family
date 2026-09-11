# The graph GOSPA metric family

This repository contains Python code with the linear programming implementation of the family of graph generalised optimal subpattern assignment (GOSPA) metrics proposed in [1]. The graph GOSPA metric family is a mathematically principled family of metrics for graphs. Given two graphs, it penalises node attribute error for properly assigned nodes, the number of missed nodes and false nodes, and edge mismatches. In contrast to the graph GOSPA metric, the family provides more general penalties for edge mismatches through the hyperparameters `beta` and `eta`.

The graph GOSPA metric family contains the graph GOSPA metric of [2] as a special case, which itself is an extension of the GOSPA metric for sets of objects proposed in [3], also extended for sets of trajectories in [4].

[1] J. Gu, A. F. García-Fernández, Robert E. Firth, L. Svensson, “A family of graph GOSPA metrics for graphs with different sizes” (https://arxiv.org/abs/2506.17316)

[2] J. Gu, A. F. García-Fernández, Robert E. Firth, L. Svensson, “Graph GOSPA metric: a metric to measure the discrepancy between graphs of different sizes” in IEEE Transactions on Signal Processing, 2024 (https://arxiv.org/abs/2311.07596)

[3] A. S. Rahmathullah, Á. F. García-Fernández and L. Svensson, "Generalized optimal sub-pattern assignment metric," 2017 20th International Conference on Information Fusion (Fusion), Xi'an, China, 2017, pp. 1-8, doi: 10.23919/ICIF.2017.8009645.

[4] Á. F. García-Fernández, A. S. Rahmathullah and L. Svensson, "A Metric on the Space of Finite Sets of Trajectories for Evaluation of Multi-Target Tracking Algorithms," in IEEE Transactions on Signal Processing, vol. 68, pp. 3917-3928, 2020, doi: 10.1109/TSP.2020.3005309.

## Usage:
Below is a usage example of the graph GOSPA metric family.
### Python:
```python
import numpy as np
from graphGOSPAfamily import graph_gospa_metric_family

# define graph X and Y 
X_attr= np.array([[0,0],[10,10],[10,20]])
X_adj=np.array([[0,1,1],[1,0,1],[1,1,0]])

Y_attr= np.array([[0,0],[10,10],[10,20]])
Y_adj=np.array([[0,1,1],[1,0,1],[1,1,0]])

# choose parameters
c=3 # penalty for missing or false nodes 
p=1 # p-norm
epsilon=1 # penalty for edge mismatch
beta=1 # penalty for unassigned edges
eta=0.5 # penalty for half-assigned edges

dxy,loc_cost,miss_cost,false_cost,assigned_edge_cost,unassigned_edge_cost,half_assigned_edge_cost=\
    graph_gospa_metric_family(X_attr,Y_attr,X_adj,Y_adj,c,p,epsilon,beta,eta)
```

The optional `flag` argument controls the integrality of the linear program: `flag=0` (default) solves the continuous relaxation, while `flag=1` solves the binary/integer assignment problem.

```python
dxy,loc_cost,miss_cost,false_cost,assigned_edge_cost,unassigned_edge_cost,half_assigned_edge_cost=\
    graph_gospa_metric_family(X_attr,Y_attr,X_adj,Y_adj,c,p,epsilon,beta,eta,flag=1)
```
