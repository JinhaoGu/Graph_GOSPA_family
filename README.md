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
beta=0.3 # penalty for unassigned edges
eta=0.7 # penalty for half-assigned edges (0 <= beta <= eta <= 1)

dxy,loc_cost,miss_cost,false_cost,assigned_edge_cost,unassigned_edge_cost,half_assigned_edge_cost=\
    graph_gospa_metric_family(X_attr,Y_attr,X_adj,Y_adj,c,p,epsilon,beta,eta)
```

### Hyperparameters

`beta` and `eta` must satisfy `0 <= beta <= eta <= 1`; the function raises a
`ValueError` otherwise. The values above are the ones used in [1]. Setting
`beta=0` and `eta=0.5` reduces the metric family to the graph GOSPA metric
of [2].

Each returned value is the p-th root of the corresponding cost. The costs add
up, so the p-th powers of the components add up to the p-th power of the total.
The returned values therefore add up to the total only for `p=1`.

### Integer and relaxed assignments

The optional `flag` argument controls the integrality of the linear program: `flag=0` (default) solves the continuous relaxation, while `flag=1` solves the binary/integer assignment problem.

```python
dxy,loc_cost,miss_cost,false_cost,assigned_edge_cost,unassigned_edge_cost,half_assigned_edge_cost=\
    graph_gospa_metric_family(X_attr,Y_attr,X_adj,Y_adj,c,p,epsilon,beta,eta,flag=1)
```

`flag=0` solves a relaxation, so it is faster but returns a lower bound of the
metric and can produce a fractional assignment. The cost components are exact
only for an integer assignment: with a fractional one `assigned_edge_cost` can
be negative, and the function issues a `RuntimeWarning` when that can happen.
Use `flag=1` when you need an exact decomposition.

## Tests

```
cd python_implementation
python test_graphGOSPAfamily.py
```
