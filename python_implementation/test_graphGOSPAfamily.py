'''Regression tests for the graph GOSPA metric family.

Run with `python test_graphGOSPAfamily.py`.  The script exits with a non-zero
status if a test fails.  It needs numpy and scipy only.
'''
import sys
import warnings

import numpy as np

from graphGOSPAfamily import graph_gospa_metric_family

# The parameters used in the paper.
BETA=0.3
ETA=0.7

_failures=[]


def check(condition,message):
    if not condition:
        _failures.append(message)
    print(('  PASS  ' if condition else '  FAIL  ')+message)


def random_graph(rng,n,dim=2,density=0.5):
    '''Return the attributes and the adjacency matrix of an undirected graph.'''
    adj=(rng.random((n,n))<density).astype(float)
    adj=np.triu(adj,1)
    adj=adj+adj.T
    return rng.random((n,dim))*10,adj


def raw(value,p):
    '''Undo the signed p-th root that the metric applies to its output.'''
    return np.sign(value)*np.abs(value)**p


# p, epsilon, beta, eta.  Every pair satisfies 0<=beta<=eta<=1.
GRID=[(1,1,0.3,0.7),(2,3,0.3,0.7),(1,2,0.3,0.7),(2,1,1.0,1.0),
      (1,2,0.0,0.5),(3,2,0.5,0.9),(1,2,0.0,0.0),(2,2,0.9,1.0)]


def test_components_add_up_in_the_pth_power():
    '''The p-th powers of the components add up to the p-th power of the total.

    The function returns the p-th root of every cost, so the returned values
    themselves add up only for p=1.  test_p_one_adds_up_directly covers that
    case separately.
    '''
    print('the p-th powers of the components add up (only p=1 adds up directly)')
    for p,epsilon,beta,eta in GRID:
        for flag in (0,1):
            rng=np.random.default_rng(7)
            worst=0.0
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                for _ in range(120):
                    n_x=int(rng.integers(1,6))
                    n_y=int(rng.integers(1,6))
                    X_attr,X_adj=random_graph(rng,n_x)
                    Y_attr,Y_adj=random_graph(rng,n_y)
                    out=np.array(graph_gospa_metric_family(
                        X_attr,Y_attr,X_adj,Y_adj,3,p,epsilon,beta,eta,flag=flag))
                    worst=max(worst,abs(raw(out[0],p)-raw(out[1:],p).sum()))
            check(worst<1e-8,'p=%d epsilon=%g beta=%g eta=%g flag=%d: max error %.3e'
                  %(p,epsilon,beta,eta,flag,worst))


def test_p_one_adds_up_directly():
    '''For p=1 the returned values add up to the returned total.'''
    print('for p=1 the returned values add up directly')
    rng=np.random.default_rng(11)
    for flag in (0,1):
        worst=0.0
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            for _ in range(150):
                n_x=int(rng.integers(1,6))
                n_y=int(rng.integers(1,6))
                X_attr,X_adj=random_graph(rng,n_x)
                Y_attr,Y_adj=random_graph(rng,n_y)
                for epsilon in (1,2,3):
                    out=np.array(graph_gospa_metric_family(
                        X_attr,Y_attr,X_adj,Y_adj,3,1,epsilon,BETA,ETA,flag=flag))
                    worst=max(worst,abs(out[0]-out[1:].sum()))
        check(worst<1e-8,'p=1 flag=%d: max error %.3e'%(flag,worst))


def test_no_nan():
    '''No component is ever nan.  A negative cost used to give nan for p>1.'''
    print('no component is nan')
    rng=np.random.default_rng(7)
    count=0
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(300):
            n_x=int(rng.integers(1,6))
            n_y=int(rng.integers(1,6))
            X_attr,X_adj=random_graph(rng,n_x)
            Y_attr,Y_adj=random_graph(rng,n_y)
            for p,epsilon in [(2,3),(2,1),(3,2)]:
                out=graph_gospa_metric_family(
                    X_attr,Y_attr,X_adj,Y_adj,3,p,epsilon,BETA,ETA)
                count+=np.isnan(np.array(out)).any()
    check(count==0,'nan results in 900 solves: %d'%count)


def test_nan_regression():
    '''This input returned assigned_edge_cost=nan before the fix.'''
    print('regression: the input that used to return nan')
    X_attr=np.array([[0.3,9.945],[6.049,0.739],[4.97,0.495],[5.489,1.855]])
    X_adj=np.array([[0.,0.,1.,0.],[0.,0.,0.,1.],[1.,0.,0.,1.],[0.,1.,1.,0.]])
    Y_attr=np.array([[9.365,3.289],[5.697,0.059]])
    Y_adj=np.array([[0.,1.],[1.,0.]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        out=np.array(graph_gospa_metric_family(
            X_attr,Y_attr,X_adj,Y_adj,3,2,3,BETA,ETA))
    check(not np.isnan(out).any(),'no nan: %s'%np.array2string(out,precision=4))
    check(abs(raw(out[0],2)-raw(out[1:],2).sum())<1e-8,'the components add up')


def test_negative_component_warns():
    '''The relaxation can make assigned_edge_cost negative.  It must warn.'''
    print('a negative component is only assigned_edge_cost, and it warns')
    rng=np.random.default_rng(7)
    wrong_slot=0
    silent=0
    negative=0
    for _ in range(300):
        n_x=int(rng.integers(1,6))
        n_y=int(rng.integers(1,6))
        X_attr,X_adj=random_graph(rng,n_x)
        Y_attr,Y_adj=random_graph(rng,n_y)
        for p,epsilon in [(1,2),(1,3),(2,3)]:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                out=np.array(graph_gospa_metric_family(
                    X_attr,Y_attr,X_adj,Y_adj,3,p,epsilon,BETA,ETA))
            index=np.where(out<-1e-9)[0]
            if len(index):
                negative+=1
                if set(index)!={4}:
                    wrong_slot+=1
                if not caught:
                    silent+=1
    check(wrong_slot==0,'negatives outside assigned_edge_cost: %d of %d'
          %(wrong_slot,negative))
    check(silent==0,'negative results without a warning: %d'%silent)


def test_integer_solution_is_exact():
    '''flag=1 gives non-negative components, no warning and an exact total.'''
    print('flag=1 is exact and non-negative')
    rng=np.random.default_rng(7)
    bad=0
    worst=0.0
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        for _ in range(200):
            n_x=int(rng.integers(1,6))
            n_y=int(rng.integers(1,6))
            X_attr,X_adj=random_graph(rng,n_x)
            Y_attr,Y_adj=random_graph(rng,n_y)
            for p,epsilon,beta,eta in GRID:
                out=np.array(graph_gospa_metric_family(
                    X_attr,Y_attr,X_adj,Y_adj,3,p,epsilon,beta,eta,flag=1))
                bad+=(out<-1e-9).any() or np.isnan(out).any()
                worst=max(worst,abs(raw(out[0],p)-raw(out[1:],p).sum()))
    check(bad==0 and worst<1e-8,
          'bad components %d, max total error %.3e'%(bad,worst))


def test_weighted_adjacency():
    '''flag=1 must constrain the assignment variables only.

    The auxiliary variables are continuous.  Rounding them gave 1.0 instead of
    0.05 for two graphs whose only difference is an edge weight of 0.05.
    '''
    print('flag=1 with a weighted adjacency matrix')
    attr=np.array([[0.,0.],[1.,0.],[0.,1.]])
    X_adj=np.array([[0.,0.30,0.7],[0.30,0.,0.2],[0.7,0.2,0.]])
    Y_adj=np.array([[0.,0.35,0.7],[0.35,0.,0.2],[0.7,0.2,0.]])
    value=graph_gospa_metric_family(attr,attr,X_adj,Y_adj,3,1,1,0.0,0.5,flag=1)[0]
    check(abs(value-0.05)<1e-9,'one edge weight differs by 0.05, got %.6f'%value)


def test_eta_below_beta_raises():
    '''The metric family needs 0<=beta<=eta<=1.'''
    print('beta and eta outside 0<=beta<=eta<=1 raise ValueError')
    attr=np.array([[0.,0.],[1.,0.]])
    adj=np.array([[0.,1.],[1.,0.]])
    for beta,eta in [(1.0,0.5),(0.5,1.5),(1.5,1.5),(-0.1,0.5),(0.3,-0.1)]:
        try:
            graph_gospa_metric_family(attr,attr,adj,adj,3,1,1,beta,eta)
            check(False,'beta=%r, eta=%r did not raise'%(beta,eta))
        except ValueError:
            check(True,'beta=%r, eta=%r raises ValueError'%(beta,eta))
    for beta,eta in [(0.0,0.0),(0.3,0.7),(1.0,1.0),(0.0,1.0)]:
        try:
            graph_gospa_metric_family(attr,attr,adj,adj,3,1,1,beta,eta)
            check(True,'beta=%r, eta=%r accepted'%(beta,eta))
        except ValueError:
            check(False,'beta=%r, eta=%r was rejected but is valid'%(beta,eta))


def test_empty_graphs():
    '''An empty graph works on either side and the metric stays symmetric.'''
    print('empty graphs')
    empty_attr=np.zeros((0,2))
    empty_adj=np.zeros((0,0))
    attr=np.array([[0.,0.],[10.,10.]])
    adj=np.array([[0.,1.],[1.,0.]])
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        forward=graph_gospa_metric_family(
            attr,empty_attr,adj,empty_adj,3,1,1,BETA,ETA)[0]
        backward=graph_gospa_metric_family(
            empty_attr,attr,empty_adj,adj,3,1,1,BETA,ETA)[0]
        both=graph_gospa_metric_family(
            empty_attr,empty_attr,empty_adj,empty_adj,3,1,1,BETA,ETA)[0]
    check(abs(forward-backward)<1e-9,
          'one empty graph: %.4f either way'%forward)
    check(both==0.0,'two empty graphs give 0')


def test_metric_properties():
    '''The metric is symmetric and satisfies the triangle inequality.'''
    print('symmetry and the triangle inequality (flag=1)')
    rng=np.random.default_rng(5)
    asymmetric=0
    triangle=0

    def distance(first,second):
        return graph_gospa_metric_family(
            first[0],second[0],first[1],second[1],3,1,1,BETA,ETA,flag=1)[0]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for _ in range(50):
            a=random_graph(rng,int(rng.integers(1,4)))
            b=random_graph(rng,int(rng.integers(1,4)))
            c=random_graph(rng,int(rng.integers(1,4)))
            asymmetric+=abs(distance(a,b)-distance(b,a))>1e-7
            triangle+=distance(a,c)>distance(a,b)+distance(b,c)+1e-7
    check(asymmetric==0 and triangle==0,
          'symmetry violations %d, triangle violations %d'%(asymmetric,triangle))


def test_identical_graphs():
    '''The metric between a graph and itself is zero.'''
    print('a graph has distance zero to itself')
    X_attr=np.array([[0.,0.],[10.,10.],[10.,20.]])
    X_adj=np.array([[0.,1.,1.],[1.,0.,1.],[1.,1.,0.]])
    for p in (1,2,3):
        for flag in (0,1):
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                out=np.array(graph_gospa_metric_family(
                    X_attr,X_attr,X_adj,X_adj,3,p,1,BETA,ETA,flag=flag))
            check(np.allclose(out,0),'p=%d flag=%d: %s'
                  %(p,flag,np.array2string(out,precision=6)))


def main():
    for test in [test_components_add_up_in_the_pth_power,
                 test_p_one_adds_up_directly,test_no_nan,test_nan_regression,
                 test_negative_component_warns,test_integer_solution_is_exact,
                 test_weighted_adjacency,test_eta_below_beta_raises,
                 test_empty_graphs,test_metric_properties,
                 test_identical_graphs]:
        test()
        print()
    print('='*60)
    if _failures:
        print('FAILED: %d'%len(_failures))
        for failure in _failures:
            print('   '+failure)
        return 1
    print('All tests passed.')
    return 0


if __name__=='__main__':
    sys.exit(main())
