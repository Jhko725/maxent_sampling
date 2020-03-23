import abc, random
import numpy as np
from .GraphIO import SimpleGraph
from .Utility import RecursiveLinReg, StopWatch, PrintProgress
from statistics import mean
from collections import deque

class Sampler(abc.ABC):
    """
    A abstract base class to represent a common interface for the sampling algorithms

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    sample : an instance of the class SimpleGraph
        The sampled graph.    
    N_sample : int
        Number of nodes in the sampled graph.
    
    Methods
    -------
    _RandomNode(seed = None)
        Returns a random node from the graph. Used to choose the initial node for sampling.
    sample()
        Samples nodes and edges from the graph according to some algorithm until N_sample nodes have been sampled. 
    __call__()
        Calls the self.Sample() to sample from the given graph, and returns the sampled graph as a SimpleGraph instance.
    """

    def __init__(self, graph, sample_ratio):
        """
        Parameters
        ----------
        graph : an instance of the class SimpleGraph
            The graph to sample from.
        sample_ratio : float
            The ratio of the size of the sampled graph to the original graph. The number of nodes of the 
            sampled graph N_sample is given by int(sampling_ratio*N_graph).
        """
        self.graph = graph
        self.N_sample = sample_ratio*len(graph)
        self.sample = SimpleGraph(graph.type_)

    def _RandomNode(self, seed = None):
        """
        Returns a random node from the graph. Used to choose the initial node for sampling.

        Parameters
        ----------
        seed : int
            Seed for the random number generator.
        """
        random.seed(seed)
        return random.choice(list(self.graph.graph.keys()))

    @abc.abstractmethod
    def Sample(self):
        pass

    def __call__(self):
        self.Sample()
        return self.sample


class RandomWalk(Sampler):
    """
    An abstract subclass of the abstract base class Sampler for random walk graph sampling.

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    N_sample : int
        Number of nodes in the sampled graph.
    N_relax : int
        Number of steps for the relaxation stage prior to sampling. The first N_relax nodes traveled by the random walker is not sampled, as the random walker has not equilibrated at that stage.
    relax : boolean
        True if the random walker is in relaxation stage. False otherwise.
    trajectory : deque
        A deque of the nodes traversed by the random walker. 
    seed : int
        Seed for the random number generator for sample initialization.
    
    Methods
    -------
    _Jump()
        Default method for moving the random walker at a given timestep. Jumps to one of the candidate nodes based on transition probability given by _TransitionProb()
    _TransitionProb(candidates)
        An abstract method to be overwritten by the child classes. Outputs transition probability of the random walk for the given candidate nodes
    Sample()
        Performs random walk sampling. 
    """
    def __init__(self, graph, sample_ratio, N_relax, N_history = 50000, seed = None):
        super().__init__(graph, sample_ratio)
        self.N_relax = N_relax
        self.relax = True
        self.n_current = self._RandomNode(seed)
        self.trajectory = deque([self.n_current], maxlen = N_history)
        

    def _Jump(self):
        candidates = self.graph(self.n_current)

        if not candidates: # If there are no nodes to jump to, go to a random node
            self.n_current = self._RandomNode()
            self.trajectory.append(self.n_current)
            if not self.relax:
                if len(self.sample) < self.N_sample:
                    self.sample.AddNode(self.n_current)
            candidates = self.graph(self.n_current)

        candidates = list(candidates)
        jump_prob = self._TransitionProb(candidates)
        n_next = random.choices(candidates, jump_prob)[0]
        
        self.trajectory.append(n_next)
        if not self.relax:
            if len(self.sample) < self.N_sample:
                self.sample.AddEdge(self.n_current, n_next)
        self.n_current = n_next    
    
    @abc.abstractmethod
    def _TransitionProb(self, candidates):
        pass

    def Sample(self):
        
        # Relaxation stage
        for _ in range(self.N_relax):
            self._Jump()
        self.relax = False
        print('Relaxed and ready to sample!')
        
        # Sampling stage
        stopwatch = StopWatch(0.5)
        while len(self.sample) < self.N_sample:
            self._Jump()
            PrintProgress(stopwatch(), len(self.sample)/self.N_sample)
           
        PrintProgress(True, 1)
        print('\n')
        print(self.sample)

class SimpleRW(RandomWalk):
    """
    An concrete subclass of the abstract superclass RandomWalk for simple random walk graph sampling. In simple random walk, the probability of transitioning to a neighboring node is equal for all neighbors.

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    N_sample : int
        Number of nodes in the sampled graph.
    N_relax : int
        Number of steps for the relaxation stage prior to sampling. The first N_relax nodes traveled by the random walker is not sampled, as the random walker has not equilibrated at that stage.
    p_restart : float
        Probability of returning to the starting node. p_restart must be in the range (0, 1)
    trajectory : list
        A list of the nodes traversed by the random walker
    seed : int
        Seed for the random number generator for sample initialization.

    Methods
    -------
    _TransitionProb(candidates)
        Returns the transition probability for each of the candidate nodes. In simple random walk, the transition probability is the same for all candidate notes.
    """
    def __init__(self, graph, sample_ratio, N_relax, p_restart, seed = None):
        super().__init__(graph, sample_ratio, N_relax, seed = None)
        self.p_restart = p_restart

    def _TransitionProb(self, candidates):
        prob = [1/len(candidates) for c in candidates]
        return prob

    def _Restart(self):
        coin = random.random()
        if coin < self.p_restart:
            self.n_current = self.trajectory[0]
            self.trajectory.append(self.n_current)
            if not self.relax:
                if len(self.sample) < self.N_sample:
                    self.sample.AddNode(self.n_current)
            return True
        return False
    
    def Sample(self):
        # Relaxation stage
        for _ in range(self.N_relax):
            restart = self._Restart()
            if not restart:
                self._Jump()
        self.relax = False
        print('Relaxed and ready to sample!')
        
        # Sampling stage
        stopwatch = StopWatch(0.5)
        while len(self.sample) < self.N_sample:
            restart = self._Restart()
            if not restart:
                self._Jump()

            PrintProgress(stopwatch(), len(self.sample)/self.N_sample)

        PrintProgress(True, 1)
        print('\n')
        print(self.sample)


class UniformNodeRW(RandomWalk):
    """
    An concrete subclass of the abstract superclass RandomWalk for uniform node random walk graph sampling. This random walk version of uniform node sampling is realized through Metropolis-Hastins sampling. The transition probability p_ij from node i to j is given by min(1/d_i, 1/d_j), where d_i denotes the degree of node i. 
    
    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    N_sample : int
        Number of nodes in the sampled graph.
    type_ : "dir" or "undir"
            "dir" for directed graph or "undir" for undirected graphs.
    sampled_nodes : set
        The set of sampled nodes. Nodes are represented by integer node Ids.
    sampled_edges : set
        The set of sampled edges. If type_ == "dir", then edges are tuple of the form (FromNode, ToNode), and if type_ == "undir", the edges are frozensets of the form {FromNode, ToNode}.
    N_relax : int
        Number of steps for the relaxation stage prior to sampling. The first N_relax nodes traveled by the random walker is not sampled, as the random walker has not equilibrated at that stage.
    trajectory : list
        A list of the nodes traversed by the random walker
    seed : int
        Seed for the random number generator for sample initialization.

    Methods
    -------
    _TransitionProb(candidates)
        Returns the transition probability for each of the candidate nodes. In uniform node random walk graph sampling, the transition probability for transition i -> j is given by p_ij = min{d_i, d_j} for i != j and (1- sum_j p_ij) for self transition i -> i.
    """

    def __init__(self, graph, sample_ratio, N_relax, seed = None):
        super().__init__(graph, sample_ratio, N_relax, seed = None)

    def _TransitionProb(self, candidates):
        d_i = len(self.graph(self.n_current)) # current node degree
        prob = [min(1/d_i, 1/len(self.graph(c))) for c in candidates]
        
        # Add self transition conditions
        candidates.append(self.n_current)
        total = sum(prob)
        prob.append(0) if total >= 1 else prob.append(1-total)
        
        return prob

class DegreePropRW(RandomWalk):
    """
    An concrete subclass of the abstract superclass RandomWalk for degree proportional random walk graph sampling. This is equivalent to maximal entropy random walk of order 1. The transition probability to a node is proportional to the degree of the node.

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    N_sample : int
        Number of nodes in the sampled graph.
    type_ : "dir" or "undir"
            "dir" for directed graph or "undir" for undirected graphs.
    sampled_nodes : set
        The set of sampled nodes. Nodes are represented by integer node Ids.
    sampled_edges : set
        The set of sampled edges. If type_ == "dir", then edges are tuple of the form (FromNode, ToNode), and if type_ == "undir", the edges are frozensets of the form {FromNode, ToNode}.
    N_relax : int
        Number of steps for the relaxation stage prior to sampling. The first N_relax nodes traveled by the random walker is not sampled, as the random walker has not equilibrated at that stage.
    trajectory : list
        A list of the nodes traversed by the random walker
    seed : int
        Seed for the random number generator for sample initialization.

    Methods
    -------
    _TransitionProb(candidates)
        Returns the transition probability for each of the candidate nodes. In uniform node random walk graph sampling, the transition probability for transition i -> j is given by p_ij = min{d_i, d_j} for i != j and (1- sum_j p_ij) for self transition i -> i.
    """ 

    def __init__(self, graph, sample_ratio, N_relax, seed = None):
        super().__init__(graph, sample_ratio, N_relax, seed = None)

    def _TransitionProb(self, candidates):
        temp = [len(self.graph(c)) for c in candidates]
        prob = [t/sum(temp) for t in temp]
        return prob       

class MaxEntropyRW(RandomWalk):
    """
    An concrete subclass of the abstract superclass RandomWalk for maximum entropy random walk. The transition probability to a node j is proportional d_j^(1-nu), where nu is the degree correlation exponent.

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    N_sample : int
        Number of nodes in the sampled graph.
    type_ : "dir" or "undir"
            "dir" for directed graph or "undir" for undirected graphs.
    sampled_nodes : set
        The set of sampled nodes. Nodes are represented by integer node Ids.
    sampled_edges : set
        The set of sampled edges. If type_ == "dir", then edges are tuple of the form (FromNode, ToNode), and if type_ == "undir", the edges are frozensets of the form {FromNode, ToNode}.
    N_relax : int
        Number of steps for the relaxation stage prior to sampling. The first N_relax nodes traveled by the random walker is not sampled, as the random walker has not equilibrated at that stage.
    trajectory : list
        A list of the nodes traversed by the random walker
    seed : int
        Seed for the random number generator for sample initialization.

    Methods
    -------
    _TransitionProb(candidates)
        Returns the transition probability for each of the candidate nodes. In uniform node random walk graph sampling, the transition probability for transition i -> j is given by p_ij = min{d_i, d_j} for i != j and (1- sum_j p_ij) for self transition i -> i.
    """ 

    def __init__(self, graph, sample_ratio, N_relax, adaptive = False, seed = None):
        super().__init__(graph, sample_ratio, N_relax, seed = None)
        self.nu = 0
        self.nu_history = []
        self.log_d_i = []
        self.log_d_neigh = []
        self.adaptive = adaptive

    def _TransitionProb(self, candidates):
        temp = [len(self.graph(c)) for c in candidates] # list of the degrees of the neighbors
        
        # update log_d_i and log_d_neigh to use for the estimation of nu
        self.log_d_i.append(np.log(len(self.graph(self.n_current))))
        self.log_d_neigh.append(np.log(mean(temp)))

        prob = [t**(1-self.nu)/sum(temp) for t in temp]
        return prob       

    def Sample(self):
        # Initialize recursive linear regressor
        RecLinReg = RecursiveLinReg()

        # Relaxation stage
        for _ in range(self.N_relax):
            self._Jump()
        print('Relaxed and ready to sample!')
        self.relax = False

        # Calculate the correlation exponent
        _, self.nu = RecLinReg(self.log_d_i, self.log_d_neigh)
        self.nu *= -1
        self.nu_history.append(self.nu)
        print('Correlation exponent estimated: nu = {}'.format(self.nu))
        
        # Sampling stage
        stopwatch = StopWatch(0.5)
        while len(self.sample) < self.N_sample:
            self._Jump()
            if self.adaptive:
                _, self.nu = RecLinReg(self.log_d_i[-1:], self.log_d_neigh[-1:])
                self.nu *= -1
                self.nu_history.append(self.nu)
            
            PrintProgress(stopwatch(), len(self.sample)/self.N_sample)

        PrintProgress(True, 1)
        print('\n')
        print(self.sample)


class ForestFire(Sampler):
    """
    A concrete subclass of the abstract base class Sampler for forest fire sampling

    ...

    Attributes
    ----------
    graph : an instance of the class SimpleGraph
        The graph to sample from. 
    sample : an instance of the class SimpleGraph
        The sampled graph.
    N_sample : int
        Number of nodes in the sampled graph.
    type_ : "dir" or "undir"
            "dir" for directed graph or "undir" for undirected graphs.
    p_burn : float
        Probability of "burning" an edge. p_burn must be in the range (0, 1)

    Methods
    -------
    _Burn(node)
        For a given node, return a list of nodes to be burnt by the fire. 
    _SpreadFire()
        Update self.firefront, which are the nodes most recently burnt by the fire.
    Sample()
        Sample N_sample nodes from the graph.
    """
    
    def __init__(self, graph, sample_ratio, p_burn = 0.7, seed = None):
        super().__init__(graph, sample_ratio)
        self.p_burn = p_burn
        n0 = self._RandomNode(seed)
        self.firefront = {n0}
        self.sample.AddNode(n0)

    def _Burn(self, node):
        """
        For a given node, return a list of nodes to be burnt by the fire. Also add burnt edges and nodes to self.sample.

        Parameters
        ----------
        node : int
            ID of the node currently caught on fire
        """
        N_burn = np.random.geometric(self.p_burn, 1)[0]
        candidates = self.graph(node) - self.sample(node)
        
        if len(candidates) < N_burn:
            sampled_nodes = candidates
        else:
            sampled_nodes = set(random.sample(candidates, N_burn))
            
        for ToNode in sampled_nodes:
            if len(self.sample) < self.N_sample:
                self.sample.AddEdge(node, ToNode)
        return sampled_nodes

    def _SpreadFire(self):
        """
        Update self.firefront, which are the nodes most recently burnt by the fire.
        """
        firefront_new = set()
        for node in self.firefront:
            firefront_new.union(self._Burn(node))
        self.firefront = firefront_new

    def Sample(self):
        stopwatch = StopWatch(0.5)
        while len(self.sample) < self.N_sample:
            if not self.firefront:
                n0_new = self._RandomNode()
                self.sample.AddNode(n0_new)
                self.firefront = {n0_new} # if firefront is empty, move to a new random node
            self._SpreadFire()
            
            PrintProgress(stopwatch(), len(self.sample)/self.N_sample)

        PrintProgress(True, 1)
        print('\n')
        print(self.sample)
