import snap

class SimpleGraph():
    """
    A class for unified representation of unweighted graphs to be used in this package

    ...

    Attributes
    ----------
    graph : dict
        A dictionary representing the graph. Keys are the nodes, and values indicate 
        sets of nodes connected to the key
    type_ : "dir" or "undir"
        "dir" for directed graph or "undir" for undirected graphs.

    Methods
    -------
    __len__()
        Returns the number of nodes in the graph.

    __eq__()
        Compares two SimpleGraph instances to determine their equality. Returns true only if all nodes and edges are identical.

    NumNodes()
        Same as __len__(), returns the number of nodes in the graph.
    
    __call__()
        Returns the set of nodes connected to Node.        

    NumEdges()
        Returns the number of edges in the graph.

    __repr()__
        Returns the string representation of the graph class instance.

    AddNode(Node)
        Adds a node with ID Node.

    AddEdge(FromNode, ToNode)
        Adds an edge between FromNode and ToNode.

    LoadFromTable(filepath)
        Loads graph data from table saved in text format. The table must be of the form "FromNodeId" 
        and "ToNodeID" as the two columns.

    SaveToTable(savepath)
        Saves graph data to the table saved in text format. The table must be of the form "FromNodeId" 
        and "ToNodeID" as the two columns.
    """


    def __init__(self, type_ = "undir"):
        """
        Parameters
        ----------
        type_ : "dir" or "undir"
            "dir" for directed graph or "undir" for undirected graphs.

        Raises
        ------
        AssertionError
            If type_ is not "dir" or "undir"
        """
        self.graph = {}

        assert type_ in ["dir", "undir"], """Graph type must be either "dir" for directed graphs or "undir" for undirected graphs"""
        self.type_ = type_

    def __len__(self):
        """
        Returns the number of nodes in the graph
        """
        return len(self.graph.keys())

    NumNodes = __len__
    
    def __eq__(self, other):
        """
        Compares two SimpleGraph instances to determine their equality. Returns true only if all nodes and edges are identical.

        Parameters
        ----------
        other : an instance of class SimpleGraph.
        """
        if set(self.graph.keys()) == set(other.graph.keys()):
            for k in self.graph.keys():
                if self.graph[k] != other.graph[k]:
                    return False
            return True
        return False

    def __call__(self, Node):
        """
        Returns the set of nodes connected to Node.

        Parameters
        ----------
        Node : int
            Node ID for the node in question.
        """
        return self.graph[Node]

    def NumEdges(self):
        """
        Returns the number of edges in the graph
        """
        total_edges = sum([len(a) for a in self.graph.values()])
        
        if self.type_ == "undir":
            self_edges = sum([k in v for k, v in self.graph.items()])
            total_edges = (total_edges - self_edges)/2 + self_edges 

        return int(total_edges)

    def __repr__(self):
        """
        Returns the string representation of the class instance
        """
        N_nodes = self.NumNodes()
        N_edges = self.NumEdges()
        type_string = "directed" if self.type_ == "dir" else "undirected"
        
        return "An {} graph with {} nodes and {} edges".format(type_string, N_nodes, N_edges)   

    def AddNode(self, Node):
        """
        Adds a node with ID Node.

        Parameters
        ----------
        Node: int
            Node ID for the node to be added
        """
        assert isinstance(Node, int), "IDs of nodes must be int!"

        if Node not in self.graph.keys():
            self.graph[Node] = set()

    def AddEdge(self, FromNode, ToNode):
        """
        Adds an edge between FromNode and ToNode.  

        Parameters
        ----------
        FromNode : int
            Node ID for the starting node
        ToNode : int
            Node ID for the ending node
        """
        assert isinstance(FromNode, int) and isinstance(ToNode, int), "IDs of FromNode and ToNode must be int!"

        self.AddNode(FromNode)
        self.graph[FromNode].add(ToNode)

        if self.type_ == "undir":
            self.AddNode(ToNode)
            self.graph[ToNode].add(FromNode)

    def LoadFromTable(self, filepath):
        """
        Loads graph data from table saved in text format. The table must be of the form "FromNodeId" 
        and "ToNodeID" as the two columns.

        Parameters
        ----------
        filepath : str
            filepath to the text file for the table
        """
        with open(filepath, 'rt') as f:
            for line in f:
                temp = line.split()
                
                if len(temp) == 2:
                    FromNode = int(temp[0])
                    ToNode = int(temp[1])

                    self.AddEdge(FromNode, ToNode)
       
        print('Loading finished.')
        print(self)

    def SaveToTable(self, savepath):
        """
        Saves graph data to the table saved in text format. The table must be of the form "FromNodeId" 
        and "ToNodeID" as the two columns.

        Parameters
        ----------
        savepath : str
            path where the saved text file will be saved at
        """
        if self.type_ == "dir":
            edge_list = {(FromNode, ToNode) for FromNode, v in self.graph.items() for ToNode in v}
        else:
            edge_list = {frozenset((FromNode, ToNode)) for FromNode, v in self.graph.items() for ToNode in v}

        with open(savepath, 'w') as f:
            f.write('# Directed graph\n') if self.type_ == "dir" else f.write('# Undirected graph\n')
            f.write('# Nodes: {} Edges: {}\n'.format(self.NumNodes(), self.NumEdges()))
            f.write('# FromNodeId\tToNodeId\n')
            for item in edge_list:
                item = tuple(item)
                if len(item) == 1:
                    f.write('{}\t{}\n'.format(item[0], item[0]))
                else:
                    f.write('{}\t{}\n'.format(item[0], item[1]))
        print('Successfully saved to {}'.format(savepath))


    def ToSnap(self):
        """
        Converts the given graph to the graph format used by python's SNAP library

        ...
        
        Returns
        -------
        graph_snap : an instance of SNAP.TUNGraph()/SNAP.TNGraph() for undirected/directed graph
            The converted graph
        """
        
        if self.type_ == "undir":
            graph_snap = snap.TUNGraph.New()
        else:
            graph_snap = snap.TNGraph.New()

        for node in self.graph.keys():
            if not graph_snap.IsNode(node):
                graph_snap.AddNode(node)
            for node_end in self.graph[node]:
                if not graph_snap.IsNode(node_end):
                    graph_snap.AddNode(node_end)
                graph_snap.AddEdge(node, node_end)
        
        return graph_snap