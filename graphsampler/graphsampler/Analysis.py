import numpy as np
import snap
from .Utility import StopWatch, PrintProgress
from .GraphIO import SimpleGraph
import matplotlib.pyplot as plt

def NodeDegDist(graph, deg_type = "both"):
    """
    Calculates the node - degree distribution of a given graph. The graph must be in snap format.

    ...
    
    Parameters
    ----------
    graph : an instance of SNAP.TUNGraph()/SNAP.TNGraph() for undirected/directed graph
        The graph for which the degree distribution is to be calculated
    type : "both" / "in" / "out"
        Decides whether to calculate the total degree / outdegree / indegree distribution. Does not affect undirected graphs.

    Returns
    -------
    deg_dist : 2D numpy array with shape (2, :)
        The calculated degree distribution for graph.
        deg_dist[0, :] : degree value
        deg_dist[1, :] : number of occurances
    """
   
    N_deg = graph.GetNodes()
    deg_arr = np.zeros((1, N_deg))
    i = 0
    stopwatch = StopWatch(0.5)
    # If the graph is directed, assign proper degree type
    if type(graph).__name__ == 'PNGraph':
        if deg_type == "in":
            for node in graph.Nodes():
                PrintProgress(stopwatch(), i/N_deg)
                deg_arr[0, i] = node.GetInDeg()
                i+=1

        elif deg_type == "out":
            for node in graph.Nodes():
                PrintProgress(stopwatch(), i/N_deg)
                deg_arr[0, i] = node.GetInDeg()
                i+=1

        else:
            for node in graph.Nodes():
                PrintProgress(stopwatch(), i/N_deg)
                deg_arr[0, i] = node.GetDeg()
                i+=1    
    
    else:
        for node in graph.Nodes():
            PrintProgress(stopwatch(), i/N_deg)
            deg_arr[0, i] = node.GetDeg()
            i+=1    
    
    deg, occ = np.unique(deg_arr, return_counts = True)
    
    # Sort results
    sort_ind = np.argsort(deg)
    deg_dist = np.vstack([deg[sort_ind], occ[sort_ind]])

    PrintProgress(True, 1)
    print('\n')

    return deg_dist

def DegClustDist(graph):
    """
    Calculates the degree - clustering coefficient distribution of a given graph. The graph must be in snap format.

    ...
    
    Parameters
    ----------
    graph : an instance of SNAP.TUNGraph()/SNAP.TNGraph() for undirected/directed graph
        The graph for which the degree distribution is to be calculated
    
    Returns
    -------
    coeff_dist : 2D numpy array with shape (2, :)
        The calculated degree distribution for graph.
        coeff_dist[0, :] : degree value
        coeff_dist[1, :] : mean clustering coefficient of the nodes with a given degree
    """
    
    deg_coeff = {}
    stopwatch = StopWatch(0.5)
    N_deg = graph.GetNodes()

    j = 0
    for node in graph.Nodes():
        PrintProgress(stopwatch(), j/N_deg)
        deg = node.GetDeg()
        clust = snap.GetNodeClustCf(graph, node.GetId())

        if deg in deg_coeff.keys():
            deg_coeff[deg].append(clust) 
        else:
            deg_coeff[deg] = [clust]
        j+=1

    coeff_dist = np.zeros((2, len(deg_coeff.keys())))
     
    i = 0
    for k, v in deg_coeff.items():
        coeff_dist[0, i] = k
        coeff_dist[1, i] = np.mean(v)
        i+=1
    
    # Sort results
    sort_ind = np.argsort(coeff_dist[0, :])
    coeff_dist = coeff_dist[:, sort_ind]
    PrintProgress(True, 1)
    print('\n')

    return coeff_dist

def NodeMaxhopDist(graph):
    """
    Calculates the node - maximum hop distribution of a given graph. The graph must be in snap format. Maximum hop of a node is the minimum number of steps to reach all other reachable nodes in the graph.

    ...
    
    Parameters
    ----------
    graph : an instance of SNAP.TUNGraph()/SNAP.TNGraph() for undirected/directed graph
        The graph for which the degree distribution is to be calculated

    Returns
    -------
    maxhop_dist : 2D numpy array with shape (2, :)
        The calculated maximum hop distribution for graph.
        maxhop_dist[0, :] : maximum hop value
        maxhop_dist[1, :] : number of occurances
    """
    
    N_deg = graph.GetNodes()
    maxhop_arr = np.zeros((1, N_deg))
    stopwatch = StopWatch(0.5)

    # Initialize hash table to store intermediate values in
    NIdToDistH = snap.TIntH()
    i = 0
    for node in graph.Nodes():
        PrintProgress(stopwatch(), i/N_deg)
        maxhop_arr[0, i] = snap.GetShortPath(graph, node.GetId(), NIdToDistH)
        
        i+=1
        
    maxhop, occ = np.unique(maxhop_arr, return_counts = True)
    
    # Sort results
    sort_ind = np.argsort(maxhop)
    maxhop_dist = np.vstack([maxhop[sort_ind], occ[sort_ind]])

    PrintProgress(True, 1)
    print('\n')

    return maxhop_dist
    
def RankEigvecDist(graph):
    """
    Calculates the degree - clustering coefficient distribution of a given graph. The graph must be in snap format.

    ...
    
    Parameters
    ----------
    graph : an instance of SNAP.TUNGraph()/SNAP.TNGraph() for undirected/directed graph
        The graph for which the degree distribution is to be calculated
    
    Returns
    -------
    eigvec_dist : 2D numpy array with shape (2, :)
        The calculated eigenvector distribution for graph. Here, the eigenvector refers to the principal eigenvector, which should be positive definite due to Perrin-Frobenius theorem. 
        eigvec_dist[0, :] : rank of the eigenvector
        eigvec_dist[1, :] : value of the element of the first eigenvector, sorted in the order of descending value.
    """

    EigVec = snap.TFltV()
    snap.GetEigVec(graph,EigVec)
    
    eigvec = np.abs(np.array(EigVec))
    N = len(eigvec)
    eigvec.reshape((1, N))
    rank = np.arange(N).reshape((1, N))
    eigvec = eigvec[np.argsort(-eigvec)]
    eigvec_dist = np.vstack([rank, eigvec])

    return eigvec_dist

def TopKRankEigvalDist(graph, K):
    N = graph.GetNodes()
    assert K<=N, "K must be smaller than the number of nodes in the graph"
    
    flag = 0
    N_try = min(2*K, N)
    # The code is written in this convoluted way, because for some unknown reasons regarding the snap library, the function snap.GetEigVals does not properly return the same number of the top eigenvalues as requested
    while flag < K:
        EigVal =  snap.TFltV()
        snap.GetEigVals(graph, N_try, EigVal)
        eig = np.array(EigVal)
        flag = len(eig)
        N_try = min(2*N_try, N)
    
    eig = np.abs(eig)
    
    return np.vstack([np.arange(K).reshape((1, K)), eig[:K].reshape((1, K))])
    
def D_Statistic(dist1, dist2):
    """
    Calculates the Kolmogorov-Smirnov D statistic between the two distributions. The distributions need not be normalized. 

    ...

    Parameters
    ----------
    dist1 : 2D numpy array
        Input distribution. Assumed to be a probability distribution with dist[0,:] as the independent variable and dist[1,:] as the unnormalized probability associated with the independent variable.
    dist2 : 2D numpy array
        Input distribution. Assumed to be a probability distribution with dist[0,:] as the independent variable and dist[1,:] as the unnormalized probability associated with the independent variable.
    
    Returns
    -------
    D : The Kolmogorov-Smirnov D statistic between the two input distributions. 
    """

    x = np.union1d(dist1[0,:], dist2[0,:]) # union of the independent variable
    dist1_union = np.zeros(x.shape)
    dist2_union = np.zeros(x.shape)

    if dist1[1,:].sum() == 0 or dist2[1,:].sum() == 0:
        return 1

    dist1_union[np.isin(x, dist1[0,:])] = dist1[1,:]/dist1[1,:].sum()
    dist2_union[np.isin(x, dist2[0,:])] = dist2[1,:]/dist2[1,:].sum()

    dist1_cum = np.cumsum(dist1_union)
    dist2_cum = np.cumsum(dist2_union)

    return np.amax(np.abs(dist1_cum - dist2_cum))

def TopKEigCent(graph, K):
    eigscore = np.zeros((graph.GetNodes(), 2))
    
    NIdEigenH = snap.TIntFltH()
    snap.GetEigenVectorCentr(graph, NIdEigenH)
    
    i = 0
    for item in NIdEigenH:
        eigscore[i, 0] = int(item)
        eigscore[i, 1] = NIdEigenH[item]
        i+=1
    
    ind_sorted = np.argsort(-eigscore[:,1])
    eigscore = eigscore[ind_sorted, :]
    
    return eigscore[:K, :]

def PlotDist(savepath, distname, dist_dict, datasetname, labelmap, colormap):
    _, ax  = plt.subplots(1, 1, figsize = (8, 6))
    for methodname, data in dist_dict.items():
        ax.plot(data[0,:], data[1,:], label = methodname, color = colormap[methodname])
    ax.set_xlabel(labelmap[distname][0], fontsize = 15)
    ax.set_ylabel(labelmap[distname][1], fontsize = 15)
    ax.set_title("{} for the {} dataset".format(distname, datasetname))
    ax.legend()
    ax.grid(ls = '--')
    if distname not in ["Maxium Hop Distribution"]:
        ax.set_xscale('log')
    if distname in ["Degree Distribution", "Maxium Hop Distribution"]:
        ax.set_yscale("log")
    plt.savefig(savepath)
    plt.close()

def PlotDHist(savepath, distname, D_dict, colormap):
    fig, ax = plt.subplots(1, 1, figsize= (8, 6))
    # Set width of bar
    barwidth = 0.2

    # Create bar position
    barloc = {}
    barloc0 = np.arange(5)
    for methodname in D_dict.keys() : 
        barloc[methodname] = barloc0
        barloc0 = [loc + barwidth for loc in barloc0]

    # Get the datasetnames
    datasets = list(D_dict[list(D_dict.keys())[0]].keys())
    # Make the plot
    for methodname in D_dict.keys():
        data = []
        # This is done because we want a fixed ordering of the bars for all the datasets, but python dictionaries do not necessarily preserve the order in which they present their elements
        for datasetname in datasets:
            data.append(D_dict[methodname][datasetname])
        ax.bar(barloc[methodname], data, width=barwidth, edgecolor='white', label=methodname, color = colormap[methodname])

    ax.set_xlabel('Sampling Algorithm', fontsize = 15)
    ax.set_xticks([r + barwidth for r in range(5)])
    ax.set_xticklabels(datasets)
    ax.set_ylabel('D-Statistics for the {}'.format(distname))
    ax.set_ylim((0,1.2))
    ax.grid(ls = '--')
    ax.legend()

    plt.savefig(savepath)
    plt.close()

def PlotRMSEHist(savepath, rmse_dict, colormap):
    fig, ax = plt.subplots(1, 1, figsize= (8, 6))
    # Set width of bar
    barwidth = 0.2

    # Create bar position
    barloc = {}
    barloc0 = np.arange(5)
    for methodname in rmse_dict.keys() : 
        barloc[methodname] = barloc0
        barloc0 = [loc + barwidth for loc in barloc0]

    # Get the datasetnames
    datasets = list(rmse_dict[list(rmse_dict.keys())[0]].keys())
    # Make the plot
    for methodname in rmse_dict.keys():
        data = []
        # This is done because we want a fixed ordering of the bars for all the datasets, but python dictionaries do not necessarily preserve the order in which they present their elements
        for datasetname in datasets:
            data.append(rmse_dict[methodname][datasetname])
        ax.bar(barloc[methodname], data, width=barwidth, edgecolor='white', label=methodname, color = colormap[methodname])

    ax.set_xlabel('Sampling Algorithm', fontsize = 15)
    ax.set_xticks([r + barwidth for r in range(5)])
    ax.set_xticklabels(datasets)
    ax.set_ylabel("RMSE of the top 10 Eigencentrality Scores")
    ax.grid(ls = '--')
    ax.legend()

    plt.savefig(savepath)
    plt.close()

def SaveEigCentTable(savepath, table):
    row, _ = table.shape
    with open(savepath, 'w') as f:
        f.write('{:8s}\t{:8s}\n'.format("Node ID", "Score"))
        f.write('--------------------\n')
        for i  in range(row):
            f.write('{:8d}\t{:8.7f}\n'.format(int(table[i,0]), table[i,1]))

    print('Successfully saved to {}'.format(savepath))

def ImportTableAsSnap(filepath, directed = False):
    # Does not use the LoadEdgeListStr function of the SNAP library, because that function discards the original naming of the nodes
    type_ = "dir" if directed else "undir"
    sg = SimpleGraph(type_)

    sg.LoadFromTable(filepath)

    return sg.ToSnap()
 
