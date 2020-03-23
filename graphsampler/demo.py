import matplotlib.pyplot as plt
from graphsampler.GraphIO import SimpleGraph
from graphsampler.Sampling import ForestFire, SimpleRW, UniformNodeRW, DegreePropRW, MaxEntropyRW
from graphsampler.Analysis import NodeDegDist

def demo_main():
    # Welcome!
    print("""This is a demo for the package "graphsampler". In this demo, we will:\n""")
    print("1) Import the arXiv cond-mat dataset\n")
    print("2) Perform maximal entropy random walk sampling and forest fire sampling to sample 20%% of its nodes\n")
    print("3) Plot the node-degree distribution for the original graph and the sampled graphs\n")
    print("-----------------------------------------------------------------------------------\n")
    
    cond_mat_path = "./graphsampler/datasets/ca-CondMatGCC.txt"
    
    # Create a SimpleGraph instance and load data from table
    print("1)\n")
    cond_mat = SimpleGraph()
    cond_mat.LoadFromTable(cond_mat_path)
    print("-----------------------------------------------------------------------------------\n")

    # Sample 20% of the graph using MERW with 200 relaxation steps at the beginning
    print("2)\n")
    print("Starting maximal entropy random walk sampling...\n")
    merw_adpt = MaxEntropyRW(cond_mat, 0.2, 200, adaptive = True)
    merw_adpt_sampled = merw_adpt()
    print("\n")

    # Sample 20% of the graph using FF
    print("Starting forest fire sampling...\n")
    ff = ForestFire(cond_mat, 0.2)
    ff_sampled = ff()
    print("-----------------------------------------------------------------------------------\n")


    print("3)\n")
    print("Remember: Close the figure to return to shell!\n")
    dist_true = NodeDegDist(cond_mat.ToSnap())
    dist_merw = NodeDegDist(merw_adpt_sampled.ToSnap())
    dist_ff = NodeDegDist(ff_sampled.ToSnap())

    fig, ax = plt.subplots(1, 1, figsize = (7, 5))
    ax.plot(dist_true[0,:], dist_true[1,:], 'k', label = 'True')
    ax.plot(dist_merw[0,:], dist_merw[1,:], 'b', label = 'Maximal Entropy')
    ax.plot(dist_ff[0,:], dist_ff[1,:], 'r', label = 'Forest Fire')
    ax.grid(ls = '--')
    ax.set_xscale('log')
    ax.set_xlabel('Degree')
    ax.set_ylabel('Frequency') 
    ax.set_title('Node-Degree Distributions for the Original and Sampled Graphs')
    ax.legend()
    plt.show()

    plt.close(fig = fig)


if __name__ == "__main__":
    demo_main()