import matplotlib.pyplot as plt
from graphsampler.GraphIO import SimpleGraph
from graphsampler.Sampling import ForestFire, SimpleRW, UniformNodeRW, DegreePropRW, MaxEntropyRW
from graphsampler.Analysis import NodeDegDist, DegClustDist, NodeMaxhopDist, RankEigvecDist, TopKRankEigvalDist, D_Statistic, PlotDist, PlotDHist, TopKEigCent, SaveEigCentTable, PlotRMSEHist, ImportTableAsSnap
import os
import numpy as np

datasets =  {'ca-CondMat': "./graphsampler/datasets/ca-CondMatGCC.txt", 'AS': "./graphsampler/datasets/as20000102.txt", 'Barabási–Albert': "./graphsampler/datasets/BA(10000,10).txt", 'Erdős–Rényi': "./graphsampler/datasets/ER(10000,100000).txt", 'RoadNet': "./graphsampler/datasets/roadNet-CA-20k.txt"}
sample_ratio = 0.2
seed = 100
distributions = {"Degree Distribution": NodeDegDist, "Degree-Clustering Coefficient Distribution": DegClustDist, "Maxium Hop Distribution": NodeMaxhopDist, "Rank-Eigenvector Distribution": RankEigvecDist, "Top 50 Rank-Eigenvalue Distribution": lambda graph: TopKRankEigvalDist(graph, 50)}
colormap = {"True": "blue", "FF":"red", "GRW":"orange", "MHRW":"green", "MERW":"purple"}
def GenerateSamples(savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    for dataname, filepath in datasets.items():
        # Create simple graph instance to load data
        graph = SimpleGraph()
        graph.LoadFromTable(filepath)
        samplemethod = {"FF": ForestFire(graph, sample_ratio, 0.7, seed), "GRW": SimpleRW(graph, sample_ratio, 0, 0.15, seed), "MHRW": UniformNodeRW(graph, sample_ratio, 1000, seed), "MERW": MaxEntropyRW(graph, sample_ratio, 1000, True, seed)}
        print("{} dataset loaded\n".format(dataname))
           
        for methodname, method in samplemethod.items():
            sample = method()
            savepath = os.path.join(savedir, "{}_{}.txt".format(dataname, methodname))
            sample.SaveToTable(savepath)
            print("{} sampled result saved to {}\n".format(methodname, savepath))

    print("Sampling finished!!!\n")

def ComputeDistributions(sampledir, savedir):
    if not os.path.exists(savedir):
        os.makedirs(savedir)

    methods = ["FF", "GRW", "MHRW", "MERW"]
    # Create a nested dictionary for the distributions of the form dist_dict[distname][datasetname][methodname] = (distname) distribution of the (datasetname) dataset sampled with (methodname) method
    dist_dict = {}
    for distname, distfunc in distributions.items():
        dist_dict[distname] = {}
        
        for datasetname, filepath in datasets.items():
            dist_dict[distname][datasetname] = {}
            dist_dict[distname][datasetname]['True'] = distfunc(ImportTableAsSnap(filepath))

            for method in methods:
                samplepath = os.path.join(sampledir, datasetname+"_"+method+".txt")
                dist_dict[distname][datasetname][method] = distfunc(ImportTableAsSnap(samplepath))

    print("All distributions computed. Now creating figures\n")

    labelmap = {"Degree Distribution": ("Degree", "Counts"), "Degree-Clustering Coefficient Distribution": ("Degree", "Clustering Coefficient"), "Maxium Hop Distribution": ("Maximum Hop", "Counts"), "Rank-Eigenvector Distribution": ("Rank", "1st Eigenvector Component"), "Top 50 Rank-Eigenvalue Distribution": ("Rank", "abs(Eigenvalue)")}
    for distname in distributions.keys():
        for datasetname in datasets.keys():
            figpath = os.path.join(savedir, "{}_{}.png".format(datasetname, distname.replace(" ", "_")))
            PlotDist(figpath, distname, dist_dict[distname][datasetname], datasetname, labelmap, colormap)

    print("Distribution plots generated. Now computing the D-statistics\n")

    # Create the nested dictionary of D-statistics of the dorm D_dict[distname][methodname][datasetname] = D-statistic of the distributions (datasetname) data sampled with (methodname) method against the original distribution
    D_dict = {}
    for distname in distributions.keys():
        D_dict[distname] = {}
        
        for methodname in methods:
            D_dict[distname][methodname] = {}

            for datasetname in datasets.keys():
                D_dict[distname][methodname][datasetname] = D_Statistic(dist_dict[distname][datasetname]['True'], dist_dict[distname][datasetname][methodname])

    print("Computation finished. Now generating histograms\n")
    
    for distname in distributions.keys():
        histpath = os.path.join(savedir, "{}_DHistogram.png".format(distname.replace(" ", "_")))
        PlotDHist(histpath, distname, D_dict[distname], colormap)

    print("D-statistics histgrams drawn. Now the top 10 eigencentrality scores for each of the datasets as well as the corresponding sampled graphs will be calculated and saved.\n")

    # Create a nested dictionary of the form eigcent_dict[datasetname][methodname] = top 10 eigencentrality score table of the (datasetname) dataset sampled with (methodname) method
    eigcent_dict = {}
    for datasetname, filepath in datasets.items():
            eigcent_dict[datasetname] = {}
            table = TopKEigCent(ImportTableAsSnap(filepath), 10)
            eigcent_dict[datasetname]['True'] = table
            SaveEigCentTable(os.path.join(savedir, "EigCentScore_{}_{}.txt".format(datasetname, 'True')), table)

            for method in methods:
                samplepath = os.path.join(sampledir, datasetname+"_"+method+".txt")
                table = TopKEigCent(ImportTableAsSnap(samplepath), 10)
                eigcent_dict[datasetname][method] = table
                SaveEigCentTable(os.path.join(savedir, "EigCentScore_{}_{}.txt".format(datasetname, method)), table)

    print("Tables drawn and saved. Finally calculating the rmse for the eigencentrality scores and generating corresponding histograms.\n")
    
    # Create a nested dictionary of the form rmse_dict[methodname][datasetname] = rmse of the top 10 eigencentrality score for the (datasetname) dataset sampled with (methodname) method and that of the original graph
    rmse_dict = {}
    for method in methods:
        rmse_dict[method] = {}
        for datasetname in datasets.keys():
            rmse_dict[method][datasetname] = np.sqrt((eigcent_dict[datasetname][method][:,1] - eigcent_dict[datasetname]['True'][:,1])**2).mean()
    rmsepath = os.path.join(savedir, "EigCentRMSEHist.png")
    PlotRMSEHist(rmsepath, rmse_dict, colormap)

    print("All finished!\n The original datasets are at: {}\n The sampled graphs are at: {}\n And the generated figures can be found at: {}".format("./datasets", sampledir, savedir))

def main():
    pass
    
if __name__ == "__main__":
    resultdir = "./graphsampler/results"
    print("""The results will be stored in the directory {}\n""".format(resultdir))
    # Create the results directory if not exists
    if not os.path.exists(resultdir):
        os.makedirs(resultdir)

    savedir = os.path.join(resultdir, "samples")
    figdir = os.path.join(resultdir, "figures")

    GenerateSamples(savedir)
    ComputeDistributions(savedir, figdir)