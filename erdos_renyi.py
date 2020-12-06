# Import Required modules 
import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
from scipy import stats
from statistics import mean
import seaborn as sns
import pandas as pd
import matplotlib as mpl
from networkx.algorithms import degree_centrality as DC, eigenvector_centrality as EC
from networkx.algorithms import bridges, local_reaching_centrality as TCC, percolation_centrality as LAPC
from networkx.algorithms import betweenness_centrality as BC, communicability_betweenness_centrality as CBC
from networkx.algorithms import edge_current_flow_betweenness_centrality as RWBC, information_centrality as IC
from networkx.algorithms import subgraph_centrality as SC, communicability_betweenness_centrality as CBC
from networkx.algorithms import katz_centrality as KC, pagerank as PR, closeness_centrality as CC

def erdos_renyi(G, P):
    # Add edges to the graph randomly.
    for i in g.nodes():
        for j in g.nodes():
            if (i < j):

                # Take random number R.
                R = random.random()

                # Check if R<P add the edge to the graph else ignore.
                if (R < P):
                    g.add_edge(i, j, weight=random.randint(1, 20))

def create_unweighted_random_graph(G, P):
    for i in g.nodes():
        for j in g.nodes():
            if (i < j):

                # Take random number R.
                R = random.random()

                # Check if R<P add the edge to the graph else ignore.
                if (R < P):
                    g.add_edge(i, j)


def find_correlation(measurement_list, correlation_lists_list):
    for i in measurement_list:
        correlation_list = []
        for j in i:
            for k in i:
                if(j < k):
                    a, b = stats.spearmanr(j, k)
                    correlation_list.append(a)
        correlation_lists_list.append(correlation_list)


def create_data_set(measurement_list, data_for_mean, data_for_std):
    for i in measurement_list:
        correlation_list = []
        for j in i:
            print("j ", i.index(j))
            for k in i:
                print("k ", i.index(k))
                print()
                a, b = stats.spearmanr(j, k)
                correlation_list.append(a)
                mean_correlation = sum(correlation_list) / len(correlation_list)
                std = np.std(correlation_list)
                data_for_std[i.index(j)][i.index(k)] = std
                data_for_mean[i.index(j)][i.index(k)] = mean_correlation
        print(correlation_list)



# Take N number of nodes from user 
print("Enter number of nodes")
N = int(input())

# Take P probability value for edges
print("Enter value of probability of every node")
P = float(input())

listOfWeightedGraphs = []
listOfUnweightedGraphs = []


for i in range(0, 20):
    # Create an empty graph object for erdos renyi algorithm
    g = nx.Graph()
    # Adding nodes
    g.add_nodes_from(range(1, N + 1))
    erdos_renyi(g, P)
    listOfWeightedGraphs.append(g)

    #create new graph for maslov snepen
    g1 = nx.Graph()
    g1.add_nodes_from(range(1, N + 1))
    create_unweighted_random_graph(g1, P)
    g1 = nx.random_reference(g, 1, True, 1)
    listOfUnweightedGraphs.append(g1)

# plot created weighted and unweighted graphs respectively.
nx.draw(g, with_labels=1)
plt.title("Example visualization of Erdos-Renyi graph")
plt.show()

nx.draw(g1, with_labels=1)
plt.title("Example visualization of Maslov-Snepen graph")
plt.show()

list_of_weighted_cms_lists = []
list_of_unweighted_cms_lists = []

# Centrality measurements for weighted graphs
for i in listOfWeightedGraphs:
    centrality_measurements = []
    centrality_measurements.append(list(nx.degree_centrality(i).values()))
    centrality_measurements.append(list(nx.eigenvector_centrality(i).values()))
    centrality_measurements.append(list(nx.betweenness_centrality(i).values()))
    centrality_measurements.append(list(nx.subgraph_centrality(i).values()))
    centrality_measurements.append(list(nx.katz_centrality_numpy(i).values()))
    centrality_measurements.append(list(nx.pagerank(i).values()))
    centrality_measurements.append(list(nx.closeness_centrality(i).values()))
    #centrality_measurements.append(list(nx.current_flow_betweenness_centrality(i).values()))
    #centrality_measurements.append(list(nx.communicability_betweenness_centrality(i).values()))
    #centrality_measurements.append(list(nx.average_neighbor_degree(i).values()))
    list_of_weighted_cms_lists.append(centrality_measurements)

# Centrality measurements for unweighted graphs
for i in listOfUnweightedGraphs:
    centrality_measurements = []
    centrality_measurements.append(list(nx.degree_centrality(i).values()))
    centrality_measurements.append(list(nx.eigenvector_centrality(i).values()))
    centrality_measurements.append(list(nx.betweenness_centrality(i).values()))
    centrality_measurements.append(list(nx.subgraph_centrality(i).values()))
    centrality_measurements.append(list(nx.katz_centrality_numpy(i).values()))
    centrality_measurements.append(list(nx.pagerank(i).values()))
    centrality_measurements.append(list(nx.closeness_centrality(i).values()))
    #centrality_measurements.append(list(nx.current_flow_betweenness_centrality(i).values()))
    #centrality_measurements.append(list(nx.communicability_betweenness_centrality(i).values()))
    #centrality_measurements.append(list(nx.average_neighbor_degree(i).values()))
    list_of_unweighted_cms_lists.append(centrality_measurements)

list_of_weighted_correlations_lists = []
list_of_unweighted_correlations_lists = []

find_correlation(list_of_weighted_cms_lists, list_of_weighted_correlations_lists)

#plot figure 2 for weighted graphs
ax = sns.violinplot(data=list_of_weighted_correlations_lists, inner=None, color=".8")
ax = sns.stripplot(data=list_of_weighted_correlations_lists)
plt.xlabel("Weighted networks")
plt.ylabel("Centrality Measure Correlations")
plt.show()

find_correlation(list_of_unweighted_cms_lists, list_of_unweighted_correlations_lists)

#plot figure 2 for unweighted graphs
ax = sns.violinplot(data=list_of_unweighted_correlations_lists, inner=None, color=".8")
ax = sns.stripplot(data=list_of_unweighted_correlations_lists)
plt.xlabel("Unweighted networks")
plt.ylabel("Centrality Measure Correlations")
plt.show()

# plot figure 3 for weighted graphs
centrality_measurement_count = len(list_of_weighted_cms_lists[0])
data_for_mean = [ [ 0 for i in range(centrality_measurement_count) ] for j in range(centrality_measurement_count) ]
data_for_std = [ [ 0 for i in range(centrality_measurement_count) ] for j in range(centrality_measurement_count) ]

create_data_set(list_of_weighted_cms_lists, data_for_mean, data_for_std)

#mean
ax = sns.heatmap(data=data_for_mean, vmin=-1, vmax=1)
plt.title("Mean Spearman correlation across weighted networks")
plt.show()

#std
ax = sns.heatmap(data=data_for_std, vmin=0, vmax=0.5)
plt.title("Spearman correlation standard deviation across weighted networks")
plt.show()

# plot figure 3 for unweighted graphs
create_data_set(list_of_unweighted_cms_lists, data_for_mean, data_for_std)

#mean
ax = sns.heatmap(data=data_for_mean, vmin=-1, vmax=1)
plt.title("Mean Spearman correlation across unweighted networks")
plt.show()

#std
ax = sns.heatmap(data=data_for_std, vmin=0, vmax=0.5)
plt.title("Spearman correlation standard deviation across unweighted networks")
plt.show()

#plot figure 4 for weighted graphs
ax = sns.barplot(data=list_of_weighted_correlations_lists)
plt.xlabel("Weighted")
plt.ylabel("Centrality Measure Correlations")
plt.show()

#plot figure 4 for unweighted graphs
ax = sns.barplot(data=list_of_unweighted_correlations_lists)
plt.xlabel("Unweighted")
plt.ylabel("Centrality Measure Correlations")
plt.show()

#plot figure 5
mean_list_for_weighted = []
mean_list_for_unweighted = []
assortativity_list_weighted = []
clustering_list_weighted = []
assortativity_list_unweighted = []
clustering_list_unweighted = []


for i in listOfWeightedGraphs:
    assortativity_list_weighted.append(nx.degree_assortativity_coefficient(i, weight='weight'))
    clustering_list_weighted.append(nx.average_clustering(i))

for i in listOfUnweightedGraphs:
    assortativity_list_unweighted.append(nx.degree_assortativity_coefficient(i))
    clustering_list_unweighted.append(nx.average_clustering(i))

for i in list_of_weighted_correlations_lists:
    mean = sum(i) / len(i)
    mean_list_for_weighted.append(mean)

for i in list_of_unweighted_correlations_lists:
    mean = sum(i) / len(i)
    mean_list_for_unweighted.append(mean)

plt.scatter(assortativity_list_weighted, mean_list_for_weighted, c='blue')
plt.scatter(assortativity_list_unweighted, mean_list_for_unweighted, c='red')
plt.xlabel("Assortativity")
plt.ylabel("Mean within-network CMC")
plt.show()

plt.scatter(clustering_list_weighted, mean_list_for_weighted, c='blue')
plt.scatter(clustering_list_unweighted, mean_list_for_unweighted, c='red')
plt.xlabel("Clustering")
plt.ylabel("Mean within-network CMC")
plt.show()