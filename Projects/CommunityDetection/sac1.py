import sys
import pandas as pd
import numpy as np
from igraph import *
from scipy import spatial


# For fileSuffix graph, and vertex ids, returns cosine similarity
def simA(v1, v2, graph):
  vv1 = graph.vs[v1].attributes().values()
  vv2 = graph.vs[v2].attributes().values()
  return 1 - spatial.distance.cosine(vv1, vv2)


# Implements phase1 of the algorithm.
# Input: graph graph, alpha, and initial clusters assignments.
def phase1(graph, alpha, C):

  Vertices = len(graph.vs)
  m = len(graph.es)
  iter = 0
  check = 0
  #C = range(0,Vertices)
  while (check == 0 and iter < 15):
    check = 1
    for vi in range(Vertices):
      maxV = -1
      maxDeltaQ = 0.0
      clusters = list(set(C))
      for vj in clusters:
        if (C[vi] != vj):
          dQ = DeltaQ(alpha, C, graph, vi, vj)
          if (dQ > maxDeltaQ):
            maxDeltaQ = dQ
            maxV = vj
      if (maxDeltaQ > 0.0 and maxV != -1):
        check = 0
        C[vi] = maxV
    iter = iter + 1
  return C


# Implements phase2 of the algorithm
def phase2 (graph, C):
  newC = sequentialClusters(C)
  temp = list(Clustering(newC))
  L = len(set(newC))
  simMatrix = np.zeros((L,L))
  
  for i in range(L):
    for j in range(L):
      similarity = 0.0
      for k in temp[i]:
        for l in temp[j]:
          similarity = similarity + simMatrix2[k][l]
      simMatrix[i][j] = similarity
  
  graph.contract_vertices(newC)
  graph.simplify(combine_edges=sum)
  return

# makes the clusters sequential. For example, cluster assignments - [2, 2, 4, 4, 5] will become: [0, 0, 1, 1, 2]
def sequentialClusters(C):
  mapping = {}
  newC = []
  c = 0
  for i in C:
    if i in mapping:
      newC.append(mapping[i])
    else:
      newC.append(c)
      mapping[i] = c
      c = c + 1
  return newC

# Calculates change in modularity of graph when v1 is moved to the cluster of v2
def DeltaQNew(C, graph, v1, v2):
  Q1 = graph.modularity(C, weights='weight')
  temp = C[v1]
  C[v1] = v2
  Q2 = graph.modularity(C, weights='weight')
  C[v1] = temp
  return (Q2-Q1);


# Calculates teh change in attribute similarity for the cluster of v2 after addition of v1
# Normalization by dividing it by the number of clusters and number of items in that paritulcar cluster
def DeltaQAttr(C, graph, v1, v2):
  S = 0.0;
  indices = [i for i, x in enumerate(C) if x == v2]
  for Vertices in indices:
    S = S + simMatrix[v1][Vertices]
  return S/(len(indices)*len(set(C)))


# Calculates the total attribute similarity for the complete graph for fileSuffix given clustering
# Normalization: for each cluster divide by cluster size. Overall, divide by number of clusters.
def QAttr(C, graph):
  clusters = list(Clustering(C))
  Vertices = graph.vcount()
  S = 0.0
  for c in clusters:
    T = 0.0
    for v1 in c:
      for v2 in C:
        if (v1 != v2):
          T = T + simMatrix[v1][v2]
    T = T/len(c)
    S = S + T
  return S/(len(set(C)))


def compositeModularity(graph, C):
  return graph.modularity(C, weights='weight') + QAttr(C, graph)


# Calculates overall change in modularity (structural and attribute based) by changing the 
# cluster of vertex v1 to cluster of vertex v2
def DeltaQ(alpha, C, graph, v1, v2):
  d1 = DeltaQNew(C, graph, v1, v2)
  d2 = DeltaQAttr(C, graph, v1, v2)
  return (alpha*d1) + ((1-alpha)*d2)


# writes the clusters to the file
def writeToFile(clusters, alpha):
  file = open("communities_"+alpha+".txt", 'w+')
  for c in clusters:
    for i in range(len(c)-1):
      file.write("%s," % c[i])
    file.write(str(c[-1]))
    file.write('\n')
  file.close()


# checks runtime arguments and accordingly runs the appropriate algorithm
def main(argv):

    alpha = sys.argv[1]
    alpha = float(alpha)
    print ('alpha value is -', alpha)
    
    attributes = pd.read_csv('data/fb_caltech_small_attrlist.csv')

    Vertices = len(attributes)

    with open('data/fb_caltech_small_edgelist.txt') as f:
      edges = f.readlines()

    edges = [tuple([int(x) for x in line.strip().split(" ")]) for line in edges]  

    graph = Graph()
    graph.add_vertices(Vertices)
    graph.add_edges(edges)
    graph.es['weight'] = [1]*len(edges)

    for col in attributes.keys():
      graph.vs[col] = attributes[col]

    # Pre-Computing Similarity Matrix
    global simMatrix
    global simMatrix2
    simMatrix = np.zeros((Vertices,Vertices))
    for i in range(Vertices):
      for j in range(Vertices):
        simMatrix[i][j] = simA(i, j, graph)

    # Create fileSuffix copy. Required Later
    simMatrix2 = np.array(simMatrix)

    # Run the Algorithm
    Vertices = graph.vcount()
    print (Vertices)
    C = phase1(graph, alpha, range(Vertices))
    print('Number of Communities after Phase 1')
    print(len(set(C)))
    C = sequentialClusters(C)
    #Composite modularity of phase 1 clustering
    compModularityPhase1Clustering = compositeModularity(graph, C)

    # Phase 2
    phase2(graph, C)

    # Re-running Phase 1
    Vertices = graph.vcount()
    C2 = phase1(graph, alpha, range(Vertices))
    C2new = sequentialClusters(C2)
    clustersPhase2 = list(Clustering(C2new))
    #Composite modularity of contracted clustering
    compModularityPhase2Clustering = compositeModularity(graph, C)

    # Suffix for the output file based on alpha
    fileSuffix = 0
    if alpha == 0.5:
      fileSuffix = 5
    elif alpha == 0.0:
      fileSuffix = 0
    elif alpha == 1.0:
      fileSuffix = 1

    finalC = []
    C1new = sequentialClusters(C)
    clustersPhase1 = list(Clustering(C1new))

    # Mapping the super clusters from phase 2 to original vertices.
    for cluster in clustersPhase2:
      t = []
      for Vertices in cluster:
        t.extend(clustersPhase1[Vertices])
      finalC.append(t)

    # based on composite modularity, the corresponding clusters are written to output file
    if (compModularityPhase1Clustering > compModularityPhase2Clustering):
      writeToFile(clustersPhase1, str(fileSuffix))
      print ('Phase 1 clusters have higher modularity')
      return clustersPhase1
    else:
      writeToFile(clustersPhase2, str(fileSuffix))
      print ('Phase 2 clusters have higher modularity')
      return clustersPhase2


if __name__ == "__main__":
  if len(sys.argv) != 2:
    print ("Invalid input for alpha")
  else: 
    main(sys.argv[1:])