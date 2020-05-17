import networkx as nx

G = nx.read_graphml('../../dataset/vnet.graphml')
print G
print type(G)
print G.adj
