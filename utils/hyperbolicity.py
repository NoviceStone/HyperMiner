import networkx as nx
import numpy as np
import pickle
import time
from tqdm import tqdm


def hyperbolicity_sample(G, num_samples=50000):
    curr_time = time.time()
    hyps = []
    for _ in tqdm(range(num_samples)):
        curr_time = time.time()
        node_tuple = np.random.choice(G.nodes(), 4, replace=False)
        s = []
        try:
            d01 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[1], weight=None)
            d23 = nx.shortest_path_length(G, source=node_tuple[2], target=node_tuple[3], weight=None)
            d02 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[2], weight=None)
            d13 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[3], weight=None)
            d03 = nx.shortest_path_length(G, source=node_tuple[0], target=node_tuple[3], weight=None)
            d12 = nx.shortest_path_length(G, source=node_tuple[1], target=node_tuple[2], weight=None)
            s.append(d01 + d23)
            s.append(d02 + d13)
            s.append(d03 + d12)
            s.sort()
            hyps.append((s[-1] - s[-2]) / 2)
        except Exception as e:
            continue
    print('Time for hyp:', time.time() - curr_time)
    return max(hyps)


if __name__ == '__main__':
    kg_path = '../data/20news/20ng_wordnet_tree.pkl'

    print('==> Loading graph...')
    with open(kg_path, 'rb') as f:
        adj, n_nodes_per_layer, concepts = pickle.load(f)
    adj = adj - np.eye(adj.shape[0])

    graph = nx.from_numpy_array(adj)
    print(f'Number of nodes: {graph.number_of_nodes()}')
    print(f'Number of edges: {graph.number_of_edges()}')
    print('\n==> Computing hyperbolicity...')
    hyp = hyperbolicity_sample(graph)
    print('Hyp:', hyp)
