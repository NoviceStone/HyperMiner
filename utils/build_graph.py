import numpy as np
import pickle

from nltk.corpus import wordnet as wn
from time import time
from utils.data_util import TextDataset


datafile = '../data/20NG/20news_groups.pkl'
dataname = '20ng'
mode = 'train'

print('==> Loading dataset...')
t0 = time()
entire_dataset = TextDataset(dataname, datafile, mode)
vocabulary = entire_dataset.vocab
vocab_size = len(vocabulary)
del entire_dataset
print("Done in %0.3fs." % (time() - t0))


print('\n==> Extracting a subgraph from WordNet...')
t0 = time()
max_depth = 5
graph_from_wordnet = dict()
for word in vocabulary:
    try:
        leaf_node = wn.synset(word + '.n.01')
        path_to_root = leaf_node.hypernym_paths()[0]

        if len(path_to_root) > max_depth:
            end_idx = max_depth
        else:
            end_idx = -1

        for layer_id, concept in enumerate(path_to_root[: end_idx]):
            layer_name = 'layer_' + str(layer_id)
            if layer_name not in graph_from_wordnet.keys():
                graph_from_wordnet[layer_name] = []
            if concept not in graph_from_wordnet[layer_name]:
                graph_from_wordnet[layer_name].append(concept)
    except:
        pass
print("Done in %0.3fs." % (time() - t0))


num_topics_list = []
for (k, v) in graph_from_wordnet.items():
    num_topics_list.append(len(v))


print('\n==> Generating the adjacency matrix...')
t0 = time()
total_nodes = sum(num_topics_list) + vocab_size
adj_mat = np.eye(total_nodes).astype('float32')
all_concepts = list(graph_from_wordnet.values())[0]
for concept_group in list(graph_from_wordnet.values())[1:]:
    all_concepts = all_concepts + concept_group
for m, concept in enumerate(all_concepts):
    if concept.hypernyms():
        n = all_concepts.index(concept.hypernyms()[0])
        adj_mat[m, n] = 1.0
        adj_mat[n, m] = 1.0
for word in vocabulary:
    m = m + 1
    try:
        leaf_node = wn.synset(word + '.n.01')
        path_to_root = leaf_node.hypernym_paths()[0]
        if len(path_to_root) > max_depth:
            parent_node = path_to_root[max_depth - 1]
        else:
            parent_node = path_to_root[-2]

        n = all_concepts.index(parent_node)
        adj_mat[m, n] = 1.0
        adj_mat[n, m] = 1.0
    except:
        pass
print("Done in %0.3fs." % (time() - t0))


taxonomy = dict()
for (k, v) in graph_from_wordnet.items():
    new_k = k.split('_')[-1]
    new_v = [concept.name() for concept in v]
    taxonomy[new_k] = new_v


print('\n==> Saving knowledge graph to .pkl file...')
t0 = time()
with open('../data/20NG/{}_wordnet_tree_3layer.pkl'.format(dataname), 'wb') as f:
    pickle.dump([adj_mat[3:, 3:], num_topics_list[::-1][:-2], taxonomy], f)
print("Done in %0.3fs." % (time() - t0))
