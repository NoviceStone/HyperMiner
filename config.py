import argparse
from utils.train_util import add_flags_from_config

config_args = {
    'data_config': {
        'seed': (2022, 'manual random seed'),
        'cuda': (0, 'which cuda device to use'),
        'dataset': ('20ng', 'which dataset to use'),
        'data-path': ('./data/20NG/20news_groups.pkl', 'path to load data'),
        'batch-size': (200, 'number of examples in a mini-batch'),
    },
    'training_config': {
        'epochs': (300, 'number of epochs to train'),
        'lr': (0.01, 'initial learning rate'),
        'lr-reduce-freq': (None, 'reduce lr every lr-reduce-freq or None to keep lr constant'),
        'gamma': (0.1, 'gamma for lr scheduler'),
        'dropout': (0, 'dropout probability'),
        'momentum': (0.999, 'momentum in optimizer'),
        'weight-decay': (1e-5, 'l2 regularization strength'),
        'optimizer': ('Adam', 'which optimizer to use, can be any of [Adam, RiemannianAdam]'),
        'grad-clip': (None, 'max norm for gradient clipping, or None for no gradient clipping'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (20, 'how often to compute val metrics (in epochs)'),
        'save': (1, '1 to save model and logs and 0 otherwise'),
        'save-dir': (None, 'path to save training logs and model weights (defaults to logs/task/date/run/)'),
        'sweep-c': (0, ''),
        'print-epoch': (True, ''),
    },
    'model_config': {
        # topic model
        'vocab-size': (8581, 'vocabulary size'), 
        'embed-size': (50, 'dimensionality of word and topic embeddings'), 
        'num-topics-list': ([560, 325, 83, 12, 2], 'number of topics in each latent layer'), 
        'num-hiddens-list': ([300, 300, 300, 300, 300], 'number of units in each hidden layer'),
        'pretrained-embeddings': (False, 'whether to use pretrained embeddings to initialize words and topics'),
        'manifold': ('PoincareBall', 'which manifold to use, can be any of [Euclidean, Hyperboloid, PoincareBall]'),
        'c': (-0.01, 'hyperbolic radius, set to None for trainable curvature'),
        'clip_r': (None, 'avoid the vanishing gradients problem'),
        # hyperbolic gcn
        'add-knowledge': (False, 'whether inject prior knowledge to topic modeling'),
        'file-path': ('./data/20NG/20ng_wordnet_tree_5layers.pkl', 'path to load tree knowledge'),
        'act': ('relu', 'which activation function to use (or None for no activation)'),
        'double-precision': ('0', 'whether to use double precision'),
    },
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
