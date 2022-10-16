import argparse
import numpy as np
import os
from time import time
import scipy.sparse as sp
# import torch
# import torch.nn.functional as F
# import torch.nn.modules.loss
import torch

from .eval_util import topic_diversity


def format_metrics(metrics, split):
    """Format metric in metric dict for logging."""
    return " ".join([
        "{}_{}: {:.4f}".format(split, metric_name, metric_val) for metric_name, metric_val in metrics.items()
    ])


def get_dir_name(models_dir):
    """Gets a directory to save the model.

    If the directory already exists, then append a new integer to the end of
    it. This method is useful so that we don't overwrite existing models
    when launching new jobs.

    Args:
        models_dir: The directory where all the models are.

    Returns:
        The name of a new directory to save the training logs and model weights.
    """
    if not os.path.exists(models_dir):
        save_dir = os.path.join(models_dir, '0')
        os.makedirs(save_dir)
    else:
        existing_dirs = np.array(
                [
                    d
                    for d in os.listdir(models_dir)
                    if os.path.isdir(os.path.join(models_dir, d))
                    ]
        ).astype(np.int)
        if len(existing_dirs) > 0:
            dir_id = str(existing_dirs.max() + 1)
        else:
            dir_id = "1"
        save_dir = os.path.join(models_dir, dir_id)
        os.makedirs(save_dir)
    return save_dir


def add_flags_from_config(parser, config_dict):
    """Adds a flag (and default value) to an ArgumentParser for each parameter in a config.
    """

    def OrNone(default):
        def func(x):
            # Convert "none" to proper None object
            if x.lower() == "none":
                return None
            # If default is None (and x is not None), return x without conversion as str
            elif default is None:
                return str(x)
            # Otherwise, default has non-None type; convert x to that type
            else:
                return type(default)(x)

        return func

    for param in config_dict:
        default, description = config_dict[param]
        try:
            if isinstance(default, dict):
                parser = add_flags_from_config(parser, default)
            elif isinstance(default, list):
                if len(default) > 0:
                    # pass a list as argument
                    parser.add_argument(
                            f"--{param}",
                            action="append",
                            type=type(default[0]),
                            default=default,
                            help=description
                    )
                else:
                    pass
                    parser.add_argument(f"--{param}", action="append", default=default, help=description)
            else:
                pass
                parser.add_argument(f"--{param}", type=OrNone(default), default=default, help=description)
        except argparse.ArgumentError:
            print(
                f"Could not add flag for param {param} because it was already present."
            )
    return parser


def convert_to_coo_adj(dense_adj):
    """convert the dense adjacent matrix (numpy array) to sparse_coo matrix (torch Tensor).
    """
    dense_adj = dense_adj - np.eye(dense_adj.shape[0])
    coo_mat = sp.coo_matrix(dense_adj)
    edge_weights = coo_mat.data
    edge_indices = np.vstack((coo_mat.row, coo_mat.col))
    return torch.sparse_coo_tensor(
        indices=torch.from_numpy(edge_indices).long(),
        values=torch.from_numpy(edge_weights).float(),
        size=coo_mat.shape
    )


def load_glove_embeddings(embed_size, vocab):
    """Initial word embeddings with pretrained glove embeddings if necessary.
    """
    glove_path = './data/glove/glove.6B.{}d.txt'.format(embed_size)

    print('\n==> Loading pretrained glove embeddings...')
    t0 = time()
    embeddings_dict = dict()
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f.readlines():
            values = line.split()
            word = values[0]
            embedding = np.asarray(values[1:], dtype=np.float32)
            embeddings_dict[word] = embedding
    print("Done in %0.3fs." % (time() - t0))

    print('==> Initialize word embeddings with glove embeddings...')
    t0 = time()
    vocab_embeddings = list()
    for word in vocab:
        try:
            vocab_embeddings.append(embeddings_dict[word])
        except:
            vocab_embeddings.append(0.02 * np.random.randn(embed_size))
    print("Done in %0.3fs." % (time() - t0))

    return np.array(vocab_embeddings, dtype=np.float32)


def get_top_n(phi_column, vocab, top_n=25):
    top_n_words = ''
    indices = np.argsort(-phi_column)
    for n in range(top_n):
        top_n_words += vocab[indices[n]]
        top_n_words += ' '
    return top_n_words


def visualize_topics(phis, save_dir, vocab, top_n=25, concepts=None):
    if isinstance(phis, list):
        phis = [phi.cpu().numpy() for phi in phis]
    else:
        phis = [phis.cpu().numpy()]

    num_layers = len(phis)
    factorial_phi = 1
    for layer_id, phi in enumerate(phis):
        factorial_phi = np.dot(factorial_phi, phi)
        cur_td = topic_diversity(factorial_phi.T, top_n)

        num_topics = factorial_phi.shape[1]
        path = os.path.join(save_dir, 'phi_' + str(layer_id) + '.txt')
        f = open(path, 'w')
        for k in range(num_topics):
            top_n_words = get_top_n(
                factorial_phi[:, k], vocab, top_n)
            if concepts is not None:
                f.write('({})'.format(concepts[str(num_layers - layer_id)][k]))
            f.write(top_n_words)
            f.write('\n')
        f.write('Topic diversity:{}'.format(cur_td))
        f.write('\n')
        f.close()
