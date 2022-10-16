"""Data utils functions for pre-processing and data loading."""

import numpy as np
import pickle
import random
import torch.utils.data


class TextDataset(torch.utils.data.Dataset):
    def __init__(self, name, path, mode='train'):
        super(TextDataset, self).__init__()
        with open(path, 'rb') as f:
            data = pickle.load(f)

        if name in ['20ng', 'tmn', 'webs']:
            # vocab = data['vocab']
            # train_bows = data['train_data']
            # train_labels = data['train_labels']
            # test_bows = data['test_data']
            # test_labels = data['test_labels']
            train_id = data['train_id']
            test_id = data['test_id']
            label = np.squeeze(np.array(data['label']))

            train_bows = data['data_2000'][train_id]
            train_labels = label[train_id]
            test_bows = data['data_2000'][test_id]
            test_labels = label[test_id]
            vocab = data['voc2000']
        elif name == 'wiki':
            vocab = data['vocab']
            train_bows = data['data']
            train_labels = None
            test_bows = None
            test_labels = None
        elif name == 'rcv':
            vocab = data['rcv2_voc']
            train_bows = data['rcv2_bow']
            train_labels = None
            test_bows = None
            test_labels = None
        else:
            raise NotImplementedError(f'unknown dataset: {name}')

        if mode == 'train':
            self.data = train_bows
            self.labels = train_labels
        elif mode == 'test':
            self.data = test_bows
            self.labels = test_labels
        else:
            raise ValueError("argument 'mode' must be either train or test")
        self.vocab = vocab

        if self.labels is not None:
            assert self.data.shape[0] == len(self.labels)

    def __getitem__(self, index):
        if self.labels is not None:
            return self.data[index].toarray().squeeze(), self.labels[index]
        else:
            return self.data[index].toarray().squeeze(), 0

    def __len__(self):
        return self.data.shape[0]


class PPLDataset(torch.utils.data.Dataset):
    def __init__(self, name, data_path='./data/20NG/20news_groups.pkl'):
        super(PPLDataset, self).__init__()
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        if name in ['20ng', 'tmn']:
            bows_matrix = np.concatenate(
                data['train_data'].toarray(), data['test_data'].toarray(), axis=0
            )
        else:
            bows_matrix = data['data'].toarray()

        context_ratio = 0.7
        context_bows = np.zeros_like(bows_matrix)
        mask_bows = np.zeros_like(bows_matrix)
        for doc_id, bow in enumerate(bows_matrix):
            indices = np.nonzero(bow)[0]
            indices_rep = []
            for idx in indices:
                indices_rep += bow[idx] * [idx]

            random.seed(2022)
            random.shuffle(indices_rep)
            temp1 = indices_rep[:int(len(indices_rep) * context_ratio)]
            temp2 = indices_rep[int(len(indices_rep) * context_ratio):]
            for word_id in temp1:
                context_bows[doc_id][word_id] += 1
            for word_id in temp2:
                mask_bows[doc_id][word_id] += 1

        self.context_data = context_bows
        self.mask_data = mask_bows
        self.vocab = data['vocab']

    def __getitem__(self, index):
        return torch.from_numpy(self.context_data[index].squeeze()).float(), \
               torch.from_numpy(self.mask_data[index].squeeze()).float()

    def __len__(self):
        return self.context_data.shape[0]


def get_data_loader(data_name, data_path, mode='train', batch_size=200, shuffle=True, drop_last=True, num_workers=4):
    dataset = TextDataset(name=data_name, path=data_path, mode=mode)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    ), dataset.vocab


def get_ppl_dataloader(data_name, data_path, batch_size=200, shuffle=True, drop_last=True, num_workers=4):
    dataset = PPLDataset(data_name, data_path)
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers
    ), dataset.vocab
