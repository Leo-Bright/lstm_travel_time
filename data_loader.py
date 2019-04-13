import time

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np
import ujson as json


class MySet(Dataset):
    def __init__(self, travel_file, embedding_file):

        self.osmid2emb = extract_embeddings(embedding_file)
        self.travels = open(travel_file, 'r').readlines()
        self.travels = [travel.strip() for travel in self.travels]

    def __getitem__(self, idx):
        travel = self.travels[idx]
        nodes_time = travel.split(" ")
        nodes = nodes_time[:-1]
        time = nodes_time[-1]
        osmid2emb = self.osmid2emb
        sample = {"travel":[]}
        zero_list = [0 for i in range(128)]
        for node in nodes:
            if node not in osmid2emb:
                continue
            id_embedding = osmid2emb[node]
            _emb = [float(ele) for ele in id_embedding]
            sample["travel"].append(_emb)
        while len(sample["travel"]) < 1000:
            tmp_zero = zero_list.copy()
            sample["travel"].append(tmp_zero)
        sample["time"] = time
        return sample

    def __len__(self):
        return len(self.travels)


def extract_embeddings(embeddings_file):
    osmid2embeddings = {}
    with open(embeddings_file, 'r') as emb_file:
        for line in emb_file:
            line = line.strip()
            osmid_vector = line.split(' ')
            osmid, node_vec = osmid_vector[0], osmid_vector[1:]
            if len(node_vec) < 10:
                continue
            osmid2embeddings[osmid] = node_vec
    return osmid2embeddings


def collate_fn(data):
    stat_attrs = ['dist', 'time']
    info_attrs = ['driverID', 'dateID', 'weekID', 'timeID']
    traj_attrs = ['lngs', 'lats', 'states', 'time_gap', 'dist_gap']

    attr, traj = {}, {}

    lens = np.asarray([len(item['lngs']) for item in data])

    for key in stat_attrs:
        x = torch.FloatTensor([item[key] for item in data])
        attr[key] = utils.normalize(x, key)

    for key in info_attrs:
        attr[key] = torch.LongTensor([item[key] for item in data])

    for key in traj_attrs:
        # pad to the max length
        seqs = np.asarray([item[key] for item in data])
        mask = np.arange(lens.max()) < lens[:, None]
        padded = np.zeros(mask.shape, dtype = np.float32)
        padded[mask] = np.concatenate(seqs)

        if key in ['lngs', 'lats', 'time_gap', 'dist_gap']:
            padded = utils.normalize(padded, key)

        padded = torch.from_numpy(padded).float()
        traj[key] = padded

    lens = lens.tolist()
    traj['lens'] = lens

    return attr, traj


class BatchSampler:
    def __init__(self, dataset, batch_size):
        self.count = len(dataset)
        self.batch_size = batch_size
        self.lengths = dataset.lengths
        self.indices = range(self.count)

    def __iter__(self):
        '''
        Divide the data into chunks with size = batch_size * 100
        sort by the length in one chunk
        '''
        np.random.shuffle(self.indices)

        chunk_size = self.batch_size * 100

        chunks = (self.count + chunk_size - 1) // chunk_size

        # re-arrange indices to minimize the padding
        for i in range(chunks):
            partial_indices = self.indices[i * chunk_size: (i + 1) * chunk_size]
            partial_indices.sort(key = lambda x: self.lengths[x], reverse = True)
            self.indices[i * chunk_size: (i + 1) * chunk_size] = partial_indices

        # yield batch
        batches = (self.count - 1 + self.batch_size) // self.batch_size

        for i in range(batches):
            yield self.indices[i * self.batch_size: (i + 1) * self.batch_size]

    def __len__(self):
        return (self.count + self.batch_size - 1) // self.batch_size


def get_loader(input_file, batch_size):
    dataset = MySet(input_file = input_file)

    batch_sampler = BatchSampler(dataset, batch_size)

    data_loader = DataLoader(dataset = dataset, \
                             batch_size = 1, \
                             collate_fn = lambda x: collate_fn(x), \
                             num_workers = 4,
                             batch_sampler = batch_sampler,
                             pin_memory = True
    )

    return data_loader
