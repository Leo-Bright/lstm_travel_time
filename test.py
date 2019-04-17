from data_loader import MySet
from torch.utils.data import Dataset, DataLoader


dataset = MySet('sf_trajectory_node_travel_time_450.travel', 'sf_random_node2vec_d128_wl1280.embedding')

data_loader = DataLoader(dataset = dataset, \
                         batch_size = 64, \
                         # collate_fn = lambda x: collate_fn(x), \
                         # num_workers = 4,
                         # batch_sampler = batch_sampler,
                         # pin_memory = True
                        )

for idx, sample in enumerate(data_loader):
    print(sample["travel"])
    print(len(sample["travel"]))