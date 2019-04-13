import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.autograd import Variable
import os


emb = 'po_random_1280_128d.emb'

samples_file = 'pt_trajectory_node_travel_time.travel'


# 定义模型
class LstmRegression(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, num_layers=2):
        super(LstmRegression, self).__init__()

        self.rnn = nn.LSTM(input_size, hidden_size, num_layers)  # rnn
        self.reg = nn.Linear(hidden_size, output_size)  # 回归

    def forward(self, x):
        x, _ = self.rnn(x)  # (seq, batch, hidden)
        s, b, h = x.shape
        x = x.view(s * b, h)  # 转换成线性层的输入格式
        x = self.reg(x)
        x = x.view(s, b, -1)
        return x

    def train(self, x, y):
        pass



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


def extract_samples(all_nodes_time, osmid_embeddings):
    zero_list = [0 for i in range(128)]
    for item in all_nodes_time:
        bag_line = False
        sample = []
        for node in item[:-1]:
            if node not in osmid_embeddings:
                bag_line = True
                break
            id_embeddings = osmid_embeddings[node]
            tmp_emb = [float(ele) for ele in id_embeddings]
            sample.append(tmp_emb)
        target = item[-1]
        while len(sample) < 1000:
            tmp_zero = zero_list.copy()
            sample.append(tmp_zero)
        if bag_line:
            continue
        else:
            yield (sample, target)


os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

samples_in_file = []
with open(samples_file, 'r') as sam_file:
    line_count = 0
    for line in sam_file:
        line_count += 1
        line = line.strip()
        nodes_time = line.split(' ')
        length = len(nodes_time)
        if 10 > length or length > 1000:
            continue
        samples_in_file.append(nodes_time)
    print('extract samples from file done: ', samples_file)

node_embeddings = extract_embeddings(emb)

samples_targets = extract_samples(samples_in_file, node_embeddings)

# model = build_model()
model = LstmRegression(128, 100)
# criterion = nn.MAELoss()
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

samples = []
targets = []
test_samples = []
test_targets = []
test_result = []
have_test_result = False

samples_count = len(samples_in_file)
train_num = round(samples_count * 0.9)
train_count = 0
iteration_batch = 128

for epoch in range(10):
    for sample_target in samples_targets:
        (_sample, _target) = sample_target
        train_count += 1
        if train_count <= train_num:
            samples.append(_sample)
            targets.append(_target)
            if len(samples) >= iteration_batch or train_count == train_num:
                assert len(samples) == len(targets)
                print('training samples at: ', train_count)
                x_train = np.array(samples)
                y_train = np.array(targets)
                loss = model.train(x_train, y_train)
                samples = []
                targets = []
        else:
            test_samples.append(_sample)
            test_targets.append(_target)
        if len(test_samples) >= iteration_batch:
            have_test_result = True
            x_test = np.array(test_samples)
            y_test = np.array(test_targets)
            metri = model.evaluate(x_test, y_test)
            print('test result:', metri)
            test_result.append(metri)
            test_samples = []
            test_targets = []

print(emb)
if not have_test_result and len(test_samples) > 0:
    x_test = np.array(test_samples)
    y_test = np.array(test_targets)
    metri = model.evaluate(x_test, y_test)
    print('Finally result:', metri)
else:
    print('mean test loss:', sum(test_result)/len(test_result))
